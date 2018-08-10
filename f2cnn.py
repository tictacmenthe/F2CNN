import argparse
from os.path import join

from scripts.processing.OrganiseFiles import OrganiseAllFiles
from scripts.processing.GammatoneFiltering import FilterAllOrganisedFiles
from scripts.processing.EnvelopeExtraction import ExtractAllEnvelopes
from scripts.processing.LabelDataGenerator import GenerateLabelData
from scripts.processing.InputGenerator import GenerateInputData
from scripts.plotting.PlottingProcessing import PlotEnvelopesAndF2FromFile
from scripts.CNN.CNN import TrainAndPlotLoss, EvaluateOneFile, EvaluateRandom


def All(testMode):
    """
    Does all the treatments required for the training
    :param testMode:
    """
    if not testMode:
        OrganiseAllFiles(testMode)
    FilterAllOrganisedFiles(testMode)
    ExtractAllEnvelopes(testMode)
    GenerateLabelData(testMode)
    GenerateInputData(testMode)


def main():
    # Dictionaries linking arguments and functions to call
    PREPARE_FUNCTIONS = {
        'all': All,
        'organize': OrganiseAllFiles,
        'filter': FilterAllOrganisedFiles,
        'envelope': ExtractAllEnvelopes,
        'label': GenerateLabelData,
        'input': GenerateInputData
    }

    CNN_FUNCTIONS = {
        'train': TrainAndPlotLoss,
        'eval': EvaluateOneFile,  # Applies the CNN to one specified file
        'evalrand': EvaluateRandom
    }

    PLOT_FUNCTIONS = {
        'gtg': PlotEnvelopesAndF2FromFile
    }
    # Help texts for some argument groups
    preparationHelpText = """Data processing commands:\n\t\
organize:\tOrganizes the files as needed for the rest (Check OrganiseFiles.py documentation)\n\t\
filter:\t\tApplies the GammaTone FilterBank to the organized files and saves its outputs in .GFB.npy format\n\t\
envelope:\tExtracts the filtered files' envelopes and saves them in .ENV1.npy format\n\t\
label:\t\tGenerates Labeling data for the CNN\n\t\
input\t\tGenerates Input data for the CNN, requires label first\n\t\
all:\t\tDoes all of the above, can take some time.
    """
    parser = argparse.ArgumentParser(description="F2CNN Project's entry script.",
                                     epilog="For additional information, add -h after any positional argument")
    # TODO silent
    parser.add_argument('--silent', '-s', action='store_false', help="Remove logs, for mostly silent processing.")
    subparsers = parser.add_subparsers()

    # Parser for data processing purposes
    parser_prepare = subparsers.add_parser('prepare', help='Runs the command given in argument.',
                                           formatter_class=argparse.RawTextHelpFormatter)
    parser_prepare.add_argument('--lpf', '-l', action='store', dest='CUTOFF')
    parser_prepare.add_argument('prepare_command', choices=PREPARE_FUNCTIONS.keys(), help=preparationHelpText)
    parser_prepare.add_argument('--file', '-f', action='store', dest='file', nargs=1)

    # Parser for plotting purposes
    parser_plot = subparsers.add_parser('plot', help='For plotting spectrogram-like figure from .WAV file.')
    parser_plot.add_argument('plot_type', choices=PLOT_FUNCTIONS.keys(),
                             help="gtg: Plots a spectrogram like figure from the output of a GammaTone FilterBank applied\
                                  to the given file, and if a .FB file exists in the dir, also plots the Formants.")
    parser_plot.add_argument('--file', '-f', action='store', dest='file', nargs=1)

    # Parser for the CNN
    parser_cnn = subparsers.add_parser('cnn', help='Commands related to training, testing and using the CNN.',
                                       formatter_class=argparse.RawTextHelpFormatter)
    parser_cnn.add_argument('--file', '-f', action='store', dest='file', nargs=1)
    parser_cnn.add_argument('cnn_command', choices=CNN_FUNCTIONS.keys())

    # Processes the input arguments
    args = parser.parse_args()

    print(args)
    # Calls to functions according to arguments
    if 'prepare_command' in args:
        if args.prepare_command in ['envelope', 'input']:
            PREPARE_FUNCTIONS[args.prepare_command](False if args.CUTOFF is None else True, args.CUTOFF)

        else:
            PREPARE_FUNCTIONS[args.prepare_command]()
    elif 'plot_type' in args:
        if args.file is None:
            print("Please use --file or -f to give input file")
        else:
            print("Plotting for file {}...".format(args.gfilename))
            PlotEnvelopesAndF2FromFile(args.gfilename)

    elif 'cnn_command' in args:
        if args.cnn_command == 'eval':
            if args.file is not None:
                CNN_FUNCTIONS[args.cnn_command](args.file)
            else:
                print("Please use --lpf or -f command to use")
        elif args.cnn_command == 'train':
            inputFile = args.file or join('trainingData','input_data.npy')
            print(inputFile)
            exit()
            CNN_FUNCTIONS[args.cnn_command](inputFile)
        else:
            CNN_FUNCTIONS[args.cnn_command]()


if __name__ == '__main__':
    main()
