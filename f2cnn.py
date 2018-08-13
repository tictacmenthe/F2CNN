import argparse
from os.path import join

from scripts.processing.OrganiseFiles import OrganiseAllFiles
from scripts.processing.GammatoneFiltering import FilterAllOrganisedFiles
from scripts.processing.EnvelopeExtraction import ExtractAllEnvelopes
from scripts.processing.LabelDataGenerator import GenerateLabelData
from scripts.processing.InputGenerator import GenerateInputData
from scripts.plotting.PlottingProcessing import PlotEnvelopesAndF2FromFile
from scripts.CNN.Evaluating import EvaluateOneFile, EvaluateRandom
from scripts.CNN.Training import TrainAndPlotLoss


def All():
    """
    Does all the treatments required for the training
    """
    OrganiseAllFiles()
    FilterAllOrganisedFiles()
    ExtractAllEnvelopes()
    GenerateLabelData()
    GenerateInputData()


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
    preparationHelpText = """Data Processing Commands:\n\t\
organize:\tOrganizes the files as needed for the rest\n\t\t\t(Check OrganiseFiles.py documentation)\n\t\
filter:\t\tApplies the GammaTone FilterBank to the organized files.\n\t\t\tSaves its outputs in .GFB.npy format\n\t\
envelope:\tExtracts the filtered files' envelopes.\n\t\t\tUsing --cutoff CUTOFF as low pass filter cutoff frequency.\n\t\t\tSaves them in .ENV1.npy format\n\t\
label:\t\tGenerates Labeling data for the CNN\n\t\
input\t\tGenerates Input data for the CNN, requires label first\n\t\
all:\t\tDoes all of the above, can take some time.
    """

    cnnHelpText = """CNN Related Commands:\n\t\
train:\tTrains the CNN.\n\t\tUse --file command to give the path to an input data numpy matrix\n\t\tOtherwise, uses the input_data.npy file in trainingData/ directory.\n\t\
eval:\tEvaluates a keras model using one WAV file.\n\t\t
evalrand:\tEvaluates all the .WAV files in resources/f2cnn/* in a random order.\n\t\tMay be interrupted whenever, if needed.
    """
    fileHelpText = "Used to give a file path as an argument to some scripts."

    # #####  PARSING
    parser = argparse.ArgumentParser(description="F2CNN Project's entry script.",
                                     epilog="For additional information, add -h after any positional argument")
    # TODO silent
    parser.add_argument('--silent', '-s', action='store_false', help="Remove logs, for mostly silent processing.")
    subparsers = parser.add_subparsers()

    # Parser for data processing purposes
    parser_prepare = subparsers.add_parser('prepare', help='Runs the command given in argument.',
                                           formatter_class=argparse.RawTextHelpFormatter)
    parser_prepare.add_argument('--cutoff', '-c', action='store', dest='CUTOFF', type=int,
                                help="If used, low pass filter of given argument as cutoff frequency will be used")
    parser_prepare.add_argument('prepare_command', choices=PREPARE_FUNCTIONS.keys(), help=preparationHelpText)
    parser_prepare.add_argument('--file', '-f', action='store', dest='file', nargs='?', help=fileHelpText)

    # Parser for plotting purposes
    parser_plot = subparsers.add_parser('plot', help='For plotting spectrogram-like figure from .WAV file.')
    parser_plot.add_argument('plot_type', choices=PLOT_FUNCTIONS.keys(),
                             help="gtg: Plots a spectrogram like figure from the output of a GammaTone FilterBank applied\
                                  to the given file, and if a .FB file exists in the dir, also plots the Formants.")
    parser_plot.add_argument('--file', '-f', action='store', dest='file', nargs='?', help=fileHelpText)

    # Parser for the CNN
    parser_cnn = subparsers.add_parser('cnn', help='Commands related to training, testing and using the CNN.',
                                       formatter_class=argparse.RawTextHelpFormatter)
    parser_cnn.add_argument('--file', '-f', action='store', dest='file', nargs='?', help=fileHelpText)
    parser_cnn.add_argument('cnn_command', choices=CNN_FUNCTIONS.keys(), help=cnnHelpText)
    parser_cnn.add_argument('--count','-c', action='store', type=int, help="Number of files to be evaluated")

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
            print("Plotting for file {}...".format(args.file))
            PlotEnvelopesAndF2FromFile(args.file)

    elif 'cnn_command' in args:
        if args.cnn_command == 'eval':
            import keras
            if args.file is not None:
                CNN_FUNCTIONS[args.cnn_command](args.file, keras)
            else:
                print("Please use --file or -f command to give an input file")
        elif args.cnn_command == 'train':
            inputFile = args.file or join('trainingData', 'input_data.npy')
            CNN_FUNCTIONS[args.cnn_command](inputFile)
        else:
            CNN_FUNCTIONS[args.cnn_command]()


if __name__ == '__main__':
    main()
