import argparse

from scripts.OrganiseFiles import OrganiseAllFiles
from scripts.GammatoneFiltering import FilterAllOrganisedFiles
from scripts.EnvelopeExtraction import ExtractAllEnvelopes
from scripts.LabelDataGenerator import GenerateLabelData
from scripts.InputGenerator import GenerateInputData
from scripts.Plotting import PlotEnvelopesAndF2FromFile
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
        'eval': EvaluateOneFile,
        'evalrand': EvaluateRandom
    }
    helpText = """Data processing commands:\n\t\
organize:\tOrganizes the files as needed for the rest (Check OrganiseFiles.py documentation)\n\t\
filter:\t\tApplies the GammaTone FilterBank to the organized files and saves its outputs in .GFB.npy format\n\t\
envelope:\tExtracts the filtered files' envelopes and saves them in .ENV1.npy format\n\t\
label:\t\tGenerates Labeling data for the CNN\n\t\
input\t\tGenerates Input data for the CNN, requires label first\n\t\
all:\t\tDoes all of the above, can take some time.
    """
    parser = argparse.ArgumentParser(description="F2CNN Project's entry script.",
                                     epilog="For additional information, add -h after any positional argument")
    parser.add_argument('--silent', '-s', action='store_false', help="Remove logs, for mostly silent processing.")
    subparsers = parser.add_subparsers()

    parser_prepare = subparsers.add_parser('prepare', help='Runs the command given in argument.',
                                           formatter_class=argparse.RawTextHelpFormatter)
    parser_prepare.add_argument('prepare_command', choices=PREPARE_FUNCTIONS.keys(), help=helpText)
    parser_prepare.add_argument('--test', '-t', action='store_true', help="Test mode, uses the files located in the "
                                                                          "testFiles directory.")
    parser_plot = subparsers.add_parser('plot', help='For plotting spectrogram-like figure from .WAV file.')
    parser_plot.add_argument('--gammatonegram', '-g', dest='gfilename', nargs=None, type=str,
                             help="Plots a spectrogram like figure from the output of a GammaTone FilterBank applied\
                                  to the given file, and if a .FB file exists in the dir, also plots the Formants.")

    parser_cnn = subparsers.add_parser('cnn', help='Commands related to training, testing and using the CNN.',
                                       formatter_class=argparse.RawTextHelpFormatter)
    parser_cnn.add_argument('cnn_command', choices=CNN_FUNCTIONS.keys())

    args = parser.parse_args()

    # Starts the command requested
    # TODO: silent
    if 'prepare_command' in args:
        PREPARE_FUNCTIONS[args.prepare_command](args.test)
    if 'gfilename' in args:
        print("Plotting for file {}...".format(args.gfilename))
        PlotEnvelopesAndF2FromFile(args.gfilename)
    if 'cnn_command' in args:
        CNN_FUNCTIONS[args.cnn_command]()


if __name__ == '__main__':
    main()
