import argparse

from scripts.OrganiseFiles import OrganiseAllFiles
from scripts.GammatoneFiltering import FilterAllOrganisedFiles
from scripts.EnvelopeExtraction import ExtractAllEnvelopes
from scripts.LabelDataGenerator import GenerateLabelData
from scripts.InputGenerator import GenerateInputData


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
    FUNCTIONS = {
        'all': All,
        'organize': OrganiseAllFiles,
        'filter': FilterAllOrganisedFiles,
        'envelope': ExtractAllEnvelopes,
        'label': GenerateLabelData,
        'input': GenerateInputData
    }
    helpText = """Data processing commands:\n\t\
organize:\tOrganizes the files as needed for the rest (Check OrganiseFiles.py documentation)\n\t\
filter:\t\tApplies the GammaTone FilterBank to the organized files and saves its outputs in .GFB.npy format\n\t\
envelope:\tExtracts the filtered files' envelopes and saves them in .ENV1.npy format\n\t\
label:\t\tGenerates Labeling data for the CNN\n\t\
input\t\tGenerates Input data for the CNN, requires label first\n\t\
all:\t\tDoes all of the above
    """
    parser = argparse.ArgumentParser(description="F2CNN Project's entry script")
    subparsers = parser.add_subparsers()
    parser_prepare = subparsers.add_parser('prepare', help='Runs the command given in argument',
                                           formatter_class=argparse.RawTextHelpFormatter)
    parser_prepare.add_argument('command', choices=FUNCTIONS.keys(), help=helpText)
    parser_prepare.add_argument('--test', '-t', action='store_true', help="Test mode, uses the files located in the\
                                                                    testFiles directory")
    parser.add_argument('--silent', '-s', action='store_false', help="Remove logs, for mostly silent processing")

    args = parser.parse_args()

    # Starts the command requested
    # TODO: silent
    FUNCTIONS[args.command](args.test)


if __name__ == '__main__':
    main()
