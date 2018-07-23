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
        OrganiseAllFiles()
    FilterAllOrganisedFiles(testMode)
    ExtractAllEnvelopes(testMode)
    GenerateLabelData(testMode)
    GenerateInputData(testMode)


def main():
    FUNCTIONS = {
        'all': All,
        'organise': OrganiseAllFiles,
        'filter': FilterAllOrganisedFiles,
        'envelope': ExtractAllEnvelopes,
        'label': GenerateLabelData,
        'input': GenerateInputData
    }
    parser = argparse.ArgumentParser(description="F2CNN Project's entry script")
    subparsers = parser.add_subparsers()

    parser_prepare = subparsers.add_parser('prepare', help='runs the command given in argument')
    parser_prepare.add_argument('command', choices=FUNCTIONS.keys())
    parser_prepare.add_argument('--test','-t', action='store_true')

    args=parser.parse_args()

    # Starts the command requested
    FUNCTIONS[args.command](args.test)


if __name__ == '__main__':
    main()
