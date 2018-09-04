import os
import argparse

from scripts.processing.OrganiseFiles import OrganiseAllFiles
from scripts.processing.GammatoneFiltering import FilterAllOrganisedFiles
from scripts.processing.EnvelopeExtraction import ExtractAllEnvelopes
from scripts.processing.LabelDataGenerator import GenerateLabelData
from scripts.processing.InputGenerator import GenerateInputData
from scripts.plotting.PlottingProcessing import PlotEnvelopesAndF2FromFile
from scripts.CNN.Evaluating import EvaluateOneWavFile, EvaluateRandom, EvaluateWithNoise
from scripts.CNN.Training import TrainAndPlotLoss


def All(LPF=False, CUTOFF=100):
    """
    Does all the treatments required for the training
    """
    OrganiseAllFiles()
    FilterAllOrganisedFiles()
    ExtractAllEnvelopes(LPF, CUTOFF)
    GenerateLabelData()
    GenerateInputData(LPF, CUTOFF)


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
        'eval': EvaluateOneWavFile,  # Applies the CNN to one specified file
        'evalnoise': EvaluateWithNoise,  # Applies the CNN to one specified file
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
    inputHelpText = "Used to give a path to an input numpy file as an argument to some scripts."
    labelHelpText = "Used to give a path to a label csv file as an argument to some scripts."
    modelHelpText = "Used to give the path to a keras model as an argument to some scripts."

    # #####  PARSING
    parser = argparse.ArgumentParser(description="F2CNN Project's entry script.",
                                     epilog="For additional information, add -h after any positional argument")
    subparsers = parser.add_subparsers()

    # Parser for data processing purposes
    parser_prepare = subparsers.add_parser('prepare', help='Runs the command given in argument.',
                                           formatter_class=argparse.RawTextHelpFormatter)
    parser_prepare.add_argument('--cutoff', '-c', action='store', dest='CUTOFF', type=int,
                                help="If used, low pass filter of given argument as cutoff frequency will be used")
    parser_prepare.add_argument('prepare_command', choices=PREPARE_FUNCTIONS.keys(), help=preparationHelpText)
    parser_prepare.add_argument('--file', '-f', action='store', dest='file', nargs='?', help=fileHelpText)
    parser_prepare.add_argument('--input', '-i', action='store', dest='inputFile', nargs='?', help=inputHelpText)
    parser_prepare.add_argument('--label', '-l', action='store', dest='labelFile', nargs='?', help=labelHelpText)

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
    parser_cnn.add_argument('--input', '-i', action='store', dest='inputFile', nargs='?', help=inputHelpText)
    parser_cnn.add_argument('--label', '-l', action='store', dest='labelFile', nargs='?', help=labelHelpText)
    parser_cnn.add_argument('--model', '-m', action='store', dest='model', nargs='?', help=modelHelpText)
    parser_cnn.add_argument('cnn_command', choices=CNN_FUNCTIONS.keys(), help=cnnHelpText)
    parser_cnn.add_argument('--count', '-c', action='store', type=int, help="Number of files to be evaluated")
    parser_cnn.add_argument('--lpf', action='store', type=int, dest='CUTOFF',
                            help="Use Low Pass Filtering on Input Data")
    parser_cnn.add_argument('--noise', '-n', action='store', type=float, dest='SNRdB',
                            help="To use with evalnoise to give a SNR in dB.")
    # Processes the input arguments
    args = parser.parse_args()
    print(args)
    # Calls to functions according to arguments
    if 'prepare_command' in args:
        prepare_args={}
        if args.prepare_command in ['envelope', 'input', 'all']:  # In case we need to use a low pass filter
            prepare_args['LPF']=False if args.CUTOFF is None else True
            prepare_args['CUTOFF']=args.CUTOFF
        if args.prepare_command == 'input':
            if args.labelFile is not None:
                prepare_args['labelFile']=args.labelFile
            if args.inputFile is not None:
                prepare_args['inputFile']=args.inputFile
        PREPARE_FUNCTIONS[args.prepare_command](**prepare_args)
    elif 'plot_type' in args:
        if args.file is None:
            print("Please use --file or -f to give input file")
        else:
            print("Plotting for file {}...".format(args.file))
            PlotEnvelopesAndF2FromFile(args.file)
    elif 'cnn_command' in args:
        if args.cnn_command == 'train':
            inputFile = args.file or os.path.join('trainingData', 'last_input_data.npy')
            labelFile = args.labelFile or os.path.join('trainingData', 'label_data.csv')
            if not os.path.isfile(inputFile):
                print(
                    "Please first generate the input data file with 'prepare input',\n\
                    or give a path to an input data file with --input")
                print(
                    "Reminder: input data files generated with 'prepare input' are stored in \n\
                    trainingData/ as 'input_data_LPFX.npy or 'input_data_NOLPF.npy', depending on Low Pass Filtering used.")
            if not os.path.isfile(labelFile):
                print(
                    "Please first generate a label data file with 'prepare label',\n\
                    or give a path to a label data file with --label")
                print(
                    "Reminder: label data files generated with 'prepare label' are stored in \n\
                    trainingData/ as 'label_data.csv'.")
            CNN_FUNCTIONS[args.cnn_command](labelFile=labelFile, inputFile=inputFile)
            return
        elif 'file' in args and args.file is not None:
            evalArgs = {'file': args.file}
            if 'CUTOFF' in args:
                evalArgs['LPF'] = True
                evalArgs['CUTOFF'] = args.CUTOFF
            if 'model' in args and args.model is not None:
                evalArgs['model'] = args.model
            if args.cnn_command == 'evalnoise' and 'SNRdB' in args and args.SNRdB is not None:
                evalArgs['SNRdB'] = args.SNRdB
            if args.cnn_command == 'evalrand' and 'count' in args and args.count is not None:
                evalArgs['COUNT'] = args.count
            CNN_FUNCTIONS[args.cnn_command](**evalArgs)
    else:
        print("For help, use python3 f2cnn.py --help or -h, or check the documentation on github.")
        print("No valid command given.")


if __name__ == '__main__':
    main()
