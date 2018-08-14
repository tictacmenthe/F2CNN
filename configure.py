from configparser import ConfigParser

framerate = input('Enter the working framerate(default 16000):') or '16000'
nchannels = input('Enter the number of filterbank channels(default 128):') or '128'
lowcutoff = input('Enter the low cutoff frequency(default 100):') or '100'
sampPeriod = input('Enter the label database sampling period(default 10000):') or '10000'
classes = input(
    'How many input labels should there be? \n\t\t2:rising and falling\n\t\t3:falling, rising, none)\nChoice:(default 2)') or '2'
inputRadius = input(
    'Enter the CNN input radius(frames used will be between i+-radius*samplitperiod of .FB files) in ms(default 5):') or '5'
batchsize = input('Enter the CNN batch size(default 32):') or '32'
epochs = input('Enter the CNN max epochs(default 20):') or '20'
risk = input('Enter the CNN labeling slope risk(default 5%):') or '0.05'

parser = ConfigParser()

parser.add_section('FILTERBANK')
parser['FILTERBANK']['FRAMERATE'] = framerate
parser['FILTERBANK']['NCHANNELS'] = nchannels
parser['FILTERBANK']['LOW'] = lowcutoff

parser.add_section('CNN')
parser['CNN']['CLASSES'] = classes
parser['CNN']['RADIUS'] = inputRadius
parser['CNN']['BATCHSIZE'] = batchsize
parser['CNN']['EPOCHS'] = epochs
parser['CNN']['RISK'] = risk
parser['CNN']['SAMPPERIOD'] = sampPeriod

with open('F2CNN.conf', 'w') as fp:
    parser.write(fp)
