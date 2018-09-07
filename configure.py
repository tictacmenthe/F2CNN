from configparser import ConfigParser


def configure():
    formant = input('What will you be working on ? Enter k for Fk the formant that is used(default 2):') or '2'
    framerate = input('Enter the working framerate(default 16000):') or '16000'
    nchannels = input('Enter the number of filterbank channels(default 128):') or '128'
    lowcutoff = input('Enter the low cutoff frequency(default 100):') or '100'
    sampPeriod = input('Enter the label database sampling period(default 10000):') or '10000'
    centered = input('Are the labeling frames centered on a timeframe or not? y/n (default y)') or 'y'
    centered='True' if centered.lower()[0] == 'y' else 'False'
    print('Will use successive values of formant for slope computation.')
    if centered == 'True':
        inputRadius = input('Enter the CNN input radius\n(frames used will be between i+-radius*samplingperiod of .FB files)\n(default 5 values):') or '5'
    else:
        inputRadius = 1
    batchsize = input('Enter the CNN batch size(default 32):') or '32'
    epochs = input('Enter the CNN max epochs(default 20):') or '20'
    risk = input('Enter the CNN labeling slope risk(default 5%):') or '0.05'

    parser = ConfigParser()

    parser.add_section('FILTERBANK')
    parser['FILTERBANK']['FRAMERATE'] = framerate
    parser['FILTERBANK']['NCHANNELS'] = nchannels
    parser['FILTERBANK']['LOW_FREQ'] = lowcutoff

    parser.add_section('CNN')
    parser['CNN']['FORMANT'] = formant
    parser['CNN']['CENTERED'] = centered
    parser['CNN']['RADIUS'] = inputRadius
    parser['CNN']['BATCH_SIZE'] = batchsize
    parser['CNN']['EPOCHS'] = epochs
    parser['CNN']['RISK'] = risk
    parser['CNN']['SAMPLING_PERIOD'] = sampPeriod

    print("Saving configuration file as 'configF2CNN.conf")
    with open('configF2CNN.conf', 'w') as fp:
        parser.write(fp)
if __name__ == '__main__':
    configure()