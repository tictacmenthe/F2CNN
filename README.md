# F2CNN
Speech feature extraction using Gammatone FilterBank and CNN.\
The project uses TIMIT database and VTR Formants database as resources for the CNN.


## Usage
**Use the ```configure.py``` script with python3, or run ```python3 f2cnn.py --configure/-c``` to initialize a few parameters for the program.**\
Run it every time one of the project parameters need to change (Sampling rates, cutoff frequencies, numbers of channels, number of classes...).\
Use the f2cnn.py **Python3** script as en entrance like in the examples below, or use the functions in the scripts/ directories in your own.

###Available commands:
```python3 f2cnn.py -h/--help``` \
-> gives some help and tips after any argument.

##### Data processing scripts
``` python3 f2cnn.py prepare all ``` \
-> prepares all the data for CNN usage (may take a few minutes and requires a lot of disk space (~50GB if full Timit/vtr database).\

``` python3 f2cnn.py prepare organize``` \
-> prepares Project Structure with Timit and VTR databases organized as mentionne din the "REQUIRED STRUCTURE" section down below.\

``` python3 f2cnn.py prepare filter ``` \
-> prepares Filtered outputs from the gammatone filterbank\
Saves all the outputs as '.GFB.npy' files.
``` python3 f2cnn.py prepare envelope```\
__Optional command:__ ```--cutoff FREQ ``` for a low pass filtering on the envelopes with a cutoff of FREQ Hz  \
-> prepares extracted envelope numpy array files using a low pass filter at 50Hz\
Saves all outputs as '.ENV1.npy' files. The 1 means that the method used is the first one, should there be more in the future.

``` python3 f2cnn.py prepare label ``` \
-> prepares CNN output labels from the previous files, using VTR .FB files, .PHN files and filenames.\
Saves it as a trainingData/label_data.csv file.\

``` python3 f2cnn.py prepare input```\
__Optional command:__ ```--cutoff FREQ ``` specifies the cutoff frequency for the output file(should be the same as the envelopes)\
-> prepares CNN input data matrices from latest extracted envelopes, and saves the whole as a NxDOTS_PER_INPUTx_NB_CHANNELS ndarray trainingData/input_data.npy.\
If CUTOFF is used, will save the file as trainingData/input_data_LPFX.npy with X the frequency.\
Also makes a backup as trainingData/last_input_data.npy, just in case.

##### Data plotting scripts
```python3 f2cnn.py plot gtg --file/-f *PathToAWAVFileFile*```\
-> Plots a spectrogram like representation of GammaTone FilterBank output.
##### CNN related scripts
```python3 f2cnn.py cnn train```\
 __Optional commands:__
```--input *PathToInputDataFile*``` allows the use of a specific input data file \
```--label *PathToLabelCSVFile*``` allows the use of a specific label data file\
-> Trains a CNN using the given input data file, or by default trainingData/input_data.npy, also uses the default labe_data.csv file. \
```python3 f2cnn.py cnn eval --file *PathToAWAVFile*``` \
-> Uses the last_trained_model keras model to predict Rising or Falling for F2 on all frames of the given .WAV file, plotting results in graphs/FallingOrRising directory. \
```python3 f2cnn.py cnn evalrand``` \
__Optional command:__ ```--count N``` only uses N randomly selected files
-> Same as the above, but evaluates randomly all the VTR related TIMIT .WAV files.\
```python3 f2cnn.py cnn evalnoise```\
__Optional command:__ ```--noise SNRdB``` specifies a Signal to Noise Ratio in dB for the new WAV file, that is saved inside 'OutputWavFiles/addedNoise'.\

__Optional commands for the last 3 functions:__ \
```--model *PathToAKerasModel*``` allows the use of a specific keras model(default:'last_trained_model')\
```--cutoff FREQ``` allows the use of a FREQ Hz cutoff Low Pass Filter on envelope extraction

### Required structure
There is a certain way the project directories should be organized before running ```prepare organize``` or ```prepare all```:\
project_root/\
-scripts/   -> This file's directory\
**-resources/**\
--f2cnn/    -> The output directory (should be created automatically)\
---TRAIN/\
---TEST/\
**--TIMIT/**    -> TIMIT database as is, **including TEST and TRAIN directories**\
**--VTR/**      -> VTR FORMANTS files as is, **including TEST and TRAIN subdirectories**\
So make sure to put the TIMIT(the one including TEST and TRAIN directories) and VTR directories in the resources directory.
### Dependencies
Compatibility with Windows is not garanteed.\
This project requires the installation of:
- numpy - 1.14.5
- matplotlib - 2.2.2
- keras - 2.2.0
- scipy - 1.1.0
- sphfile - 1.0.0

Also, some packages are needed for the training:
There are two cases:
##### 1. You will use your CPU
* tensorflow with pip
##### 2. You will use a GPU (preferably from NVIDIA)
* tensorflow-cuda (or tensorflow-gpu in PIP) - 1.9.0
* cuda  -   9.2.148-1  | installed with the nvidia cuda toolkit 9.2

If you have any issues or suggestions, you can send an email at __tictacmenthedouce@gmail.com__ .