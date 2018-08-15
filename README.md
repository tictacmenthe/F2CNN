# F2CNN
Speech feature extraction using Gammatone FilterBank and CNN.\
The actual training uses TIMIT database and VTR Formants database.


## Usage
Use the configure.py script with python3 to initialize a few parameters for the program.\
Run it every time one of the project parameters need to change (Sampling rates, cutoff frequencies, numbers of channels, number of classes...).\
Use the f2cnn.py **Python3** script as en entrance or use the functions in the scripts/ directories in your own.

###Examples:
```python3 f2cnn.py -h/--help``` \
-> gives some help and tips after any argument.

##### Data processing scripts
``` python3 f2cnn.py prepare all ``` \
-> prepares all the data for CNN usage (may take a few minutes and requires a lot of disk space (50GB if full Timit/vtr database).\
``` python3 f2cnn.py prepare organize``` \
-> prepares Project Structure with Timit and VTR databases organized as mentionne din the "REQUIRED STRUCTURE" section down below.\
``` python3 f2cnn.py prepare filter ``` \
-> prepares Filtered outputs from the gammatone filterbank\
``` python3 f2cnn.py prepare envelope --cutoff 50 ``` \
-> prepares extracted envelope numpy array files using a low pass filter at 50Hz, saves them in a FILENAME.ENVi.npy format, i being the method used.\
``` python3 f2cnn.py prepare label ``` \
-> prepares CNN output labels from the previous files, using VTR .FB files, .PHN files and filenames, as a trainingData/label_data.csv file.\
``` python3 f2cnn.py prepare input --cutoff 50 ``` \
-> prepares CNN input data matrices from latest extracted envelopes, and saves the whole as a Nx11x128 ndarray trainingData/input_data.npy.\
If CUTOFF is used, will save the file as trainingData/input_data_LPFX.npy with X the frequency.\
Also makes a backup as trainingData/last_input_data.npy, just in case.
##### Data plotting scripts
```python3 f2cnn.py plot gtg --file/-f *PathToAWAVFileFile*```\
-> Plots a spectrogram like representation of GammaTone Filterbankoutput.
##### CNN related scripts
```python3 f2cnn.py cnn train --file *PathToInputDataFile*``` \
-> Trains a CNN using the given input data file, or by default trainingData/input_data.npy, also uses the default labe_data.csv file. \
```python3 f2cnn.py eval --file *PathToAWAVFile*``` \
-> Uses the last_trained_model keras model to predict Rising or Falling for F2 on all frames of the given .WAV file, plotting results in graphs/FallingOrRising directory. \
```python3 f2cnn.py evalrand --count 5``` \
-> Same as the above, but evaluates randomly all the VTR related TIMIT .WAV files. If COUNT is used, only evaluates 5 random files.


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

This project requires the installation of:
- numpy
- matplotlib
- keras
- scipy
Also, some packages are needed for the training:
There are two cases:
##### 1. You will use your CPU
* tensorflow
##### 2. You will use a GPU (preferably from NVIDIA)
* tensorflow-cuda (or tensorflow-gpu in PIP)
* cudnn
* cuda

If you have any issues or suggestions, you can send an email at *tictacmenthedouce@gmail.com* .