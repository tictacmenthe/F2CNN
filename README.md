# F2CNN
Speech feature extraction using Gammatone FilterBank and CNN.\
The actual training uses TIMIT database and VTR Formants database.


## Usage
Use the configure.py script to initialize a few parameters for the program.\
Run it every time one of the project parameters need to change (Sampling rates, cutoff frequencies, numbers of channels...).\
Use the F2CNN.py Python script as en entrance or use the functions in the scripts/ directories in your own.

*examples*:

``` python3 F2CNN.py prepare all ``` \
-> prepares all the data for CNN usage (may take a few minutes and requires a lot of disk space (50GB if full Timit/vtr database).\
```python3 F2CNN.py -h/--help``` \
-> gives some help and tips after any parameter. \
```python3 F2CNN.py plot gtg -f *PathToAFile*```\
-> Plots a spectrogram like representation of GammaTone Filterbankoutput.

#### Required structure
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
## Dependencies

This project requires the installation of:
- numpy
- matplotlib
- keras
- scipy
Also, some packages are needed for the training:
There are two cases:
#### 1. You will use your CPU
* tensorflow
#### 2. You will use a GPU (preferably from NVIDIA)
* tensorflow-cuda (or tensorflow-gpu in PIP)
* cudnn
* cuda

If you have any issues or suggestions, you can send an email at tictacmenthedouce@gmail.com .