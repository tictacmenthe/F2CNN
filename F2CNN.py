import os
import wave
import numpy
import struct
import matplotlib.pyplot as plt
from scripts.GammatoneFiltering import getFilteredOutput
file="testFiles/DR4.MTQC0.SI1441.WAV"
f0=180
dt=0.0000625


# .WAV file to list
wavFile = wave.open(file, 'r')
wavList = numpy.zeros(wavFile.getnframes())
for i in range(wavFile.getnframes()):
    a = wavFile.readframes(1)
    a = struct.unpack("<h", a)[0]
    wavList[i] = a


def Gammatone(signal):
    z=numpy.zeros(len(signal),dtype=complex)
    w=numpy.zeros(len(signal),dtype=complex)
    for k in range(len(z)):
        z[k]=signal[k]*numpy.exp(-2j*numpy.pi*k*dt)
        w[k]=(1-numpy.exp(-2*numpy.pi*f0*dt))*z[k]
    for k in range(len(z)):
        w[k]=w[k]+(1-numpy.exp(-2*numpy.pi*f0*dt))*(z[k]-w[k])
    for k in range(len(z)):
        z[k]=(w[k]*numpy.exp(2j*numpy.pi*k*dt))
    return z


selfFiltered=Gammatone(wavList)
plt.plot(wavList)
plt.plot(selfFiltered)
plt.grid()
plt.title("WAV FILE "+file)
plt.show()