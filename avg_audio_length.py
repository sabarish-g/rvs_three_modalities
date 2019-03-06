import wave
import os
import pdb
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

def get_duration_wav(wav_filename):
    f = wave.open(wav_filename, 'r')
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    f.close()
    return duration



data_path = '/shared/kgcoe-research/mil/SpeechCOCO_2014/train2014/wav'
files_list = os.listdir(data_path)

durations = []

for i in range(0,len(files_list)):
    durations.append(get_duration_wav(os.path.join(data_path,files_list[i])))

data = np.asarray(durations)
print('the min is %f' %np.min(data))
print('the max is %f' %np.max(data))
print('the mean is %f' %np.mean(data))
print('the median is %f' %np.median(data))