import numpy as np
import os.path
import sys
sys.path.append("..")
from pathlib import Path
#from keras.models import load_model
import pypianoroll
import data_processing.converter as converter
from pypianoroll import Multitrack,  Track
# test
file = '../music_data/samp.mid' #24 bar
num_timestep = 768
num_pitch = 60
resol = 24

# Test data sliding window input
def SlidingWindow(file, resol, num_timestep, num_pitch, num_consecutive_bar=8, down_sample=1):
    multitrack = pypianoroll.parse(file)
    multitrack.plot()
    #print(multitrack)
    multitrack = converter.first_note_code(multitrack)  
    #print(multitrack)
    downbeat = multitrack.downbeat #[ True False False ... False False False]
    num_bar = len(downbeat) // resol #96
    hop_iter = 0
    song_ok_segments = []
    track = multitrack.tracks[0]
    #print(track)
    #print(track.pianoroll.shape) #(2304, 128) 24x4x24
    #print(track[0:768:down_sample].pianoroll.shape)#(768, 128)
    for bidx in range(num_bar - num_consecutive_bar // 2): # 1~80
        if hop_iter > 0:
            hop_iter -= 1
            continue # once in 16
        st = bidx * resol #0 384 768 ... bidx 0 16 32 ...
        ed = st + num_consecutive_bar * resol #768 1132 ...
        tmp_pianoroll = track[st:ed:down_sample]
        song_ok_segments.append(tmp_pianoroll.pianoroll[np.newaxis, :, :])
        hop_iter = num_consecutive_bar / 2 - 1
    pianoroll_compiled = np.concatenate(song_ok_segments, axis=0)[:, :, 28:88]
    # x_pre = np.reshape(pianoroll_compiled, (-1, num_timestep, num_pitch, 1))
    x_pre = np.reshape(pianoroll_compiled, [-1, num_timestep, num_pitch, 1])
    #print(x_pre[2,10:35,30:60,:])
    x_pre[(x_pre > 0) & (x_pre < 128)] = 1 #velocity 80 -> binary
    x_pre[(x_pre > 128)] = 1
    x_pre = x_pre.astype('float32')
    return x_pre
# preprocessing
x_pre = SlidingWindow(file, resol, num_timestep, num_pitch, num_consecutive_bar=4 * 8)

print(x_pre.shape) #(5, 768, 60, 1)
#print(x_pre[0,:,:,:])


multitrack = Multitrack(str(file), beat_resolution=24)


downbeat = multitrack.downbeat

num_bar = len(downbeat) // resol 
print('num_bar: ', num_bar)
