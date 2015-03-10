from scipy.io import wavfile
import numpy as np
import os
import sys
import tables
import copy
from wrap1 import *
from librosa_ports import invmelspec

def soundsc(X, copy=True):
    """
    Approximate implementation of soundsc from MATLAB without the audio playing.
    Parameters
    ----------
    X : ndarray
        Signal to be rescaled
    copy : bool, optional (default=True)
        Whether to make a copy of input signal or operate in place.
    Returns
    -------
    X_sc : ndarray
        (-1, 1) scaled version of X as float, suitable for writing
        with scipy.io.wavfile
    """
    X = np.array(X, copy=copy)
    X = (X - X.min()) / (X.max() - X.min())
    X = 2 * X - 1
    X *= 2 ** 15
    return X.astype('int16')


"""
base_dir = '/u/kruegerd/TTS_current/speechgeneration/genwav/'
filenames = []
files_are_wav = True
my_files = []

for i in range(4):
    filename = base_dir + 'timit100_preds'+str(i)+'.wav'
    filenames.append(filename)
    if files_are_wav:
        X = wav.read(filename)[1].reshape((-1,229))
    else:
        X = np.load(filename).reshape((-1,229))

    X = np.vstack((X,X,X,X,X))

    save_name = filename

    #my_file2 = np.load('/u/kruegerd/TTS_current/speechgeneration/timit_preds9_ground_truth_residuals.npy')
    #X = my_file.reshape((-1,229))
"""

def vocoder_synth(X, save_name=None): 
    """X.shape = nframes, 229"""

    # WORLD does some strange padding in the analysis, resulting in a longer signal than expected
    len_x = (7 + X.shape[0]) * 5 * 16

    period = 5.0
    fs = 16000
    n_log_mel_components = 2 * 64
    n_residual_components = 100

    # Undo normalization
    m = np.load('/data/lisatmp/dinhlaur/kastner_invited/min_max_mean_std.npy')
    min_stats = m[0]
    max_stats = m[1]
    mean_stats = m[2]
    std_stats = m[3]
    f0_max = max_stats[0]
    spec_mean = mean_stats[1:n_log_mel_components + 1]
    f0 = X[:, 0] * max_stats[0]
    log_mel_spectrogram = X[:, 1:n_log_mel_components + 1] * std_stats[1:129] + mean_stats[1:129]
    residual = X[:, -n_residual_components:] * std_stats[129:] + mean_stats[129:]
    mel_spectrogram = np.exp(log_mel_spectrogram)
    spectrogram = np.ascontiguousarray(invmelspec(mel_spectrogram, fs, 1024)) + 1E-12

    # Undo PCA
    residual_matrix = np.load('/data/lisatmp/dinhlaur/kastner_invited/timit/test_residual_matrix.npy')
    residual_subset_mean = np.load('/data/lisatmp/dinhlaur/kastner_invited/timit/test_residual_mean.npy')
    r_means = residual_subset_mean[:]
    residual = np.dot(residual, residual_matrix) + r_means

    s = np.ascontiguousarray(spectrogram.astype('float64'))
    r = np.ascontiguousarray(residual.astype('float64'))
    f0 = np.ascontiguousarray(f0.astype('float64'))
    period = np.cast['float64'](period)
    fs = np.cast['int32'](fs)
    len_x = np.cast['int32'](len_x)
    s = copy.deepcopy(s)
    r = copy.deepcopy(r)
    f0 = copy.deepcopy(f0)

    # because of the padding, we must adjust the length.  560 = frame overlap.
    len_x -= 560
    print len_x

    print "attempting synthesis"
    y = synthesis(fs, period, f0, s, r, len_x)
    if save_name is not None:
        wavfile.write(save_name + ".wav", fs, soundsc(y))
    return y

