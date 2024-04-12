import librosa
import glob
import random
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import pandas as pd

LABEL_DICTIONARY = ['blues', 'classical', 'country','disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
LABEL_DICTIONARY = {i: idx for idx, i in enumerate(LABEL_DICTIONARY)}
SAMPLE_RATE = 22050

def path_to_spectogram(path):
    try:
        path = path.numpy().decode("utf-8")
        wave, sr = librosa.load(path, sr=SAMPLE_RATE)
        n_fft=2048
        hop_length=512
        img = librosa.feature.melspectrogram(y=wave,sr = SAMPLE_RATE, n_fft = n_fft, hop_length = hop_length)
        img = np.expand_dims(img, axis = -1)
        label = path.split("/")[-2]
        label = LABEL_DICTIONARY[label]
        return img, label
    except Exception as e:
        print(e)
        return np.zeros((2,2)), -1

def path_to_audio(path):
    try:
        path = path.numpy().decode("utf-8")
        wave = tfio.audio.AudioIOTensor(path).to_tensor()
        wave = tf.cast(wave, tf.float32) / 32768.0
        wave = tf.squeeze(wave, axis = -1) 
        label = path.split(".")[0]
        label = LABEL_DICTIONARY[label]
        return wave, label
    except Exception as e:
        print(e)
        return path
    
def parse_function(path):
    [out, label] = tf.py_function(path_to_spectogram, [path], [tf.float32, tf.int32])
    return out, label
    
def parse_function_v0(path):
    [out, label] = tf.py_function(path_to_audio, [path], [tf.float32, tf.int32])
    return out, label

def encode_audio(path, n_segments):
  num_mfcc=20
  sample_rate=22050
  n_fft=2048
  hop_length=512
  num_segment=n_segments
  samples_per_segment = int(sample_rate*30/num_segment)

  segments_data = []
  y, sr = librosa.load(path, sr=sample_rate)

  for n in range(num_segment):
        data={}

        y_seg = y[samples_per_segment*n: samples_per_segment*(n+1)]
        data['length'] = int(y_seg.shape[0])

        chroma_hop_length = 512
        chromagram = librosa.feature.chroma_stft(y=y_seg, sr=sample_rate, hop_length=chroma_hop_length)
        data["chroma_stft_mean"] = (chromagram.mean())
        data["chroma_stft_var"] = (chromagram.var())

        #Root Mean Square Energy
        RMSEn= librosa.feature.rms(y=y_seg)
        data["rms_mean"] = (RMSEn.mean())
        data["rms_var"] = (RMSEn.var())

        #Spectral Centroid
        spec_cent=librosa.feature.spectral_centroid(y=y_seg)
        data["spectral_centroid_mean"] = (spec_cent.mean())
        data["spectral_centroid_var"] = (spec_cent.var())

        #Spectral Bandwith
        spec_band=librosa.feature.spectral_bandwidth(y=y_seg,sr=sample_rate)
        data["spectral_bandwidth_mean"] = (spec_band.mean())
        data["spectral_bandwidth_var"] = (spec_band.var())

        #Rolloff
        spec_roll=librosa.feature.spectral_rolloff(y=y_seg,sr=sample_rate)
        data["rolloff_mean"] = (spec_roll.mean())
        data["rolloff_var"] = (spec_roll.var())

        #Zero Crossing Rate
        zero_crossing=librosa.feature.zero_crossing_rate(y=y_seg)
        data["zero_crossing_rate_mean"] = (zero_crossing.mean())
        data["zero_crossing_rate_var"] = (zero_crossing.var())

        #Harmonics and Perceptrual
        harmony, perceptr = librosa.effects.hpss(y=y_seg)
        data["harmony_mean"] = (harmony.mean())
        data["harmony_var"] = (harmony.var())
        data["perceptr_mean"] = (perceptr.mean())
        data["perceptr_var"] = (perceptr.var())

        #Tempo
        tempo, _ = librosa.beat.beat_track(y=y_seg, sr=sample_rate)
        data["tempo"] = (tempo)

        #MFCC
        mfcc=librosa.feature.mfcc(y=y_seg,sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc=mfcc.T
        for x in range(20):
            feat1 = "mfcc" + str(x+1) + "_mean"
            feat2 = "mfcc" + str(x+1) + "_var"
            data[feat1] = (mfcc[:,x].mean())
            data[feat2] = (mfcc[:,x].var())

        segments_data.append(data)
  return segments_data