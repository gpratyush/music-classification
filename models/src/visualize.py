import librosa
import matplotlib.pyplot as plt
import numpy as np

def harmonics_and_perceptrual(audio, sr, save_location, name):
    y_harm, y_perc = librosa.effects.hpss(audio)
    plt.figure(figsize = (6, 3))
    plt.plot(y_harm, color = '#A300F9');
    plt.plot(y_perc, color = '#FFB100');
    plt.title(f"Harmonics and Perceptruals in {name}", fontsize = 10);
    plt.savefig(save_location+"/hpss.png")
    return save_location+"/hpss.png"
    
    
def mel_spectogram(audio, sr, save_location, name):
    hop_length = 512
    n_fft = 2048
    S = librosa.feature.melspectrogram(y = audio, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize = (6, 3))
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis = 'time', y_axis = 'log',
                            cmap = 'cool');
    plt.colorbar();
    plt.title(f"Mel Spectrogram for {name}", fontsize = 10);
    plt.savefig(save_location+"/mel_spectogram.png")
    return save_location+"/mel_spectogram.png"
    
def chroma_stft(audio, sr, save_location, name):
    hop_length = 5000

    chromagram = librosa.feature.chroma_stft(y = audio, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(6, 3))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm');
    plt.title(f"Chromagram for {name}", fontsize = 10);
    plt.savefig(save_location+"/chromagram.png")
    return save_location+"/chromagram.png"