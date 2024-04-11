import librosa

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