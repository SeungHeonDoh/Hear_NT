from typing import Tuple
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from hearbaseline.tf.util import frame_audio
from sklearn.preprocessing import normalize

import sys, os

#import leaf_audio.frontend as frontend

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from melfilterbanks import MelFilterbanks

# Default hop_size in milliseconds
TIMESTAMP_HOP_SIZE = 50
SCENE_HOP_SIZE = 250

# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 2048

class HEARNT(tf.Module):
    sample_rate = 22050
    embedding_size = 4096
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size

    # These attributes are specific to this baseline model
    n_fft = 1024
    window = 1024
    hop = 512
    n_mels = 128

    subdense = True

    def __init__(self, model_file_path):
        super().__init__()
        self.load_model(model_file_path)

    def load_model(self, model_file_path: str = ""):

        n_filters1 = 128
        window_len1 = 46  # // (22050 * 46 // 1000 + 1 = )
        window_stride1 = 23
        sample_rate1 = 22050

        mel_fb1 = MelFilterbanks(
                    n_filters=n_filters1,
                    window_len=window_len1,
                    window_stride=window_stride1,
                    sample_rate=sample_rate1,
                    n_fft= 1024,
                    min_freq=0.0,
                    max_freq=None,
                    name='MelFilterBanks1'
                )

        n_filters2 = 128
        window_len2 = 23  # // (22050 * 46 // 1000 + 1 = )
        window_stride2 = 11
        sample_rate2 = 22050

        mel_fb2 = MelFilterbanks(
                    n_filters=n_filters2,
                    window_len=window_len2,
                    window_stride=window_stride2,
                    sample_rate=sample_rate2,
                    n_fft=512,
                    min_freq=0.0,
                    max_freq=None,
                    name='MelFilterBanks2'
                )

        n_filters3 = 128
        window_len3 = 92  # // (22050 * 46 // 1000 + 1 = )
        window_stride3 = 46
        sample_rate3 = 22050

        mel_fb3 = MelFilterbanks(
                    n_filters=n_filters3,
                    window_len=window_len3,
                    window_stride=window_stride3,
                    sample_rate=sample_rate3,
                    n_fft=2048,
                    min_freq=0.0,
                    max_freq=None,
                    name='MelFilterBanks3'
                )
        
        test_model = tf.keras.models.load_model(model_file_path, compile=False, custom_objects={
                                                'MelFilterbanks': mel_fb1, 'MelFilterbanks': mel_fb2, 'MelFilterbanks': mel_fb3})
        
        if self.subdense == True:
            
            base_model = []
            base_model.append(Model(inputs=test_model.input,
                outputs=test_model.get_layer('dense_1').output))
            base_model.append(Model(inputs=test_model.input,
                outputs=test_model.get_layer('dense_7').output))
            base_model.append(Model(inputs=test_model.input,
                outputs=test_model.get_layer('dense_3').output))
            base_model.append(Model(inputs=test_model.input,
                outputs=test_model.get_layer('dense_4').output))
            base_model.append(Model(inputs=test_model.input,
                outputs=test_model.get_layer('dense_5').output))
            base_model.append(Model(inputs=test_model.input,
                outputs=test_model.get_layer('dense_6').output))
            self.model = base_model

        else:
            base_model = test_model.get_layer('model')
            self.model = base_model

    def load_mel(self, signal):
        S = librosa.core.stft(
            signal, n_fft=self.n_fft, hop_length=self.hop, win_length=self.window)
        X = np.abs(S)
        mel_basis = librosa.filters.mel(self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels)
        mel_S = np.dot(mel_basis, X)
        mel_S = np.log10(1+10*mel_S)
        mel_S = mel_S.astype(np.float32)
        mel_feature = np.transpose(mel_S)
        mel_feature -= 0.2
        mel_feature /= 0.25
        return mel_feature

    def __call__(self, x):
        batch_mel = []
        # for signal in x:
        #     mel_feature = self.load_mel(signal)
        #     mel_feature = mel_feature[:43,:]
        #     batch_mel.append(mel_feature)
        # batch_mel = np.stack(batch_mel)
        # mel_feature = np.expand_dims(batch_mel, -1)
        
        # Waveform input model
        for signal in x:
            mel_feature = signal[:22050]
            batch_mel.append(mel_feature)
        mel_feature = np.stack(batch_mel)

        if self.subdense == True:

            tmp0 = self.model[0].predict(mel_feature)
            tmp1 = self.model[1].predict(mel_feature)
            tmp2 = self.model[2].predict(mel_feature)
            tmp3 = self.model[3].predict(mel_feature)
            tmp4 = self.model[4].predict(mel_feature)
            tmp5 = self.model[5].predict(mel_feature)

            # Normalize each sub-feature
            tmp0 = normalize(tmp0, 'l2')
            tmp1 = normalize(tmp1, 'l2')
            tmp2 = normalize(tmp2, 'l2')
            tmp3 = normalize(tmp3, 'l2')
            tmp4 = normalize(tmp4, 'l2')
            tmp5 = normalize(tmp5, 'l2')

            embedding_feature = np.hstack((tmp0, tmp1, tmp2, tmp3, tmp4, tmp5))
        
        else:
            embedding_feature = self.model.predict(mel_feature)
            embedding_feature = normalize(embedding_feature, 'l2')

        #embedding_feature = normalize(embedding_feature, 'l2')
        return embedding_feature

def load_model(model_file_path: str = "") -> tf.Module:
    model = HEARNT(model_file_path)
    return model

def get_timestamp_embeddings(
    audio: tf.Tensor,
    model: tf.Module,
    hop_size: float = TIMESTAMP_HOP_SIZE,
) -> Tuple[tf.Tensor, tf.Tensor]:
    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (n_sounds, num_samples)"
        )
    if not isinstance(model, HEARNT):
        raise ValueError(
            f"Model must be an instance of HEARNT"
        )

    frames, timestamps = frame_audio(
        audio, frame_size=model.sample_rate, hop_size=hop_size, sample_rate=model.sample_rate
    )
    audio_batches, num_frames, frame_size = frames.shape
    frames = tf.reshape(frames, [audio_batches * num_frames, frame_size])

    embeddings_list = []
    for i in range(0, frames.shape[0], BATCH_SIZE):
        frame_batch = frames[i : i + BATCH_SIZE]
        frame_batch = frame_batch.numpy()
        output = model(frame_batch)
        embeddings_list.extend(output)
    # Unflatten all the frames back into audio batches
    embeddings = tf.stack(embeddings_list, axis=0)
    embeddings = tf.reshape(embeddings, (audio_batches, num_frames, model.embedding_size))

    return embeddings, timestamps

def get_scene_embeddings(
    audio: tf.Tensor,
    model: tf.Module,
) -> tf.Tensor:
    embeddings, _ = get_timestamp_embeddings(audio, model, hop_size=SCENE_HOP_SIZE)
    embeddings = tf.reduce_mean(embeddings, axis=1)
    return embeddings
