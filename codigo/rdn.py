import os, pathlib
import wave

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_io as tfio

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers, models
from IPython import display

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

URL = 'https://drive.upm.es/index.php/s/xFNkMwK7CsufEod/download'
data_dir = pathlib.Path('data/Casos')

if not data_dir.exists():
    tf.keras.utils.get_file(
        'Casos.zip',
        URL,
        extract = True,
        cache_dir ='.', cache_subdir = 'data'
    )

data_train = pathlib.Path('data/train')
cases_train = np.array(tf.io.gfile.listdir(str(data_train)))
# print('cases_train: ', cases_train)

full_data_test  = 'data/test/*/*/*'
full_data_train = 'data/train/*/*/*'

filenames_train = tf.io.gfile.glob(full_data_train)
filenames_test  = tf.io.gfile.glob(full_data_test)

filenames_train = tf.random.shuffle(filenames_train)
filenames_test  = tf.random.shuffle(filenames_test)


num_samples_train = len(filenames_train)
num_samples_test  = len(filenames_test)

# Strings formatted to show the total number of examples
print(f'\nNumber of total examples from train: {num_samples_train}')
print(f'Number of total examples from test: {num_samples_test}\n')

# ========================================== #
# ===== Reading files with flac format ===== #
# ========================================== #
FRECUENCIA_MUESTREO = 16000
N_RASGOS = 64
LONGITUD_MS_VENTANA_ANALISIS = 200

def abrirFicheroAudio_flac(fichero=None):

    audio_binary = tf.io.read_file(fichero)
    audio = tfio.audio.decode_flac(audio_binary, shape=None, dtype = tf.int16, name=None)
    audio_f = tf.cast(tf.squeeze(audio), dtype = tf.float32)

    return(audio_f)

def obtenerRasgosAudio(audio=None):

    print(type(audio))
    print(type(audio[1:3]))

    y_pre = tf.math.subtract(audio[1:], tf.multiply(0.97, audio[0:-1]))

    S = tfio.experimental.audio.spectrogram(y_pre, nfft=512, window=512, stride=160)
    S_mel = tfio.experimental.audio.melscale(S, rate=FRECUENCIA_MUESTREO, mels=N_RASGOS, fmin=50, fmax=8000)
    S_mel_db = tfio.experimental.audio.dbscale(S_mel, top_db=120)

    mu = tf.math.reduce_mean(S_mel_db, axis=0)
    std = tf.math.reduce_std(S_mel_db, axis=0)

    S_mel_db_norm = tf.transpose(tf.divide(tf.math.subtract(S_mel_db,mu), tf.clip_by_value(std, 1.0, 1000.0)))

    n_vectores = tf.shape(S_mel_db_norm)[1]

    if n_vectores<LONGITUD_MS_VENTANA_ANALISIS:
        n_repeticiones = tf.cast(tf.math.floordiv(LONGITUD_MS_VENTANA_ANALISIS,n_vectores), tf.int32)+1
        rasgos_ext = tf.tile(S_mel_db_norm, [1,n_repeticiones])
        fragmento = rasgos_ext[:,0:LONGITUD_MS_VENTANA_ANALISIS]
    else:
        tramaInicial = tf.random.uniform(shape=[], minval=0, maxval=n_vectores-LONGITUD_MS_VENTANA_ANALISIS+1, dtype=tf.int32)
        fragmento = S_mel_db_norm[:,tramaInicial:tramaInicial+LONGITUD_MS_VENTANA_ANALISIS]

    fragmento_ext = tf.expand_dims(fragmento, axis=0)
    # print(f'Numero de shape: {fragmento_ext.shape.as_list()}')

    return (fragmento_ext)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-3]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    waveform = abrirFicheroAudio_flac(file_path)

    return waveform, label

def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([90000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the same length
    waveform = tf.cast(waveform, tf.float32)

    equal_length = tf.concat([waveform, zero_padding], 0)

    spectrogram = tf.signal.stft (
        equal_length, frame_length=255, frame_step=128
    )
    spectrogram = tf.abs(spectrogram)

    return spectrogram

def plot_spectrogram(spectrogram, ax):
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


files_ds  = tf.data.Dataset.from_tensor_slices(filenames_train)
waveform_ds = files_ds.map(get_waveform_and_label)

rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(18, 20))
for i, (audio, label) in enumerate(waveform_ds.take(n)):
  r = i // cols
  c = i % cols
  ax = axes[r][c]
  ax.plot(audio.numpy())
  ax.set_yticks(np.arange(-40000, 45000, 10000))
  label = label.numpy().decode('utf-8')
  ax.set_title(label)

# plt.show()

n = 1
for waveform, label in waveform_ds.take(n):

    try:
        label = label.numpy().decode('utf-8')
        spectrogram = get_spectrogram(waveform)
    except Exception:
        n += 1

fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 90000])
plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()
