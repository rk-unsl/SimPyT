from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Audio
from scipy.io import wavfile
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

'''
 General functions:
   normalize_u(x)
   normalize_b(x)
   power(x)
   energy(x)
   
 plotting functions:
   plot_time(x, fs, xlim)
   plot_frequency(x, fs, xlim)
   plot_frequency_phase(x, fs, xlim)
   
   create_spectrogram(x, fs, window_size, overlap)
   bandpass_filter(X, fs, finf, fsup)
'''

def normalize_u(x):
  '''
    Rescales the input values such that the output values are between 0 and 1 (unipolar).
  '''
  xn = x - min(x)
  xn = xn / max(xn)
  return xn

def normalize_b(x):
  '''
     Rescales the input values such that the output values are between -1 and 1 (bipolar).
  '''
  xn = 2 * normalize_u(x) - 1
  return xn

def power(x):
    '''
    Power of a vector of samples:
    Sum of the squares of the elements of the vector x divided by the length of the vector.
    Valid for real and complex inputs, in the time and frequency domain.
    '''
    p = np.sum(x * x.conjugate()) / len(x)
    return p


def energy(x):
    '''
    Energy of a vector of samples:
    Sum of the squares of the elements of the vector x.
    Valid for real and complex inputs, in the time and frequency domain.
    '''
    e = np.sum(x * x.conjugate())
    return e

def plot_time(x, fs, xlim):
    '''
       Plots in the time domain.
       First argument, array of function values over time.
       Second argument, sampling frequency.
       Third argument, time interval to plot [tmin, tmax].
    '''
    tmax = len(x) / fs
    t = np.arange(0, tmax, step=1. / fs)
    plt.plot(t, x)
    if len(xlim) >= 2:
      plt.xlim(xlim) 
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    return

  def plot_frequency(x,fs,xlim):
    '''
       Plots in the frequency domain, double-sided amplitude spectrum.
       First argument, array of function values over time.
       Second argument, sampling frequency.
       Third argument, frequency range to plot [fmin, fmax].
       Returns the complex Fourier coefficients.
    '''
    N = len(x)
    X = np.fft.fft(x)/N
    # Plot the positive frequencies.
    freqsp = np.arange(0, fs / 2, step=fs / N)
    plt.plot(freqsp, np.abs(X[:(N // 2)]))
    # Plot the negative frequencies.
    freqsn = np.arange(-fs / 2, 0, step=fs / N)
    plt.plot(freqsn, np.abs(X[(N // 2):]))
    # Now we can label the x-axis.
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    if len(xlim) >= 2:
      plt.xlim(xlim) 
    return X

def plot_frequency_phase(x, fs, xlim):
    '''
       Plots in the frequency domain, double-sided phase spectrum.
       First argument, array of function values over time.
       Second argument, sampling frequency.
       Third argument, frequency range to plot [fmin, fmax].
       Returns the complex Fourier coefficients.
    '''
    N = len(x)
    X = np.fft.fft(x)/N
    # Plot the positive frequencies.
    freqsp = np.arange(0, fs / 2, step=fs / N)
    plt.plot(freqsp, np.angle(X[:(N // 2)]))
    # Plot the negative frequencies.
    freqsn = np.arange(-fs / 2, 0, step=fs / N)
    plt.plot(freqsn, np.angle(X[(N // 2):]))
    # Now we can label the x-axis.
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase")
    if len(xlim) >= 2:
      plt.xlim(xlim) 
    return X
  
  import numpy as np
import matplotlib.pyplot as plt

def create_spectrogram(x, fs, window_size, overlap):
    '''
    Creates a spectrogram from a time samples vector.

    Arguments:
    x: Array of time samples.
    fs: Sampling frequency.
    window_size: Size of the analysis window (in samples).
    overlap: Overlap between consecutive windows (as a fraction, e.g., 0.5 for 50% overlap).

    Returns:
    spectrogram: 2D array representing the spectrogram.
    freqs: Array of frequencies.
    times: Array of time instances.
    '''
    # Calculate the number of overlapping samples
    hop_size = int(window_size * (1 - overlap))
    # Calculate the number of time windows
    num_windows = (len(x) - window_size) // hop_size + 1
    # Initialize the spectrogram array
    spectrogram = np.zeros((window_size, num_windows), dtype=complex)
    # Apply the windowing and perform FFT on each window
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        window = np.hanning(window_size)
        windowed_samples = x[start:end] * window
        spectrogram[:, i] = np.fft.fft(windowed_samples)
    # Get the frequencies and time instances
    freqs = np.fft.fftfreq(window_size, 1 / fs)
    times = np.arange(num_windows) * hop_size / fs
    return np.abs(spectrogram), freqs, times
   
def plot_spectrogram(x, fs, window_size, overlap):
    '''
    Plots the spectrogram of a time samples vector.

    Arguments:
    x: Array of time samples.
    fs: Sampling frequency.
    window_size: Size of the analysis window (in samples).
    overlap: Overlap between consecutive windows (as a fraction, e.g., 0.5 for 50% overlap).
    '''
    # Create the spectrogram
    spectrogram, freqs, times = create_spectrogram(x, fs, window_size, overlap)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(spectrogram), aspect='auto', origin='lower', cmap='jet', extent=[times[0], times[-1], freqs[0], freqs[-1]])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.show()

def plot_spectrogram_3d(x, fs, window_size, overlap):
    '''
    Plots a 3D spectrogram of a time samples vector.

    Arguments:
    x: Array of time samples.
    fs: Sampling frequency.
    window_size: Size of the analysis window (in samples).
    overlap: Overlap between consecutive windows (as a fraction, e.g., 0.5 for 50% overlap).
    '''
    # Create the spectrogram
    spectrogram, freqs, times = create_spectrogram(x, fs, window_size, overlap)

    # Create mesh grid for spectrogram plot
    times_mesh, freqs_mesh = np.meshgrid(times, freqs)

    # Plot the spectrogram in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(times_mesh, freqs_mesh, np.abs(spectrogram), cmap='jet')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_zlabel('Magnitude')
    ax.set_title('3D Spectrogram')
    plt.show()

def plot_spectrogram_2d(x, fs, window_size, overlap):
    '''
    Plots a 2D spectrogram of a time samples vector.

    Arguments:
    x: Array of time samples.
    fs: Sampling frequency.
    window_size: Size of the analysis window (in samples).
    overlap: Overlap between consecutive windows (as a fraction, e.g., 0.5 for 50% overlap).
    '''
    # Create the spectrogram
    spectrogram, freqs, times = create_spectrogram(x, fs, window_size, overlap)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(spectrogram), aspect='auto', origin='lower', cmap='jet', extent=[times[0], times[-1], freqs[0], freqs[-1]])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('2D Spectrogram')
    plt.show()


def bandpass_filter(X, fs, finf, fsup):
    '''
    Ideal bandpass filter for frequency components in X,
    with sampling frequency fs, and lower and upper frequencies
    of the passband finf and fsup.
    '''
    # Shift frequency samples
    X2 = np.fft.fftshift(X)
    # Calculate frequency component indices
    nc = len(X)
    ifni = int(-fsup * nc / fs + nc // 2)
    ifns = int(-finf * nc / fs + nc // 2)
    ifpi = int(finf * nc / fs + nc // 2)
    ifps = int(fsup * nc / fs + nc // 2)
    # Eliminate components outside the passband
    X2[:ifni] = 0
    X2[ifns:ifpi] = 0
    X2[ifps:] = 0
    # Shift frequency samples back
    X3 = np.fft.ifftshift(X2)
    return X3

def triangular_signal(fs, periodo, tmax=1, tipo='t', polaridad='b'):
    '''
    Triangular or sawtooth signal:
    - First argument: sampling frequency
    - Second argument: period of the sawtooth
    - Third argument: simulation time
    - Fourth argument: type: 't'riangular /\/\/\, 'c'rescent  /|/|/|, 'd'ecrescent  |\|\|\
    - Fifth argument: polarity: 'u'nipolar (0,1), 'b'ipolar (-1,1)
    - Returns x, t: function values, time values
    '''
    t = np.arange(0, tmax, step=1. / fs)
    if tipo == 'c':
        x = np.mod(t, periodo)
    elif tipo == 'd':
        x = periodo - np.mod(t, periodo)
    elif tipo == 't':
        x = 1 - np.abs(periodo / 2 - np.mod(t, periodo))
  
    if polaridad == 'u':
        x = normalize(x)
    else:
        x = normalizeb(x)
 
    return x, t


def pulse_signal(fs, periodo, ciclo, tmax=1, polaridad='b'):
    '''
    Pulse signal:
    - First argument: sampling frequency
    - Second argument: period of the signal
    - Third argument: duty cycle (between 0 and 1)
    - Fourth argument: simulation time
    - Fifth argument: polarity: 'u'nipolar (0,1), 'b'ipolar (-1,1)
    - Returns x, t: function values, time values
    '''
    t = np.arange(0, tmax, step=1. / fs)
    xx = t - t
    x = normalize(np.mod(t, periodo))
    xx[x < ciclo] = 1
  
    if polaridad == 'u':
        xx = normalize(xx)
    else:
        xx = normalizeb(xx)
  
    return xx, t


def cosine_signal(fs, f, tmax=1, a=1, ph=0):
    '''
    Cosine signal:
    - First argument: sampling frequency
    - Second argument: frequency of the cosine
    - Third argument: simulation time
    - Fourth argument: amplitude
    - Fifth argument: phase in radians
    - Returns x, t: function values, time values
    '''
    t = np.arange(0, tmax

, step=1. / fs)
    x = a * np.cos(2 * np.pi * f * t + ph)
    return x, t



