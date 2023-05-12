from __future__ import print_function
print("importing plotting, audio, and interaction modules")

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

print("defining helper functions")

def normalize(x):
  '''
    Rescales the input values such that the output values are between 0 and 1.
  '''
  xn = x - min(x)
  xn = xn / max(xn)
  return xn


def normalize_between_minus_one_and_one(x):
  '''
     Rescales the input values such that the output values are between -1 and 1.
  '''
  xn = 2 * normalize(x) - 1
  return xn


def plot_time_domain(x, fs, xlim):
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

def plot_single_sided_frequency_domain(x, fs, xlim):
    '''
       Plots in the frequency domain, one-sided spectrum.
       First argument, array of function values over time.
       Second argument, sampling frequency.
       Third argument, frequency range to plot [fmin, fmax].
       Returns the complex Fourier coefficients.
    '''
    N = len(x)
    X = np.fft.fft(x) / N
    freqs = np.arange(0, fs / 2, step=fs / N)
    plt.plot(freqs[:(N // 2)], 2 * np.abs(X[:(N // 2)]))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    if len(xlim) >= 2:
      plt.xlim(xlim) 
    return X

def plot_single_sided_frequency_domain_phase(x, fs, xlim):
    '''
       Plots in the frequency domain, one-sided phase spectrum.
       First argument, array of function values over time.
       Second argument, sampling frequency.
       Third argument, frequency range to plot [fmin, fmax].
       Returns the complex Fourier coefficients.
    '''
    N = len(x)
    X = np.fft.fft(x) / N
    freqs = np.arange(0, fs / 2, step=fs / N)
    plt.plot(freqs[:(N // 2)], np.angle(X[:(N // 2)], deg=True), 'b*')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.ylim([-180, 180]) 
    if len(xlim) >= 2:
        plt.xlim(xlim) 
    return X

def plot_double_sided_frequency_domain_phase(x, fs, xlim):
    '''
       Plots in the frequency domain, double-sided phase spectrum.
       First argument, array of function values over time.
       Second argument, sampling frequency.
       Third argument, frequency range to plot [fmin, fmax].
       Returns the complex Fourier coefficients.
    '''
    N = len(x)
    X = np.fft.fft(x) / N
    # Plot the positive frequencies.
   
