import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#load data from file
fft_data_all = np.load('classification/data/andy_practice/fft_data.npy')
fft_data_single = np.transpose(fft_data_all[119,:,115:145])

print(fft_data_single.shape)

# Compute the spectrogram
#f, t_spec, Sxx = signal.spectrogram(signal_data, fs)
#f, t_spec, Sxx = signal.spectrogram(fft_data_single, fs)

# create plot
plt.imshow(np.abs(fft_data_single), aspect='auto', cmap='viridis', origin='lower')

# Plot the spectrogram
#plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx))  # Convert to dB scale
plt.ylabel('Frequency [Hz]')
plt.xlabel('Antenna')
plt.colorbar(label='Signal Intensity')
plt.title('Spectrogram')
plt.show()
