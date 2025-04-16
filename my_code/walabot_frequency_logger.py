"""
walabot_frequency_visualisation.py

Description:
    Live graphing and logging of Walabot frequency data for later classification.

Author: Andy Roberts
Created: 2025-04-10
Version: 1.0

Usage:
    Run this script to visualize and log signal data from a Walabot sensor in real-time.

Dependencies:
    - pyqtgraph
    - numpy
    - datetime
    - csv
    - Walabot SDK

"""
from __future__ import print_function # WalabotAPI works on both Python 2 an 3.
import sys
sys.path.append('/usr/share/walabot/python')
sys.path.append('C:/Program Files/Walabot/WalabotSDK/python/WalabotAPI.py')
from sys import platform
from os import system
import importlib.util
from os.path import join, exists

# below used for graphing
import pyqtgraph as pg
from WalabotAPI import AntennaPair
from pyqtgraph.Qt import QtWidgets
import numpy as np


# Create a Qt application
app = QtWidgets.QApplication([])

# Create a plot window
win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle("Live Signal Graph")
win.setBackground('w')  # 'w' = white, or use '#FFFFFF'
plot = win.addPlot()
plot.getViewBox().setBackgroundColor('w')
plot.setLabel('left', 'Amplitude', units='a.u')
plot.setLabel('bottom', 'Frequency')
plot.setMenuEnabled(True)

# Initialize data
x = np.arange(100)
y = np.zeros(100, dtype=np.float64)

# Configure y-axis for small energy values
plot.setLabel('left', 'Amplitude')  # 'a.u.' = arbitrary units
plot.getAxis('left').setStyle(
    showValues=True,
    tickLength=5,
    textFillLimits=[(0, 0.1)]  # Force scientific notation for small values
)
# Enable auto-scaling with some padding
plot.enableAutoRange(axis='y', enable=True)
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

def load_walabot():
    if platform == 'win32':
        modulePath = join('C:/', 'Program Files', 'Walabot', 'WalabotSDK',
            'python', 'WalabotAPI.py')
    elif platform.startswith('linux'):
        modulePath = join('/usr', 'share', 'walabot', 'python', 'WalabotAPI.py')

    def load_source(module_name, path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    print("Loading source...")
    wlbt = load_source('WalabotAPI', modulePath)
    wlbt.Init()
    print("Starting plotting...")
    # Initializes walabot lib
    wlbt.Initialize()
    # 1) Connect : Establish communication with walabot.
    wlbt.ConnectAny()
    # 2) Configure: Set scan profile and arena
    # Set Profile - to Sensor-Narrow.
    profile = wlbt.PROF_SHORT_RANGE_IMAGING
    wlbt.SetProfile(profile)
    if profile == wlbt.PROF_SENSOR:
        # Distance scanning through air; high-resolution images, but slower capture rate.
        time_units = 10000
        plot.setYRange(0, 0.002)
        plot.setXRange(5e9, 8e9)
        # Walabot_SetArenaR - input parameters
        minInCm, maxInCm, resInCm = 5, 150, 1
        # Walabot_SetArenaTheta - input parameters
        minIndegrees, maxIndegrees, resIndegrees = -30, 30, 2
        # Walabot_SetArenaPhi - input parameters
        minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees = -30, 30, 2
        # Setup arena - specify it by Cartesian coordinates.
        wlbt.SetArenaR(minInCm, maxInCm, resInCm)
        # Sets polar range and resolution of arena (parameters in degrees).
        wlbt.SetArenaTheta(minIndegrees, maxIndegrees, resIndegrees)
        # Sets azimuth range and resolution of arena.(parameters in degrees).
        wlbt.SetArenaPhi(minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees)
    elif profile == wlbt.PROF_SHORT_RANGE_IMAGING:
        # Short-range, penetrative scanning in dielectric materials.
        time_units = 10000
        plot.setYRange(0, 0.01)
        plot.setXRange(1e9, 2e9)
        xmin, xmax, xres = -1.0, 1.0, 0.1
        ymin, ymax, yres = -1.0, 1.0, 0.1
        zmin, zmax, zres = 3.0, 5.0, 0.1

        wlbt.SetArenaX(xmin, xmax, xres)
        wlbt.SetArenaY(ymin, ymax, yres)
        wlbt.SetArenaZ(zmin, zmax, zres)
    elif profile == wlbt.PROF_SHORT_RANGE_SINGLE_LINE:
        pass
    wlbt.SetDynamicImageFilter(wlbt.FILTER_TYPE_NONE)
    wlbt.Start()

    # determine antenna pairs to use
    antenna_pairs = [AntennaPair]
    antennas_all = True
    if antennas_all:
        antenna_pairs = wlbt.GetAntennaPairs()
    else:
        antenna_pairs = [AntennaPair(1, 2), AntennaPair(1, 3)]

    #raw_data= [[0] for _ in range(len(antenna_pairs))]
    raw_data = np.zeros((iterations, len(antenna_pairs), signal_size))
    #fft_data = [[0] for _ in range(len(antenna_pairs))]
    fft_data = np.zeros_like(raw_data)
    curves = [plot.plot(x, y) for _ in antenna_pairs]
    # fudge factor, 10000 implies units of time on x axis are 10ms


    calibration = False
    if calibration:
        wlbt.StartCalibration()
        while wlbt.GetStatus()[0] == wlbt.STATUS_CALIBRATING:
            print("Calibrating ...")
            wlbt.Trigger()
    else:
        print("Calibration off.")

def Timer():
    #Timer to update the plot every 100ms
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(100)
    app.exec()

def EndWalabot():
    wlbt.Stop()
    wlbt.Disconnect()
    wlbt.Clean()
    #csv_file.close()
    print('Terminate successfully')

def get_antenna_pairs():
    print(wlbt.GetAntennaPairs())

def custom_fft(signal):
    """
    Performs custom FFT
    :param s: signal in time domain
    :return: signal in frequency domain
    """
    #print(len(signal[0]), len(signal[1]))
    time_axis = signal[1]
    num_samples = len(signal[1])
    dt = (time_axis[1] - time_axis[0])
    #print(time_axis[0], time_axis[1])
    k = np.arange(num_samples)
    freq = k/(time_units * dt)
    upperbound = int((num_samples)/ 2)
    #print(upperbound)
    freq = freq[0: upperbound + 1]
    fft_vals = np.abs(np.fft.rfft(signal[0]))/num_samples # normalises the FFT

    return freq, fft_vals


# Function to update the plot
def update():
    # get signal value
    del raw_data[0:len(antenna_pairs)]
    wlbt.Trigger()
    for n, pair in enumerate(antenna_pairs):
        signal = wlbt.GetSignal(pair)
        raw_data[iterations, n, :] = signal
        freq, fft_vals = custom_fft(signal)
        fft_data[iterations, n, :] = fft_vals

        # plot ffts
        curve = curves[n]
        color = colors[n % len(colors)]
        curve.setData(freq, fft_vals)
        curve.setPen(color)

    iterations -= 1

    if iterations >= 0:
        print("Reached 100 iterations. Shutting down.")

        # Stop the update timer (if using QTimer)
        if timer.timer:
            timer.stop()

def store_data():
    if not exists('raw_data.npy'):
        np.save('raw_data.npy', raw_data)
        print(f"Saved raw_data of shape {raw_data.shape} to 'raw_data.npy'")
    else:
        # add code to append data
        pass
    if not exists('fft_data.npy'):
        np.save('fft_data.npy', fft_data)
        print(f"Saved fft_data of shape {fft_data.shape} to 'fft_data.npy'")
    else:
        # add code to append data
        pass
    if not exists('labels.npy'):
        np.save('labels.npy', [class_name])
        print(f"Saved labels of shape {labels.shape} to 'labels.npy'")
    else:
        # add code to append data
        pass

if __name__ == '__main__':
    global class_name, iterations, signal_size
    class_name = input("Please enter the classification name:")
    iterations = int(input("Please enter the number of iterations:"))
    signal_size = 8192

    load_walabot()
    try:
        Timer()
        #get_antenna_pairs()
    finally:
        EndWalabot()

    store_data()


