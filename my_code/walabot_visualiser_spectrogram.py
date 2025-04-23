"""
walabot_trained_classifier.py

Description:
    Live reading and classifying of data

Author: Andy Roberts
Created: 2025-04-22
Version: 1.0

Usage:
    Run this script to see what the model thinks the material in front of the walabot is.

Dependencies:
    - pyqtgraph
    - numpy
    - csv
    - Walabot SDK

"""
from __future__ import print_function # WalabotAPI works on both Python 2 an 3.
import sys

sys.path.append('/usr/share/walabot/python')
sys.path.append('C:/Program Files/Walabot/WalabotSDK/python/WalabotAPI.py')
import importlib.util
from os.path import join, exists
from sys import platform
from WalabotAPI import AntennaPair

# for graphing and UI
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import matplotlib.pyplot as plt

#for model usage
import torch
import torch.nn as nn
import torch.nn.functional as F

class WalabotVisualizer:
    def __init__(self, signal_size):
        self.signal_size = signal_size
        self.antenna_pairs = []
        self.timer = None
        self.fft_window_lower_bound = 115
        self.fft_window_upper_bound = 145
        self.fft_window_size = self.fft_window_upper_bound - self.fft_window_lower_bound


    def run(self):
        try:
            self.start_plot()
            self.load_walabot()
            self.start_timer()
            self.app.exec()
        except Exception as e:
            print(f"[ERROR] An exception occurred: {e}")
        finally:
            self.end_walabot()

    def start_plot(self):
        # Create a Qt application
        self.app = QtWidgets.QApplication([])

        # Create a plot window
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle("Live Spectrogram")
        self.win.setBackground('w')  # 'w' = white, or use '#FFFFFF'

        self.plot = self.win.addPlot()
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)

        # Optional: Set color map
        colormap = pg.colormap.get('inferno')  # or 'viridis', 'plasma', etc.
        lut = colormap.getLookupTable(0.0, 1.0, 256)
        self.img.setLookupTable(lut)
        self.img.setLevels([0, 1])  # Set intensity range (adjust as needed)

        # Label axes
        self.plot.setLabel('left', 'Frequency Bin')
        self.plot.setLabel('bottom', 'Antenna Pair')
        self.plot.setYRange(0, 30)
        self.plot.setXRange(0, 153)

        # This holds the spectrogram data history (e.g., last N FFTs)
        #self.spectrogram_buffer = np.zeros((128, 100))

    def start_timer(self):
        # timer to update the plot every 100ms
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)


    def load_walabot(self):
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
        self.wlbt = load_source('WalabotAPI', modulePath)
        self.wlbt.Init()
        print("Starting plotting...")
        # Initializes walabot lib
        self.wlbt.Initialize()
        # 1) Connect : Establish communication with walabot.
        self.wlbt.ConnectAny()
        # 2) Configure: Set scan profile and arena
        # Set Profile - to Sensor-Narrow.
        profile = self.wlbt.PROF_SHORT_RANGE_IMAGING
        self.wlbt.SetProfile(profile)
        if profile == self.wlbt.PROF_SENSOR:
            # Distance scanning through air; high-resolution images, but slower capture rate.
            self.time_units = 10000
            # Walabot_SetArenaR - input parameters
            minInCm, maxInCm, resInCm = 5, 150, 1
            # Walabot_SetArenaTheta - input parameters
            minIndegrees, maxIndegrees, resIndegrees = -30, 30, 2
            # Walabot_SetArenaPhi - input parameters
            minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees = -30, 30, 2
            # Setup arena - specify it by Cartesian coordinates.
            self.wlbt.SetArenaR(minInCm, maxInCm, resInCm)
            # Sets polar range and resolution of arena (parameters in degrees).
            self.wlbt.SetArenaTheta(minIndegrees, maxIndegrees, resIndegrees)
            # Sets azimuth range and resolution of arena.(parameters in degrees).
            self.wlbt.SetArenaPhi(minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees)
        elif profile == self.wlbt.PROF_SHORT_RANGE_IMAGING:
            # Short-range, penetrative scanning in dielectric materials.
            self.time_units = 10000
            xmin, xmax, xres = -2.0, 2.0, 0.1
            ymin, ymax, yres = -2.0, 2.0, 0.1
            zmin, zmax, zres = 2.0, 4.0, 0.1

            self.wlbt.SetArenaX(xmin, xmax, xres)
            self.wlbt.SetArenaY(ymin, ymax, yres)
            self.wlbt.SetArenaZ(zmin, zmax, zres)
        elif profile == self.wlbt.PROF_SHORT_RANGE_SINGLE_LINE:
            pass
        self.wlbt.SetDynamicImageFilter(self.wlbt.FILTER_TYPE_NONE)
        self.wlbt.Start()

        # determine antenna pairs to use
        self.antenna_pairs = self.wlbt.GetAntennaPairs()
        self.fft_data = np.zeros((len(self.antenna_pairs), self.fft_window_size))
        print(f"{len(self.antenna_pairs)} antenna pairs loaded")

        calibration = False
        if calibration:
            self.wlbt.StartCalibration()
            while self.wlbt.GetStatus()[0] == self.wlbt.STATUS_CALIBRATING:
                print("Calibrating ...")
                self.wlbt.Trigger()
        else:
            print("Calibration off.")

    def update(self):
        try:
            self.wlbt.Trigger()
        except Exception as e:
            print(f"Failed to trigger walalbot: {e}")
            return

        for n, pair in enumerate(self.antenna_pairs):
            signal = np.array(self.wlbt.GetSignal(pair))
            freq, fft_vals = self.custom_fft(signal)
            #converting to correct format for model eval
            self.fft_data[n, :] = fft_vals[self.fft_window_lower_bound:self.fft_window_upper_bound]

        image = self.fft_data.astype(np.float32)
        # Normalize your data as needed (optional)
        signal_spectrogram = np.abs(image)
        signal_spectrogram = signal_spectrogram / np.max(signal_spectrogram)

        # Update the plot image
        self.img.setImage(signal_spectrogram, autoLevels=False)


    def get_antenna_pairs(self):
        print(self.wlbt.GetAntennaPairs())

    def custom_fft(self, signal):
        """
        Performs custom FFT
        :param s: signal in time domain
        :return: signal in frequency domain
        """
        # print(len(signal[0]), len(signal[1]))
        time_axis = signal[1]
        num_samples = len(signal[1])
        dt = (time_axis[1] - time_axis[0])
        # print(time_axis[0], time_axis[1])
        k = np.arange(num_samples)
        freq = k / (self.time_units * dt)
        upperbound = int((num_samples) / 2)
        # print(upperbound)
        freq = freq[0: upperbound + 1]
        fft_vals = np.abs(np.fft.rfft(signal[0])) / num_samples  # normalises the FFT

        return freq, fft_vals

    def end_walabot(self):
        if hasattr(self, 'wlbt'):
            try:
                self.wlbt.Stop()
                self.wlbt.Disconnect()
                self.wlbt.Clean()
                print('Terminated successfully.')
            except Exception as e:
                print(f"[WARNING] Could not shut down Walabot cleanly: {e}")

if __name__ == '__main__':
    print("Walabot will start and output the class name it thinks it is 'seeing'")
    signal_size = 1025
    visualizer = WalabotVisualizer(signal_size)
    #start up walabot functionality
    visualizer.run()



