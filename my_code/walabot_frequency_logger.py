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

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

class WalabotVisualizer:
    def __init__(self, class_name, iterations, signal_size):
        self.class_name = class_name
        self.iterations = iterations
        self.signal_size = signal_size
        self.antenna_pairs = []
        self.curves = []
        self.timer = None

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
            self.store_data(self.raw_data, 'raw_data')
            self.store_data(self.fft_data, 'fft_data')
            self.store_data(self.labels, 'labels')

    def start_plot(self):
        # Create a Qt application
        self.app = QtWidgets.QApplication([])

        # Create a plot window
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle("Live Signal Graph")
        self.win.setBackground('w')  # 'w' = white, or use '#FFFFFF'
        self.plot = self.win.addPlot()
        self.plot.getViewBox().setBackgroundColor('w')
        self.plot.setLabel('left', 'Amplitude', units='a.u')
        self.plot.setLabel('bottom', 'Frequency')
        self.plot.setMenuEnabled(True)

        # Initialize data
        self.x = np.arange(100)
        self.y = np.zeros(100, dtype=np.float64)

        # Configure y-axis for small energy values
        self.plot.setLabel('left', 'Amplitude')  # 'a.u.' = arbitrary units
        self.plot.getAxis('left').setStyle(
            showValues=True,
            tickLength=5,
            textFillLimits=[(0, 0.1)]  # Force scientific notation for small values
        )
        # Enable auto-scaling with some padding
        self.plot.enableAutoRange(axis='y', enable=True)
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

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
            self.plot.setYRange(0, 0.002)
            self.plot.setXRange(5e9, 8e9)
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
            self.plot.setYRange(0, 0.02)
            self.plot.setXRange(1e9, 2e9)
            xmin, xmax, xres = -1.0, 1.0, 0.1
            ymin, ymax, yres = -1.0, 1.0, 0.1
            zmin, zmax, zres = 3.0, 5.0, 0.1

            self.wlbt.SetArenaX(xmin, xmax, xres)
            self.wlbt.SetArenaY(ymin, ymax, yres)
            self.wlbt.SetArenaZ(zmin, zmax, zres)
        elif profile == self.wlbt.PROF_SHORT_RANGE_SINGLE_LINE:
            pass
        self.wlbt.SetDynamicImageFilter(self.wlbt.FILTER_TYPE_NONE)
        self.wlbt.Start()

        # determine antenna pairs to use
        self.antenna_pairs = self.wlbt.GetAntennaPairs()
        print(f"{len(self.antenna_pairs)} antenna pairs loaded")
        self.curves = [self.plot.plot(self.x, self.y) for _ in self.antenna_pairs]
        self.raw_data = np.zeros((iterations, len(self.antenna_pairs), 2, 2048))
        self.fft_data = np.zeros((iterations, len(self.antenna_pairs), self.signal_size))
        self.labels = np.array([self.class_name]* self.iterations)

        calibration = False
        if calibration:
            self.wlbt.StartCalibration()
            while self.wlbt.GetStatus()[0] == self.wlbt.STATUS_CALIBRATING:
                print("Calibrating ...")
                self.wlbt.Trigger()
        else:
            print("Calibration off.")

    def update(self):
        # get signal value
        if self.iterations <= 0:
            self.timer.stop()
            print("Reached specified iterations. Please close the graph window.")
            return

        try:
            self.wlbt.Trigger()
        except Exception as e:
            print(f"Failed to trigger walalbot: {e}")
            return

        for n, pair in enumerate(self.antenna_pairs):
            signal = np.array(self.wlbt.GetSignal(pair))
            self.raw_data[self.iterations - 1, n,:,:] = signal
            freq, fft_vals = self.custom_fft(signal)
            self.fft_data[self.iterations - 1, n, :] = fft_vals
            self.curves[n] .setData(freq, fft_vals)

            # plot ffts
            curve = self.curves[n]
            color = self.colors[n % len(self.colors)]
            curve.setPen(color)

        self.iterations -= 1

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

    def store_data(self, data, name):
        filename = f'{name}.npy'
        print(f"Saving {filename}.")
        if not exists(filename):
            np.save(filename, data)
            print(f"Saved data to '{filename}'")
        else:
            existing = np.load(filename)
            if existing.shape[1:] != data.shape[1:]:
                raise ValueError("Shape mismatch when appending data.")
            combined = np.concatenate((existing, data), axis=0)
            np.save(filename, combined)
            print(f"Appended data to '{filename}', new shape: {combined.shape}")

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
    try:
        class_name = input("Please enter the classification name:")
    except ValueError as e:
        print(f"[ERROR] Invalid input: {e}")
        sys.exit(1)
    try:
        iterations = int(input("Please enter the number of iterations: "))
        if iterations <= 0:
            raise ValueError("Number of iterations must be positive.")
    except ValueError as e:
        print(f"[ERROR] Invalid input: {e}")
        sys.exit(1)
    signal_size = 1025
    visualizer = WalabotVisualizer(class_name, iterations, signal_size)
    visualizer.run()


