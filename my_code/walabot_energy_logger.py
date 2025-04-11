"""
walabot_energy_logger.py

Description:
    Live graphing and logging of Walabot energy data.

Author: Andy Roberts
Created: 2025-04-10
Version: 1.0

Usage:
    Run this script to visualize and log energy data from a Walabot sensor in real-time.

Dependencies:
    - pyqtgraph
    - numpy
    - datetime
    - csv
    - Walabot SDK

"""
from __future__ import print_function # WalabotAPI works on both Python 2 an 3.
from sys import platform
from os import system
import importlib.util
from os.path import join, exists

# below used for graphing
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import numpy as np

# below used for logging
import csv
from datetime import datetime

#this can be ran from terminal, if not it just complains of TERM variable not being set
# it returns the total energy value from the whole image

# Setup logging
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
csv_filename = f'walabot_energy_log_{timestamp}.csv'
is_new_file = not exists(csv_filename)
csv_file = open(csv_filename, mode='a', newline='')
csv_writer = csv.writer(csv_file)

# Write header if it's a new file
if is_new_file:
    csv_writer.writerow(['Timestamp', 'Energy'])

# Create a Qt application
app = QtWidgets.QApplication([])

# Create a plot window
win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle("Live Energy Graph")
win.setBackground('w')  # 'w' = white, or use '#FFFFFF'
plot = win.addPlot()
plot.getViewBox().setBackgroundColor('w')
plot.setLabel('left', 'Energy')
plot.setLabel('bottom', 'Time')
plot.setMenuEnabled(True)

# Initialize data
x = np.arange(100)
y = np.zeros(100, dtype=np.float64)
curve = plot.plot(x, y, pen='b', )

# Configure y-axis for small energy values
plot.setLabel('left', 'Energy', units='a.u.')  # 'a.u.' = arbitrary units
plot.getAxis('left').setStyle(
    showValues=True,
    tickLength=5,
    textFillLimits=[(0, 0.1)]  # Force scientific notation for small values
)

# Enable auto-scaling with some padding
plot.enableAutoRange(axis='y', enable=True)
plot.setYRange(0, max(1e-5, y.max()*1.1))  # Start with minimum 0.0001 range


# Function to update the plot
def update():
    # get energy value
    print("getting energy")
    wlbt.Trigger()
    energy = wlbt.GetImageEnergy()
    print("energy = ", energy)

    #update graph
    print("updating")
    global x, y
    y = np.roll(y, -1)  # Shift values left
    y[-1] = energy    # Add new value at the end
    curve.setData(y)    # Update plot

    current_max = max(y) if y.any() else 1e-5
    plot.setYRange(0, max(1e-5, current_max*1.1))

    # Log to CSV
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    csv_writer.writerow([timestamp, energy])
    csv_file.flush()

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
# Walabot_SetArenaR - input parameters
minInCm, maxInCm, resInCm = 5, 150, 1
# Walabot_SetArenaTheta - input parameters
minIndegrees, maxIndegrees, resIndegrees = -30, 30, 2
# Walabot_SetArenaPhi - input parameters
minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees = -30, 30, 2
# Initializes walabot lib
wlbt.Initialize()
# 1) Connect : Establish communication with walabot.
wlbt.ConnectAny()
# 2) Configure: Set scan profile and arena
# Set Profile - to Sensor-Narrow.
wlbt.SetProfile(wlbt.PROF_SENSOR)
# Setup arena - specify it by Cartesian coordinates.
wlbt.SetArenaR(minInCm, maxInCm, resInCm)
# Sets polar range and resolution of arena (parameters in degrees).
wlbt.SetArenaTheta(minIndegrees, maxIndegrees, resIndegrees)
# Sets azimuth range and resolution of arena.(parameters in degrees).
wlbt.SetArenaPhi(minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees)
# Dynamic-imaging filter for the specific frequencies typical of breathing
wlbt.SetDynamicImageFilter(wlbt.FILTER_TYPE_NONE)
# 3) Start: Start the system in preparation for scanning.

wlbt.StartCalibration()
while wlbt.GetStatus()[0] == wlbt.STATUS_CALIBRATING:
    wlbt.Trigger()

wlbt.Start()


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
    csv_file.close()
    print('Terminate successfully')


if __name__ == '__main__':
    try:
        Timer()
    finally:
        EndWalabot()


