import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import numpy as np
import random
import time

# Create a Qt application
app = QtWidgets.QApplication([])

# Create a plot window
win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle("Live Energy Graph")
plot = win.addPlot()
plot.setLabel('left', 'Energy')
plot.setLabel('bottom', 'Time')

# Initialize data
x = np.arange(100)
y = [random.uniform(10, 90) for _ in range(100)]
curve = plot.plot(x, y, pen='b')

# Function to update the plot
def update():
    global x, y
    y.pop(0)  # Remove oldest value
    y.append(random.uniform(10, 90))  # Add new value
    curve.setData(y)  # Update plot

# Timer to update the plot every 100ms
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(100)

# Start Qt event loop
app.exec()