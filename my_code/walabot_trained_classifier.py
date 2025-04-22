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

def load_model():
    global model
    global labels
    global DEVICE

    labels = ['pillow', 'air', 'soil', 'desk']

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Architecture
    NUM_FEATURES = 28 * 28
    NUM_CLASSES = 4

    # Other
    DEVICE = "cuda:0"  # "cpu"#"cuda:1"
    GRAYSCALE = True

    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(BasicBlock, self).__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

    class ResNet(nn.Module):

        def __init__(self, block, layers, num_classes, grayscale):
            self.inplanes = 64
            if grayscale:
                in_dim = 1
            else:
                in_dim = 3
            super(ResNet, self).__init__()
            self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(2560 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, (2. / n) ** .5)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            # because MNIST is already 1x1 here:
            # disable avg pooling
            # x = self.avgpool(x)

            x = x.view(x.size(0), -1)
            logits = self.fc(x)
            probas = F.softmax(logits, dim=1)
            return logits, probas

    def resnet18(num_classes):
        """Constructs a ResNet-18 model."""
        model = ResNet(block=BasicBlock,
                       layers=[2, 2, 2, 2],
                       num_classes=NUM_CLASSES,
                       grayscale=GRAYSCALE)
        return model

    model = resnet18(NUM_CLASSES)

    # Load the saved weights
    model.load_state_dict(torch.load("classification/resnet_spectrogram_simple_4000_weights.pth"))
    # for full model unpickling
    #model = torch.load("classification/resnet_spectrogram_simple_4000_full.pth", weights_only=False)
    model.to(DEVICE)
    # Step 3: Set model to evaluation mode
    model.eval()




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
        #plt.imshow(self.img)

        # continue to prep for model usage
        image = np.expand_dims(image, axis=0) # shape [1,153,30]
        image = np.expand_dims(image, axis=0) # shape [1,1,153,30]
        #image = np.transpose(image, (0, 1, 3, 2))
        #do i need to normalise here?
        image = torch.from_numpy(image).float()
        image = image.to(DEVICE)
        #print(image.shape)
        # eval using model
        with torch.no_grad():
            logits, probas = model(image)

            # Get probabilities if you want them (optional)
            predicted_index = torch.argmax(probas, dim=1).item()

            # Convert to class label using your dataset’s index-to-label mapping
            predicted_label = labels[predicted_index]

            print(f"Predicted class: {predicted_label}  of probabilities {probas})")


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
    # load resnet 18 pre-trained model
    load_model()
    #start up walabot functionality
    visualizer.run()



