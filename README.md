# Walabot-Imaging

Code here uses the walabot python api to perform various functions that have not been readily available through the example pack or found from projects online. 

The main functions are:

### Walabot 3d interactive live plot
```
walabot_3d_viewer.py
walabot_3d_viewer_pyqt.py
```
Running these scripts pops up an interactive pyqt-based window to visualise the walabot imaging data in 3d.


### Walabot Frequency Visualisation
`walabot_frequency_visualisation.py` - pops up live pyqt plot showing Fast Fourier Transformed FFT waveforms per antenna pair
`walabot_frequency_spectrogram.py` - pops up live pyqt plot showing FFT data in spectrogram form where x axis is antenna pair and y axis is frequency- indiciative (not guaranteed atm to be the actual frequency)

### Walabot Frequency Data Collection
`walabot_frequency_logger` - can be run from CLI, when run user inputs classifcation type, to be later used in ResNet 18 model, and number of iterations. Script will run that many iterations, recording all FFT and label data into .npy format to be later used for modelling. Subsequent runs will append  to the existing .npy data. Raw data can also be caputred but is currently commented out as takes up a lot of space. 

### Data Analysis
`resnet18_spectrogram_classifier.ipynb` - the fft_data.npy and labels.npy are used in a ResNet18 model that trains on the data in the form of spectrogram images, to classify the label names. The model can then be saved and used for inference with...
`walabot_trained_classifer.py` - uses the above trained model, shows a spectrogram and infers from that what material is being 'visualised'. WIP
