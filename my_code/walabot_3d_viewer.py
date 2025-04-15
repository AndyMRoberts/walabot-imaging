"""
walabot_3d_viewer.py

Description:
    Live 3d view of walabot data

Author: Andy Roberts
Created: 2025-04-10
Version: 1.0

Usage:
    Run this script to visualize walabot imaging

Dependencies:
    - pyvista
    - Walabot SDK

"""
from __future__ import print_function # WalabotAPI works on both Python 2 an 3.
from sys import platform
from os import system
import importlib.util
from os.path import join, exists
import pyvista as pv
import numpy as np
import math
import time
from pyvista import ImageData
from pyvista import examples

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

# Set walabot volume of scan parameters, the smaller and lower res the area, the fast the scan
range_cm = 150
resolution_r = 2
all_angles = 45
resolution_angle = 2

print("Loading source...")
wlbt = load_source('WalabotAPI', modulePath)
wlbt.Init()

print("Starting plotting...")
# Walabot_SetArenaR - input parameters
minInCm, maxInCm, resInCm = 5, range_cm, resolution_r
range_r = np.arange(minInCm, maxInCm, resInCm)
len_r = len(range_r)
# Walabot_SetArenaTheta - input parameters
minIndegrees, maxIndegrees, resIndegrees = -all_angles, all_angles, resolution_angle
range_theta = np.arange(minIndegrees, maxIndegrees, resIndegrees)
len_theta = len(range_theta)
# Walabot_SetArenaPhi - input parameters
minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees = -all_angles, all_angles, resolution_angle
range_phi = (np.arange(minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees))
len_phi = len(range_phi)
# Initializes walabot lib
wlbt.Initialize()
# 1) Connect : Establish communication with walabot.
wlbt.ConnectAny()
# 2) Configure: Set scan profile and arena
# Set Profile - to Sensor-Narrow.
wlbt.SetProfile(wlbt.PROF_SENSOR)
print(f"Threshold: {wlbt.GetThreshold()}")
# Setup arena - specify it by Cartesian coordinates.
wlbt.SetArenaR(minInCm, maxInCm, resInCm)
# Sets polar range and resolution of arena (parameters in degrees).
wlbt.SetArenaTheta(minIndegrees, maxIndegrees, resIndegrees)
# Sets azimuth range and resolution of arena.(parameters in degrees).
wlbt.SetArenaPhi(minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees)
# Dynamic-imaging filter for the specific frequencies typical of breathing
wlbt.SetDynamicImageFilter(wlbt.FILTER_TYPE_NONE)
# 3) Start: Start the system in preparation for scanning.
wlbt.Start()

calibration = True

if calibration:
    print("Starting Calibration...")
    wlbt.StartCalibration()
    while wlbt.GetStatus()[0] == wlbt.STATUS_CALIBRATING:
        print("Calibrating...")
        wlbt.Trigger()
else:
    print("Calibration Off")

def EndWalabot():
    wlbt.Stop()
    wlbt.Disconnect()
    wlbt.Clean()
    print('Terminate successfully')

# Initial volume
def generate_volume():
    """ Updates the canvas cells colors acorrding to a given rawImage
        matrix and it's dimensions.
        Arguments:
            rawImage    A 2D matrix contains the current rawImage slice.
            lenOfPhi    Number of cells in Phi axis.
            lenOfR      Number of cells in R axis.
    """
    start_time = time.time()
    wlbt.Trigger()
    image_3d = wlbt.GetRawImage()[0]
    end_time = time.time()
    print(f"grab time: {end_time - start_time}")
    points = np.zeros((len_r * len_phi *len_theta, 3))
    power = np.zeros_like(points)
    point = 0
    for idx_i, i in enumerate(range_r):
        for idx_j, j in enumerate(range_theta):
            for idx_k, k in enumerate(range_phi):
                # convert from spherical to cartesian
                points[point,0:3] = (i * np.sin(math.radians(j)), i * np.cos(math.radians(j)) * np.sin(math.radians(k)), i * np.cos(math.radians(j)) * np.cos(math.radians(k)))
                power[point] = image_3d[idx_j][idx_k][idx_i]
                point += 1

    point_cloud = pv.PolyData(points)
    point_cloud["Relected Power"] = power
    return point_cloud

if __name__ == '__main__':
    # Prepare the graph and show the first result
    #p = pv.Plotter()
    p = pv.Plotter()
    point_cloud = generate_volume()
    mesh_actor = p.add_mesh(point_cloud, cmap='jet', clim=[50, 255], opacity="linear")
    p.camera_position = [-2, 5, 3]
    #p.show_axes()
    #p.show_bounds()
    p.show(interactive_update=True, auto_close=False)
    time.sleep(1)

    i = 0
    while True:
        #grab the next radar image and update the point cloud
        print(i)
        print(f"PARAM_CONFIDENCE_FACTOR: {wlbt.GetAdvancedParameter(wlbt.PARAM_CONFIDENCE_FACTOR)}")
        print(f"PARAM_DIELECTRIC_CONSTANT: {wlbt.GetAdvancedParameter(wlbt.PARAM_DIELECTRIC_CONSTANT)}")
        start_time = time.time()
        point_cloud = generate_volume()
        p.remove_actor(mesh_actor)
        mesh_actor = p.add_mesh(point_cloud, cmap='jet', clim=[50, 255], opacity="linear")
        p.show(interactive_update=True, auto_close=False)
        end_time = time.time()
        print(f"volume generation time: {end_time - start_time:.4f} seconds")
        start_time_r = time.time()
        p.update()  # Force update the renderer
        p.render()
        # added in as sometimes the render disappears and this seems to fix it
        #time.sleep(0.1)
        end_time_r = time.time()
        print(f"Render Time: {end_time_r - start_time_r:.4f} seconds")
        print(f"FPS: {1 / (end_time_r - start_time):.4f}")
        i += 1


    #plotter.close()
    EndWalabot()
