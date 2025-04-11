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

COLORS = [
    "000083", "000087", "00008B", "00008F", "000093", "000097", "00009B",
    "00009F", "0000A3", "0000A7", "0000AB", "0000AF", "0000B3", "0000B7",
    "0000BB", "0000BF", "0000C3", "0000C7", "0000CB", "0000CF", "0000D3",
    "0000D7", "0000DB", "0000DF", "0000E3", "0000E7", "0000EB", "0000EF",
    "0000F3", "0000F7", "0000FB", "0000FF", "0003FF", "0007FF", "000BFF",
    "000FFF", "0013FF", "0017FF", "001BFF", "001FFF", "0023FF", "0027FF",
    "002BFF", "002FFF", "0033FF", "0037FF", "003BFF", "003FFF", "0043FF",
    "0047FF", "004BFF", "004FFF", "0053FF", "0057FF", "005BFF", "005FFF",
    "0063FF", "0067FF", "006BFF", "006FFF", "0073FF", "0077FF", "007BFF",
    "007FFF", "0083FF", "0087FF", "008BFF", "008FFF", "0093FF", "0097FF",
    "009BFF", "009FFF", "00A3FF", "00A7FF", "00ABFF", "00AFFF", "00B3FF",
    "00B7FF", "00BBFF", "00BFFF", "00C3FF", "00C7FF", "00CBFF", "00CFFF",
    "00D3FF", "00D7FF", "00DBFF", "00DFFF", "00E3FF", "00E7FF", "00EBFF",
    "00EFFF", "00F3FF", "00F7FF", "00FBFF", "00FFFF", "03FFFB", "07FFF7",
    "0BFFF3", "0FFFEF", "13FFEB", "17FFE7", "1BFFE3", "1FFFDF", "23FFDB",
    "27FFD7", "2BFFD3", "2FFFCF", "33FFCB", "37FFC7", "3BFFC3", "3FFFBF",
    "43FFBB", "47FFB7", "4BFFB3", "4FFFAF", "53FFAB", "57FFA7", "5BFFA3",
    "5FFF9F", "63FF9B", "67FF97", "6BFF93", "6FFF8F", "73FF8B", "77FF87",
    "7BFF83", "7FFF7F", "83FF7B", "87FF77", "8BFF73", "8FFF6F", "93FF6B",
    "97FF67", "9BFF63", "9FFF5F", "A3FF5B", "A7FF57", "ABFF53", "AFFF4F",
    "B3FF4B", "B7FF47", "BBFF43", "BFFF3F", "C3FF3B", "C7FF37", "CBFF33",
    "CFFF2F", "D3FF2B", "D7FF27", "DBFF23", "DFFF1F", "E3FF1B", "E7FF17",
    "EBFF13", "EFFF0F", "F3FF0B", "F7FF07", "FBFF03", "FFFF00", "FFFB00",
    "FFF700", "FFF300", "FFEF00", "FFEB00", "FFE700", "FFE300", "FFDF00",
    "FFDB00", "FFD700", "FFD300", "FFCF00", "FFCB00", "FFC700", "FFC300",
    "FFBF00", "FFBB00", "FFB700", "FFB300", "FFAF00", "FFAB00", "FFA700",
    "FFA300", "FF9F00", "FF9B00", "FF9700", "FF9300", "FF8F00", "FF8B00",
    "FF8700", "FF8300", "FF7F00", "FF7B00", "FF7700", "FF7300", "FF6F00",
    "FF6B00", "FF6700", "FF6300", "FF5F00", "FF5B00", "FF5700", "FF5300",
    "FF4F00", "FF4B00", "FF4700", "FF4300", "FF3F00", "FF3B00", "FF3700",
    "FF3300", "FF2F00", "FF2B00", "FF2700", "FF2300", "FF1F00", "FF1B00",
    "FF1700", "FF1300", "FF0F00", "FF0B00", "FF0700", "FF0300", "FF0000",
    "FB0000", "F70000", "F30000", "EF0000", "EB0000", "E70000", "E30000",
    "DF0000", "DB0000", "D70000", "D30000", "CF0000", "CB0000", "C70000",
    "C30000", "BF0000", "BB0000", "B70000", "B30000", "AF0000", "AB0000",
    "A70000", "A30000", "9F0000", "9B0000", "970000", "930000", "8F0000",
    "8B0000", "870000", "830000", "7F0000"]

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
range_r = np.arange(minInCm, maxInCm, resInCm)
len_r = len(range_r)
# Walabot_SetArenaTheta - input parameters
minIndegrees, maxIndegrees, resIndegrees = -30, 30, 2
range_theta = np.arange(minIndegrees, maxIndegrees, resIndegrees)
print(range_theta)
len_theta = len(range_theta)
# Walabot_SetArenaPhi - input parameters
minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees = -30, 30, 2
range_phi = (np.arange(minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees))
len_phi = len(range_phi)
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

# # Setup the 3D space
# l = 650  # in pixels
# x, y, z = np.mgrid[-l:l:100j, -l:l:100j, -l:l:100j]
#
# # PyVista UniformGrid setup
# grid = ImageData()
# grid.dimensions = x.shape
# grid.origin = (-10, -10, -10)
# grid.spacing = (20/100, 20/100, 20/100)
#
# # Setup plotter
# plotter = pv.Plotter()
# plotter.add_axes()
# plotter.add_title("Walabot 3d Imaging")
#
# # Start interactive window (non-blocking)
# plotter.show(interactive_update=True, auto_close=False)

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
    wlbt.Trigger()
    image_3d = wlbt.GetRawImage()[0]
    points = np.zeros((len_r * len_phi *len_theta, 3))
    print(points.shape)
    power = np.zeros_like(points)
    point = 0
    for idx_i, i in enumerate(range_r):
        for idx_j, j in enumerate(range_theta):
            for idx_k, k in enumerate(range_phi):
                print(point)
                # convert from spherical to cartesian
                points[point,0:3] = (i * np.sin(math.radians(j)), i * np.cos(math.radians(j)) * np.sin(math.radians(k)), i * np.cos(math.radians(j)) * np.cos(math.radians(k)))
                power[point] = image_3d[idx_j][idx_k][idx_i]
                point += 1

    point_cloud = pv.PolyData(points)
    point_cloud["elevation"] = power
    return point_cloud

if __name__ == '__main__':
    # Update in a loop
    for t in range(0, 1):
        point_cloud = generate_volume()
        point_cloud.plot(render_points_as_spheres=True)
        # grid["values"] = volume.flatten(order="F")
        #
        # # Generate updated contour data
        # updated_contour = grid.contour([0.1])  # Generate new contour
        #
        # # Proper way to update the mesh:
        # plotter.remove_actor(mesh_actor)  # Remove old mesh
        # mesh_actor = plotter.add_mesh(updated_contour, color="cyan", opacity=0.6)  # Add new mesh
        #
        # plotter.update()  # Force update the renderer
        # plotter.render()


    #plotter.close()
    EndWalabot()
