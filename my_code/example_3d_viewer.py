import pyvista as pv
import numpy as np
import time
from pyvista import ImageData

# Setup the 3D space
x, y, z = np.mgrid[-10:10:100j, -10:10:100j, -10:10:100j]

# Initial volume
def generate_volume(t):
    #multiplier = t * np.random.rand() * 1000
    multiplier = t
    vol = np.sin(x**2 + y**2 + z**2 + multiplier) / (x**2 + y**2 + z**2 + 1e-5)
    print(f"Volume: {vol}")
    return vol

# PyVista UniformGrid setup
grid = ImageData()
grid.dimensions = x.shape
grid.origin = (-10, -10, -10)
grid.spacing = (20/100, 20/100, 20/100)

# Setup plotter
plotter = pv.Plotter()
plotter.add_axes()
plotter.add_title("Live 3D Updating Volume")

# Add initial mesh
volume = generate_volume(0)
grid.point_data["values"] = volume.flatten(order="F")
contour = grid.contour([0.1])
mesh_actor = plotter.add_mesh(contour, color="cyan", opacity=0.6)

# Start interactive window (non-blocking)
plotter.show(interactive_update=True, auto_close=False)

# Update in a loop
for t in range(100):
    time.sleep(0.1)
    volume = generate_volume(t)
    grid["values"] = volume.flatten(order="F")

    # Generate updated contour data
    updated_contour = grid.contour([0.1])  # Generate new contour

    # Proper way to update the mesh:
    plotter.remove_actor(mesh_actor)  # Remove old mesh
    mesh_actor = plotter.add_mesh(updated_contour, color="cyan", opacity=0.6)  # Add new mesh

    plotter.update()  # Force update the renderer
    plotter.render()

plotter.close()
