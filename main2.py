import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import imageio.v3 as imageio
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D

from functions2 import load_planet_image, generate_phase_screen, generate_multi_layer_turbulence, propagate_wavefront, simulate_lgs, shack_hartmann_sensor, reconstruct_wavefront, correct_wavefront,apply_actuators, astigmatism_creator, spherical_creator, calculate_wavefront_error, optimize_actuators, resize_grid_bilinear

# Parameters
num_layers = 3
r0 = 0.5
N = 256
L = 10
altitude = 3
num_lenslet = 32
lenslet_pitch = N // num_lenslet
scale_factor = 0.05
num_actuators = 20
dm_pitch = float(N // num_actuators)


# Generate the phase screens
phase_screens = generate_multi_layer_turbulence(num_layers, r0, N, L, scale_factor)

# Propagate through the generated phase screens
initial_wavefront = np.zeros((N, N))

# Corrected propagation for plane wavefront
distorted_wavefront = propagate_wavefront(phase_screens, initial_wavefront)

# Corrected propagation for LGS wavefront
lgs_wavefront = simulate_lgs(N, L, altitude)
distorted_lgs_wavefront = propagate_wavefront(phase_screens, lgs_wavefront)

# Set figure saving folder
folder = "figures/"

# Load and propagate planet wavefront
planet_image_path = "jupiter.png"
planet_wavefront = load_planet_image(planet_image_path, N)
distorted_planet_wavefront = propagate_wavefront(phase_screens, planet_wavefront)

# Shack-Hartmann wavefront sensing
slope_x, slope_y = shack_hartmann_sensor(distorted_wavefront, num_lenslet, lenslet_pitch)

# Reconstruct wavefront from slopes
reconstructed_wavefront = reconstruct_wavefront(slope_x, slope_y, N, lenslet_pitch)
scaling_factor = np.max(np.abs(distorted_wavefront)) / np.max(np.abs(reconstructed_wavefront))
reconstructed_wavefront *= scaling_factor

# Correct distorted planet wavefront
corrected_wavefront = correct_wavefront(distorted_planet_wavefront, reconstructed_wavefront)
   
# Find global min and max for the color scale
all_data = [
    *phase_screens, distorted_wavefront, lgs_wavefront,
    distorted_lgs_wavefront, planet_wavefront,
    distorted_planet_wavefront, reconstructed_wavefront, corrected_wavefront
]
global_min = min(np.min(data) for data in all_data)
global_max = max(np.max(data) for data in all_data)

# Plot phase screens
fig, axes = plt.subplots(1, num_layers, figsize=(15, 3))
for i, screen in enumerate(phase_screens):
    ax = axes[i]
    im = ax.imshow(screen, cmap='jet', extent=[0, L, 0, L], vmin=global_min, vmax=global_max)
    ax.set_title(f'Layer {i+1}')
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f"{folder}Phase_screen.png")

# Plot distorted wavefront
plt.figure()
plt.imshow(distorted_wavefront, cmap='jet', extent=[0, L, 0, L], vmin=global_min, vmax=global_max)
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
plt.title('Distorted Plane Wavefront')
plt.colorbar(label='Phase (rad)')
plt.savefig(f"{folder}Distorted_wavefront.png")

# Plot LGS wavefront
plt.figure()
plt.imshow(lgs_wavefront, cmap='jet', extent=[0, L, 0, L], vmin=global_min, vmax=global_max)
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
plt.title('LGS Wavefront')
plt.colorbar(label='Phase (rad)')
plt.savefig(f"{folder}LGS_wavefront.png")

# Plot distorted LGS wavefront
plt.figure()
plt.imshow(distorted_lgs_wavefront, cmap='jet', extent=[0, L, 0, L], vmin=global_min, vmax=global_max)
plt.title('Distorted LGS Wavefront')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
plt.colorbar(label='Phase (rad)')
plt.savefig(f"{folder}Distorted_LGS_wavefront.png")

# Plot planet wavefront
plt.figure()
plt.imshow(planet_wavefront, cmap='jet', extent=[0, L, 0, L], vmin=global_min, vmax=global_max)
plt.title('Jupiter Wavefront')
plt.colorbar(label='Phase (rad)')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
plt.savefig(f"{folder}planet_wavefront.png")

# Plot distorted planet wavefront
plt.figure()
plt.imshow(distorted_planet_wavefront, cmap='jet', extent=[0, L, 0, L], vmin=global_min, vmax=global_max)
plt.title('Distorted Jupiter Wavefront')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
plt.colorbar(label='Phase (rad)')
plt.savefig(f"{folder}Distorted_planet_wavefront.png")

# Plot reconstructed wavefront
plt.figure()
plt.imshow(reconstructed_wavefront, cmap='jet', extent=[0, L, 0, L], vmin=global_min, vmax=global_max)
plt.title('Reconstructed Wavefront')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
plt.colorbar(label='Phase (rad)')
plt.savefig(f"{folder}reconstructed_distortion.png")

# Plot corrected planet wavefront
plt.figure()
plt.imshow(corrected_wavefront, cmap='jet', extent=[0, L, 0, L], vmin=global_min, vmax=global_max)
plt.title('Corrected Jupiter Wavefront')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
plt.colorbar(label='Phase (rad)')
plt.savefig(f"{folder}corrected_planet_wavefront.png")




# ---------------------- Extra Astigmatismo -------------------------------------------
astigmatism = astigmatism_creator(2, 8, N)
esfera = spherical_creator(4, N)
distortion_small = resize_grid_bilinear(astigmatism - esfera, 20)
plt.figure()
plt.imshow(distortion_small, cmap='jet', extent=[0, L, 0, L])
plt.title('Corrected Jupiter Wavefront')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
plt.colorbar(label='Phase (rad)')
plt.savefig(f"{folder}spherical_wavefront.png")

# ------------------------ Deformable Mirror Júpiter -----------------------
reconstructed_wavefront_small = resize_grid_bilinear(reconstructed_wavefront, 50)
actuator_phase_grid = np.zeros((num_actuators, num_actuators))

optimized_actuators = optimize_actuators(reconstructed_wavefront_small, actuator_phase_grid) 


# Plot actuator deformations
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(num_actuators), range(num_actuators))
ax.plot_surface(X, Y, optimized_actuators, cmap='viridis')
ax.set_xlabel('Actuator X Position (arb.)')
ax.set_ylabel('Actuator Y Position (arb.)')
ax.set_zlabel('Actuator Deformation (rad)')
ax.set_title('Deformable Mirror Actuator Deformations')
plt.savefig(f"{folder}actuator_deformations.png")


