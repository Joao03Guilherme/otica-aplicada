import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import imageio.v3 as imageio
from scipy.ndimage import zoom
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr


def load_planet_image(file_path, N):
    """
    Load a PNG image of a planet and convert it to a mesh grid.
    file_path: Path to the PNG image file
    N: Size of the mesh grid (NxN)
    """
    # Load the PNG image using imageio v3
    planet_image = imageio.imread(file_path)
    # Convert the image to grayscale
    if planet_image.ndim == 3:  # RGB image
        planet_image = np.mean(planet_image, axis=2)
    # Normalize the image intensity to [0, 1]
    planet_image = planet_image.astype(float) / 255.0
    # Resize the image to match the grid size
    planet_image_resized = zoom(planet_image, (N / planet_image.shape[0], N / planet_image.shape[1]))
    # Convert intensity to wavefront phase
    phase = (planet_image_resized - 0.5) * 0.6
    return phase

def generate_phase_screen(r0, N, L, scale_factor=1.0):
    """
    Generate a phase screen based on the Kolmogorov turbulence model.
    r0: Fried parameter (in meters)
    N: Size of the phase screen (NxN)
    L: Physical size of the screen (in meters)
    scale_factor: Scaling factor to reduce the intensity of phase distortions
    """
    start_time = time.time()

    # Spatial frequency grid
    delta_f = 1.0 / L
    fx = np.fft.fftfreq(N, delta_f)
    fy = np.fft.fftfreq(N, delta_f)
    fx, fy = np.meshgrid(fx, fy)
    f_squared = fx**2 + fy**2
    f_squared[0, 0] = 1  # Avoid division by zero at the origin

    # Kolmogorov PSD
    PSD_phi = 0.023 * (r0**(-5/3)) * (f_squared**(-11/6))
    PSD_phi[0, 0] = 0  # Set the DC component to zero

    # Generate random phase screen in frequency domain
    random_phase = (np.random.randn(N, N) + 1j * np.random.randn(N, N))
    phase_screen = np.real(ifft2(fft2(random_phase) * np.sqrt(PSD_phi)))

    end_time = time.time()
    print(f"Phase screen generated in {end_time - start_time:.4f} seconds")

    return phase_screen * scale_factor

def generate_multi_layer_turbulence(num_layers, r0, N, L, scale_factor=1.0):
    """
    Generate multiple phase screens to simulate multi-layer atmospheric turbulence.
    num_layers: Number of atmospheric layers
    r0: Fried parameter (in meters)
    N: Size of each phase screen (NxN)
    L: Physical size of each screen (in meters)
    scale_factor: Scaling factor to reduce the intensity of phase distortions
    """
    phase_screens = []
    for i in range(num_layers):
        print(f"Generating phase screen {i+1}/{num_layers}")
        screen = generate_phase_screen(r0, N, L, scale_factor)
        phase_screens.append(screen)
    return phase_screens


def propagate_wavefront(phase_screens, initial_wavefront):
    """
    Propagate an initial wavefront through multiple phase screens.
    phase_screens: List of phase screens (each NxN)
    initial_wavefront: Initial wavefront phase (NxN)
    """
    wavefront = np.copy(initial_wavefront)
    for screen in phase_screens:
        wavefront += screen  # Accumulate phase distortions
    return wavefront

def simulate_lgs(N, L, altitude):
    """
    Simulate the Laser Guide Star (LGS) as a point source at the specified altitude.
    N: Size of the wavefront grid (NxN)
    L: Physical size of the wavefront grid (in meters)
    altitude: Altitude of the LGS (in meters)
    """
    # Create a grid for the wavefront
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    xx, yy = np.meshgrid(x, y)

    # Simulate the LGS as a point source at the specified altitude
    lgs_wavefront = 0.2*np.exp(-(xx**2 + yy**2) / (2 * (altitude / 2.355)**2))

    return lgs_wavefront

def shack_hartmann_sensor(wavefront, num_lenslets, lenslet_pitch):
    """
    Simulate a Shack-Hartmann sensor measuring wavefront slopes.
    wavefront: Wavefront phase (NxN)
    num_lenslets: Number of lenslets per dimension
    lenslet_pitch: Pitch of the lenslet array
    """
    N = wavefront.shape[0]
    slope_x = np.zeros((num_lenslets, num_lenslets))
    slope_y = np.zeros((num_lenslets, num_lenslets))

    # Loop over each lenslet
    for i in range(num_lenslets):
        for j in range(num_lenslets):
            # Extract the sub-aperture (lenslet) region
            sub_aperture = wavefront[i*lenslet_pitch:(i+1)*lenslet_pitch, j*lenslet_pitch:(j+1)*lenslet_pitch]

            # Compute the slopes using central difference method
            slope_x[i, j] = (sub_aperture[:, -1] - sub_aperture[:, 0]).mean() / lenslet_pitch
            slope_y[i, j] = (sub_aperture[-1, :] - sub_aperture[0, :]).mean() / lenslet_pitch

    return slope_x, slope_y

def reconstruct_wavefront(slope_x, slope_y, N, lenslet_pitch):
    """
    Reconstruct the wavefront phase from the Shack-Hartmann sensor slopes using a least-squares method.
    slope_x: Slopes in the x-direction (num_lenslets_x, num_lenslets_y)
    slope_y: Slopes in the y-direction (num_lenslets_x, num_lenslets_y)
    N: Size of the phase screen (NxN)
    lenslet_pitch: Pitch of the lenslet array
    """
    num_lenslets_x, num_lenslets_y = slope_x.shape
    
    # Resize slopes to match the full resolution of the wavefront
    slope_x_resized = zoom(slope_x, (N / num_lenslets_x, N / num_lenslets_y))
    slope_y_resized = zoom(slope_y, (N / num_lenslets_x, N / num_lenslets_y))

    # Setup the least-squares problem
    num_points = N * N
    A = lil_matrix((2 * num_points, num_points))
    b = np.zeros(2 * num_points)

    index = lambda i, j: i * N + j

    for i in range(N):
        for j in range(N):
            idx = index(i, j)
            if i < N - 1:
                A[idx, idx] = -1
                A[idx, index(i + 1, j)] = 1
                b[idx] = slope_y_resized[i, j] * lenslet_pitch
            if j < N - 1:
                A[num_points + idx, idx] = -1
                A[num_points + idx, index(i, j + 1)] = 1
                b[num_points + idx] = slope_x_resized[i, j] * lenslet_pitch

    # Solve the least-squares problem
    x = lsqr(A, b)[0]
    phase = x.reshape((N, N))

    return phase

def correct_wavefront(distorted_wavefront, phase_distortion):
    
    return distorted_wavefront - phase_distortion


'''
def influence_function(x, y, x0, y0, sigma):
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))


def simulate_deformable_mirror(phase_error, num_actuators, dm_pitch):
    """
    Simulate the deformable mirror correction.
    
    phase_error: The wavefront error to be corrected (NxN array)
    num_actuators: Number of actuators along one dimension (M)
    dm_pitch: Distance between actuators
    """
    N = phase_error.shape[0]
    M = num_actuators
    actuator_positions = np.linspace(0, N, M, endpoint=False) + dm_pitch / 2
    X, Y = np.meshgrid(actuator_positions, actuator_positions)

    # Initialize actuator deformations
    actuator_deformations = np.zeros((M, M))

    # Influence function parameters
    sigma = dm_pitch / 2
    x_grid, y_grid = np.meshgrid(np.arange(N), np.arange(N))

    # Calculate the required actuator deformations
    for i in range(M):
        for j in range(M):
            x0, y0 = X[i, j], Y[i, j]
            influence = influence_function(x_grid, y_grid, x0, y0, sigma)
            actuator_deformations[i, j] = np.sum(phase_error * influence)

    return actuator_deformations
'''

def astigmatism_creator(f_x, f_y, N):
    """
    Generates astigmatism wavefront.
    f_x: focal distance in x direction
    f_y: focal distance in y direction
    N: Size of the grid (NxN)
    """
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    Z = (X**2 / f_x**2 + Y**2 / f_y**2)
    return Z

def spherical_creator(f, N):
    """
    Generates spherical wavefront.
    f: focal distance
    
    N: Size of the grid (NxN)
    """
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    Z = (X**2 / f**2 + Y**2 / f**2)
    return Z

def apply_actuators(wavefront, actuator_phase_grid):
    """Applies actuators with Gaussian influence to the wavefront."""
    size = wavefront.shape[0]
    min_phase = np.min(wavefront)
    max_actuator = np.max(actuator_phase_grid)
    new_wavefront = np.copy(wavefront)
    diff = min_phase - max_actuator

    num_actuators = actuator_phase_grid.shape[0]
    spacing = size // (num_actuators-1)
    if min_phase <0:
        new_wavefront = new_wavefront + min_phase
        diff = 0
    actuated_wavefront = np.zeros((size, size))
    
    # Apply Gaussian influence from each actuator
    for i in range(size):
        for j in range(size):
             # Calculate actuator positions
            x0 = min(int(i / spacing), num_actuators - 1)
            y0 = min(int(j / spacing), num_actuators - 1)
                      
            phase = diff - (new_wavefront[i, j] - actuator_phase_grid[x0, y0])
            actuated_wavefront[i, j] = phase
            

    return actuated_wavefront


def calculate_wavefront_error(wavefront):
    """Calculates the error of the wavefront from being flat."""
    # Error is defined as the standard deviation of the wavefront
    return np.std(wavefront)

def optimize_actuators(wavefront, actuator_phase_grid, learning_rate=0.01, tolerance=1e-6, max_iterations=1500):
    """Optimizes the actuator positions to minimize the wavefront error."""
    num_actuators = actuator_phase_grid.shape[0]
    
    for iteration in range(max_iterations):
        actuated_wavefront = apply_actuators(wavefront, actuator_phase_grid)
        error = calculate_wavefront_error(actuated_wavefront)
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Error: {error}")
        
        if error < tolerance:
            break
        
        # Gradient descent step: Update actuators to minimize error
        gradient = np.zeros_like(actuator_phase_grid)
        
        for i in range(num_actuators):
            for j in range(num_actuators):
                original_value = actuator_phase_grid[i, j]
                
                # Perturb the actuator phase slightly
                actuator_phase_grid[i, j] += learning_rate
                perturbed_wavefront = apply_actuators(wavefront, actuator_phase_grid)
                perturbed_error = calculate_wavefront_error(perturbed_wavefront)
                
                # Compute gradient
                gradient[i, j] = (perturbed_error - error) / learning_rate
                
                # Restore the original value
                actuator_phase_grid[i, j] = original_value
        
        # Update actuator phases
        actuator_phase_grid -= learning_rate * gradient
    
    return actuator_phase_grid

def resize_grid_bilinear(grid, new_size):
    """
    Resizes the grid using bilinear interpolation.
    
    Parameters:
    grid (np.ndarray): The original grid to be resized.
    new_size (int): The size of the new grid (new_size x new_size).
    
    Returns:
    np.ndarray: The resized grid.
    """
    zoom_factor = new_size / grid.shape[0]
    resized_grid = zoom(grid, zoom_factor, order=1)  # order=1 for bilinear interpolation
    
    return resized_grid
