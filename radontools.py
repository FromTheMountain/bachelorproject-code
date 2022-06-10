import numpy as np
from skimage.transform import radon, iradon

random_generator = np.random.default_rng()

def add_sinogram_noise(sinogram, I_0):
    I = random_generator.poisson(I_0 * np.exp(-sinogram))
    return -np.log(I / I_0)
    
    # Alternatively, we could approximate the Poisson distribution
    # by a Gaussian distribution. This gives more or less the same
    # result.
    # sinogram = random_generator.normal(
    #     sinogram, 1/ np.sqrt((I_0 * np.exp(-sinogram))))
   
# Ramp filter. eps is not used.
def G_ramp(w, eps=0):
    return abs(w)

def add_scan_noise(img, I_0 = 1000, no_noise = False):
    """
    Add scan noise to an image, by performing three steps: Radon 
    Transform, adding scan noise, FBP. if no_noise is set to True,
    the step of adding noise is skipped. This is only for testing.
    """

    theta = np.linspace(0, 180, max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta, circle=False)

    # Add noise
    if not no_noise:
        sinogram = add_sinogram_noise(sinogram, I_0 = I_0)

    # Compute the Fourier Transform of the sinogram
    transformed_sinogram = np.fft.fft(sinogram, axis=0)

    # Apply the Ramp filter
    filtered_transformed_sinogram = transformed_sinogram.copy()

    height, width = filtered_transformed_sinogram.shape
    for i in range(width):
        filtered_transformed_sinogram[:, i] *= 2 * G_ramp(np.fft.fftfreq(sinogram.shape[0]))

    # Back to the sensor domain
    filtered_sinogram = np.fft.ifft(filtered_transformed_sinogram, axis=0)
    assert np.isclose(filtered_sinogram.imag, 0).all()
    filtered_sinogram = filtered_sinogram.real

    # Compute the reconstruction
    reconstruction = iradon(filtered_sinogram, theta=theta, 
                            filter_name=None, circle=False)

    return reconstruction
