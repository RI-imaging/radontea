import numpy as np


def integrate(sinogram: np.ndarray, angles: np.ndarray, count=None,
              max_count=None) -> np.ndarray:
    """2D sum-reconstruction with the Fourier slice theorem

    Computes the inverse of the Radon transform by computing the
    integral in real space.


    Parameters
    ----------
    sinogram: (A,N) ndarray
        Two-dimensional sinogram of line recordings.
    angles: (A,) ndarray
        Angular positions of the `sinogram` in radians equally
        distributed from zero to PI.
    count, max_count: multiprocessing.Value or `None`
        Can be used to monitor the progress of the algorithm.
        Initially, the value of `max_count.value` is incremented
        by the total number of steps. At each step, the value
        of `count.value` is incremented.


    Returns
    -------
    out: ndarray
        The reconstructed image.
    """
    if max_count is not None:
        with max_count.get_lock():
            max_count.value += sinogram.shape[1]**2 + 1
    # In the script we used the unitary angular frequency (uaf) Fourier
    # Transform. The discrete Fourier transform is equivalent to the
    # unitary ordinary frequency (uof) Fourier transform.
    #
    # uof: f₁(ξ) = int f(x) exp(-2πi xξ)
    #
    # uaf: f₃(ω) = (2π)^(-n/2) int f(x) exp(-i ωx)
    #
    # f₁(ξ) = (2π)^(n/2) f₃(ω)
    # ω = 2πξ
    #
    # We have a one-dimensional (n=1) Fourier transform and UB in the
    # script is equivalent to f₃(ω). Because we are working with the
    # uaf, we divide by sqrt(2π) after computing the fft with the uof.

    # Corresponding sample frequencies
    fx = np.fft.fftfreq(sinogram[0].shape[0])  # 1D array
    kx = 2 * np.pi * fx

    # Get the angles ϕ₀.
    phi0 = (angles).reshape(-1, 1)
    # Differentials for integral
    dphi0 = len(angles) / np.pi
    dkx = np.abs(kx[1] - kx[0])

    # We will later multiply with phi0.
    # Make sure we are using correct shapes
    kx = kx.reshape(1, -1)

    # Create the integrand
    # Integrals over ϕ₀ [0,2π]; kx [-k,k]
    #   - We do not have double coverage, as the input data is only
    #     from 0 to 180 degrees.
    #   - unitary angular frequency to unitary ordinary frequency
    #     conversion performed in calculation of P=FT(sinogram).
    #
    # f(r) = 1 / ( (2π)^(3/2) )                     (prefactor)
    #      * iint dϕ₀ dkx                           (prefactor)
    #      * |kx|                                   (prefactor)
    #      * P_ϕ₀(kx)                               (dependent on ϕ₀)
    #      * exp( i kx (cos(ϕ₀)*x + sin(ϕ₀)*y ) )   (dependent on ϕ₀ and r)
    #
    # everything that is not dependent on phi0:
    prefactor = 1 / ((2 * np.pi)**(3 / 2))
    prefactor *= dphi0 * dkx
    prefactor *= np.abs(kx)

    # Initiate function f
    N = len(sinogram[0])
    coords = np.linspace(-N / 2., N / 2., N, endpoint=False)
    x, y = np.meshgrid(coords, coords)
    points = np.zeros((N**2, 2))
    points[:, 0] = x.flatten()
    points[:, 1] = y.flatten()
    f = np.zeros(len(points), dtype=np.complex128)
    lenf = len(f)

    # Initiate vector r that corresponds to calculating a value of f.
    r = np.zeros((2, 1, 1), dtype=np.complex128)

    # Compute the Fourier transform of uB.
    # This is true: np.fft.fft(UB)[0] == np.fft.fft(UB[0])
    # because axis -1 is always used.
    P = np.fft.fft(np.fft.ifftshift(sinogram, axes=-1)) / np.sqrt(2 * np.pi)

    if count is not None:
        with count.get_lock():
            count.value += 1

    for j in range(lenf):
        # Get r (We compute f(r) in this for-loop)
        r[0][:] = points[j][0]  # x
        r[1][:] = points[j][1]  # y

        # Integrand changes with r, so we have to create a new
        # array:
        integrand = prefactor * P

        # Reminder:
        # f(r) = 1 / ( (2π)^(3/2) )                    (prefactor)
        #      * iint dϕ₀ dkx                          (prefactor)
        #      * |kx|                                  (prefactor)
        #      * P_ϕ₀(kx)                              (dependent on ϕ₀)
        #      * exp( i kx (cos(ϕ₀)*x + sin(ϕ₀)*y ) )  (dependent on ϕ₀ and r)
        #
        # everything that is not dependent on phi0:
        integrand *= np.exp(1j * kx * (
            r[0] * np.cos(phi0) +
            r[1] * np.sin(phi0))
        )

        # Calculate the integral for the position r
        integrand.sort()
        f[j] = np.sum(integrand)

        # Display how far we are
        if count is not None:
            with count.get_lock():
                count.value += 1

    return f.reshape((N, N))
