#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    _Back.py

    The inverse Radon transform with non-iterative techniques.
"""

from __future__ import division

import numpy as np
import os
import scipy.interpolate as intp
import warnings

__all__= ["backproject", "fourier_interp", "sum"]


def backproject(sinogram, angles, filtering="ramp",
                user_interface=None):
    """
        Computes the inverse of the radon transform using filtered
        backprojection.


        Parameters
        ----------
        sinogram : ndarray, shape (A,N)
            Two-dimensional sinogram of line recordings.
        angles : (A,) ndarray
            Angular positions of the `sinogram` in radians equally
            distributed from zero to PI.
        filtering : {'ramp', 'shepp-logan', 'cosine', 'hamming', \
                     'hann'}, optional
            Specifies the Fourier filter. Either of
            
            ``ramp``
              mathematically correct reconstruction
              
            ``shepp-logan``
            
            ``cosine``
            
            ``hamming``
            
            ``hann``
            
        user_interface : instance of `ttui.ui`, optional
            The user interface to which progress should be reported.
            The default is to output nothing.


        Returns
        -------
        out : ndarray
            The reconstructed image.


        Examples
        --------
        >>> import numpy as np
        >>> from radontea import backproject
        >>> N = 5
        >>> A = 7
        >>> x = np.linspace(-N/2, N/2, N)
        >>> projection = np.exp(-x**2)
        >>> sinogram = np.tile(projection, A).reshape(A,N)
        >>> angles = np.linspace(0, np.pi, A, endpoint=False)
        >>> backproject(sinogram, angles)
        array([[ 0.03813283, -0.01972347, -0.02221885,  0.03822303,  0.01903376],
               [ 0.04033526, -0.01591647,  0.02262173,  0.04203002,  0.02123619],
               [ 0.02658797,  0.11375576,  0.41525594,  0.17170226,  0.0074889 ],
               [ 0.04008672,  0.11139209,  0.27650193,  0.16933859,  0.02098764],
               [ 0.02140445,  0.0334597 ,  0.07691547,  0.0914062 ,  0.00230537]])



        See Also
        --------
        fourier_interp : implementation by Fourier interpolation
        sum : implementation by summation in real space
    """
    ln = len(sinogram[0])
    la = len(angles)
    # transpose so we can call resize correctly
    sino = sinogram.transpose().copy()
    # Apply a Fourier filter before projecting the sinogram slices.
    # Resize image to next power of two for fourier analysis
    # Speeds up fourier and lessens artifacts
    order = max(64., 2**np.ceil(np.log(2 * ln) / np.log(2)))
    sino.resize((order,la))
    sino = sino.transpose()
    # These artifacts are for example bad contrast. To check, set:
    #order = ln
    #sino = sinogram
    kx = 2 * np.pi * np.abs(np.fft.fftfreq(int(order)))
    # Ask for the filter. Do not include zero (first element).
    if filtering == "ramp":
        pass
    elif filtering == "shepp-logan":
        kx[1:] = kx[1:] * np.sin(kx[1:]) / (kx[1:])
    elif filtering == "cosine":
        kx[1:] = kx[1:] * np.cos(kx[1:])
    elif filtering == "hamming":
        kx[1:] = kx[1:] * (0.54 + 0.46 * np.cos(kx[1:]))
    elif filtering == "hann":
        kx[1:] = kx[1:] * (1 + np.cos(kx[1:])) / 2
    elif filtering == None:
        kx[1:] = 2*np.pi
    else:
        raise ValueError("Unknown filter: %s" % filter)
    # Resize f so we can multiply it with the sinogram.
    kx = kx.reshape(1,-1)
    projection = np.fft.fft(sino, axis=-1) * kx
    sino_filtered = np.real(np.fft.ifft(projection, axis=-1))
    # Resize filtered sinogram back to original size
    sino = sino_filtered[:, :ln]
    ## Perform backprojection
    x = np.linspace(-ln/2, ln/2, ln, endpoint=False) +.5
    outarr = np.zeros((ln,ln),dtype=np.float64)
    # Meshgrid for output array
    xv, yv = np.meshgrid(x,x)
    if user_interface is not None:
        pid = user_interface.progress_new(steps=len(angles),
                           task="backprojection.{}".format(os.getpid()))
        
    for i in np.arange(len(angles)):
        #xproj = np.linspace(-n/2.0, n/2.0, N, endpoint=False)
        projinterp = intp.interp1d(x, sino[i], fill_value=0.0,
                                   copy=False, bounds_error=False)
        # Call proj_interp with cos and sin of corresponding coordinate
        # and add it to the outarr.
        phi = angles[i]
        # Smear projection onto 2d volume
        xp = xv*np.cos(phi) + yv*np.sin(phi)
        outarr += projinterp(xp)
        ## Here a much slower version with the same result:
        #for j in x:
        #    for k in x:
        #        # shift points to match origin of coordinate system
        #        xv = j - ln/2
        #        yv = k - ln/2
        #        # perform a simple coordinate transform
        #        # 
        #        xp = xv*np.cos(phi) + yv*np.sin(phi)
        #        # yp = -xv*np.sin(phi) + yv*np.cos(phi)
        #        projval = projinterp(xp)
        #        outarr[j][k] += projval
        if user_interface is not None:
            user_interface.progress_iterate(pid)
    # Normalize output (we assume that the projections are equidistant)
    # We measure angles in degrees
    dphi = np.pi/len(angles)
    # The factor of 2π comes from the choice of the unitary angular
    # frequency Fourier transform.
    outarr *= dphi / (2*np.pi)
    if user_interface is not None:
        user_interface.progress_finalize(pid)
    return outarr

    
def fourier_interp(sinogram, angles, intp_method="cubic",
                   user_interface=None):
    """
        Computes the inverse of the radon transform using Fourier
        interpolation.
        Warning: This is the naive reconstruction that assumes that
        the image is rotated through the upper left pixel nearest to
        the actual center of the image. We do not have this problem for
        odd images, only for even images.

        
        Parameters
        ----------
        sinogram : (A,N) ndarray
            Two-dimensional sinogram of line recordings.
        angles : (A,) ndarray
            Angular positions of the `sinogram` in radians equally
            distributed from zero to PI.
        intp_method : {'cubic', 'nearest', 'linear'}, optional
            Method of interpolation. For more information see
            `scipy.interpolate.griddata`. One of
            
            ``nearest``
              instead of interpolating, use the points closest to
              the input data.

            ``linear``
              bilinear interpolation between data points.
              
            ``cubic``
              interpolate using a two-dimensional poolynimial surface.
              
        user_interface : instance of `ttui.ui`, optional
            The user interface to which progress should be reported.
            The default is to output nothing. 


        Returns
        -------
        out : ndarray
            The reconstructed image.


        See Also
        --------
        backproject : implementation by backprojection
        sum : implementation by summation in real space
        scipy.interpolate.griddata : the used interpolation method
    """
    if user_interface is not None:
        pid = user_interface.progress_new(steps=1, 
                    task="Fourier_interpolation.{}".format(os.getpid()))
    if len(sinogram[0]) %2 == 0:
        warnings.warn("Fourier interpolation with slices that have"+
                      " even dimensions leads to image distortions!")
    # projections
    p_x = sinogram

    # Fourier transform of the projections
    # The sinogram data is shifted in Fourier space
    P_fx = np.fft.fft(np.fft.ifftshift(p_x, axes=-1), axis=-1)


    # This paragraph could be useful for future version if the reconstruction
    # grid is to be changed. They directly affect *P_fx*.
    # if False:
    #    # Resize everyting
    #    factor = 10
    #    P_fx = np.zeros((len(sinogram), len(sinogram[0])*factor), dtype=np.complex128)
    #    newp = np.zeros((len(sinogram), len(sinogram[0])*factor), dtype=np.complex128)
    #    for i in range(len(sinogram)):
    #        x = np.linspace(0, len(sinogram[0]), len(sinogram[0]))
    #        dint = intp.interp1d(x, sinogram[i])
    #        xn = np.linspace(0, len(sinogram[0]), len(sinogram[0])*factor)
    #        datan = dint(xn)
    #        datfft = np.fft.fft(datan)
    #        newp[i] = datan
    #        # shift datfft
    #        P_fx[i] = np.roll(1*datfft,0)
    #    p_x = newp
    #if False:
    #    # Resize the input image
    #    P_fx = np.zeros(p_x.shape, dtype=np.complex128)
    #    for i in range(len(sinogram)):
    #        factor = 2
    #        x = np.linspace(0, len(sinogram[0]), len(sinogram[0]))
    #        dint = intp.interp1d(x, sinogram[i])#, kind="nearest")
    #        xn = np.linspace(0, len(sinogram[0]), len(sinogram[0])*factor)
    #        datan = dint(xn)
    #        datfft = np.fft.fft(datan)
    #        fftint = intp.interp1d(xn, datfft)
    #        start = (len(xn) - len(x))/2
    #        end = (len(xn) + len(x))/2
    #        fidata = fftint(x)
    #        datfft = np.fft.fftshift(datfft)
    #        datfft = datfft[start:end]
    #        datfft = np.fft.ifftshift(datfft)
    #        P_fx[i] = 1*datfft

    # angles need to be normalized to 2pi
    # if angles star with 0, then the image is falsly rotated
    # unfortunately we still have ashift in the data.
    ang = ( angles.reshape(-1, 1) )
    #ang = ang + ang[1][0]

    # compute frequency coordinates fx
    fx = np.fft.fftfreq(len(p_x[0]))

    #fx = np.linspace(-np.max(fx), np.max(fx), len(p_x[0]))
    df = np.abs(fx[1]-fx[0])/2
    fx = fx.reshape(1, -1)
    # fy is zero
    
    fxl =  (fx)*np.cos(ang)
    fyl =  (fx)*np.sin(ang)
    # now fxl, fyl, and P_fx have same shape
    
    # DEBUG: save fourier transform
    #proc_arr2im(np.fft.fftshift(1.*P_fx.real), scale=True).save("./SIN_Fourier.bmp")

    # DEBUG: plot coordinates of positions of projections in fourier domain
    #from matplotlib import pylab as plt
    #plt.figure()
    #for i in range(len(fxl)):
    #    plt.plot(fxl[i],fyl[i],"x")
    #plt.axes().set_aspect('equal')
    #plt.show()


    # flatten everything for interpolation
    Xf = fxl.flatten()
    Yf = fyl.flatten()
    Zf = P_fx.flatten()
    # rintp defines the interpolation grid
    rintp = np.fft.fftshift(fx.reshape(-1))

    ## The code block yields the same result as griddata (below)
    # interpolation coordinates
    #Rf = np.zeros((len(Xf),2))
    #Rf[:,0] = 1*Xf
    #Rf[:,1] = 1*Yf
    ## reconstruction coordinates
    #Nintp = len(rintp)
    #Xn, Yn = np.meshgrid(rintp,rintp)
    #Rn = np.zeros((Nintp**2, 2))
    #Rn[:,0] = Xn.flatten()
    #Rn[:,1] = Yn.flatten()
    #
    #if intp_method.lower() == "bilinear":
    #    Fr = intp.LinearNDInterpolator(Rf,Zf.real)
    #    Fi = intp.LinearNDInterpolator(Rf,Zf.imag)
    #elif intp_method.lower() == "nearest":
    #    Fr = intp.NearestNDInterpolator(Rf,Zf.real)
    #    Fi = intp.NearestNDInterpolator(Rf,Zf.imag)
    #else:
    #    raise NotImplementedError("Unknown interpolation type: {}".format(
    #                               intp_method.lower()))
    #    return
    #Fcomp = (Fr(Rn) + 1j*Fi(Rn)).reshape(Nintp,Nintp)

    # The actual interpolation
    Fcomp = intp.griddata((Xf, Yf), Zf, (rintp[None,:], rintp[:,None]),
                          method=intp_method)

    # removed nans
    Fcomp[np.where(np.isnan(Fcomp))] = 0

    f = np.fft.fftshift( np.fft.ifft2(np.fft.ifftshift(Fcomp)) )


    # DEBUG: save absolute value of real part of fourier image
    #proc_arr2im(np.abs(Fcomp.real), scale=True).save("./Fourier_abs.bmp")
    # DEBUG: save real part of fourier image
    #proc_arr2im(Fcomp.real, scale=True).save("./Fourier.bmp")
    # DEBUG: save space domain image
    #proc_arr2im(f.real, scale=True).save("./Space.bmp")

    #q = np.arctan2(np.sum(np.abs(f.real)),np.sum(np.abs(f.imag)))
    #quality = ((q*2/np.pi-.5)*2)*100
    
    # Negative quality means too much imaginary stuff
    # 0% Quality means real and imaginary stuff are equal (not good).
    # Postive quality 100% means no imaginary stuff.
    if user_interface is not None:
        user_interface.progress_finalize(pid)
    return f
    
    
def sum(sinogram, angles, user_interface=None):
    """
        Computes the inverse of the radon transform by computing the
        integral in real space.


        Parameters
        ----------
        sinogram : (A,N) ndarray
            Two-dimensional sinogram of line recordings.
        angles : (A,) ndarray
            Angular positions of the `sinogram` in radians equally 
            distributed from zero to PI.
        user_interface : instance of `ttui.ui`, optional
            The user interface to which progress should be reported.
            The default is to output nothing.
            

        Returns
        -------
        out : ndarray
            The reconstructed image.


        See Also
        --------
        backproject : implementation by backprojection
        fourier_interp : implementation by summation in real space
    """
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
    fx = np.fft.fftfreq(sinogram[0].shape[0]) # 1D array
    kx = 2*np.pi*fx
    
    # Get the angles ϕ₀.
    phi0 = (angles).reshape(-1,1)
    # Differentials for integral
    dphi0 = len(angles)/np.pi
    dkx = np.abs(kx[1] - kx[0])

    #kx = np.fft.fftshift(kx)
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
    #      * exp( i kx (cos(ϕ₀)*x + sin(ϕ₀)*y ) )  (dependent on ϕ₀ and r)
    #
    # everything that is not dependent on phi0:
    prefactor  = 1 / ( (2*np.pi)**(3/2) )
    prefactor *= dphi0*dkx
    prefactor *= np.abs(kx)
    
    # Initiate function f
    N = len(sinogram[0])
    coords = np.linspace(-N/2.,N/2., N, endpoint=False)
    x, y = np.meshgrid(coords, coords)
    points = np.zeros((N**2, 2))
    points[:,0] = x.flatten()
    points[:,1] = y.flatten()
    f = np.zeros(len(points), dtype=np.complex128)
    lenf = len(f)
    
    # Initiate vector r that corresponds to calculating a value of f.
    r = np.zeros((2,1,1), dtype=np.complex256)
    
    # Create counters for real and imaginary parts of f.
    freal = 0
    fimag = 0

    # Compute the Fourier transform of uB.
    # This is true: np.fft.fft(UB)[0] == np.fft.fft(UB[0])
    # because axis -1 is always used.
    P = np.fft.fft(np.fft.ifftshift(sinogram, axes=-1))/np.sqrt(2*np.pi)

    if user_interface is not None:
        pid = user_interface.progress_new(steps=lenf,
                                task="summation.{}".format(os.getpid()))

    for j in xrange(lenf):
        # Get r (We compute f(r) in this for-loop)
        r[0][:] = points[j][0] #x
        r[1][:] = points[j][1] #y
        # Display how far we are
        if user_interface is not None:
            user_interface.progress_iterate(pid)

        # Integrand changes with r, so we have to create a new
        # array:
        integrand = prefactor * P
        
        ## Reminder:
        # f(r) = 1 / ( (2π)^(3/2) )                  (prefactor)
        #      * iint dϕ₀ dkx                           (prefactor)
        #      * |kx|                                   (prefactor)
        #      * P_ϕ₀(kx)                               (dependent on ϕ₀)
        #      * exp( i kx (cos(ϕ₀)*x + sin(ϕ₀)*y ) )  (dependent on ϕ₀ and r)
        #
        # everything that is not dependent on phi0:
        integrand *= np.exp(1j*kx*(
                                      r[0] * np.cos(phi0) +
                                      r[1] * np.sin(phi0)   )
                                   )
        
        # Calculate the integral for the position r
        integrand.sort()
        f[j] = np.sum(integrand)
        
        # For quality control: Add up values for imaginary and
        # real values of f.
        freal += np.abs(np.real(f[j]))
        fimag += np.abs(np.imag(f[j]))

    q = np.arctan2(freal,fimag)
    quality = ((q*2/np.pi-.5)*2)*100
    # Negative quality means too much imaginary stuff
    # 0% Quality means real and imaginary stuff are equal (not good).
    # Postive quality 100% means no imaginary stuff.
    if user_interface is not None:
        user_interface.progress_finalize(pid)
        
    return f.real.reshape((N,N))

