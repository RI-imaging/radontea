#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    _Back_iterative.py

    Investigation of non-diffraction tomography methods.

    - perform algebraic reconstruction of a sinogram taken from angles
      that are not equidistant

    - Algebraic reconstruction technique (ART)

    - Simultaneous algebraic reconstruction technique (SART)
      - no hamming window
      - not parallelized

"""
from __future__ import division, print_function

import numpy as np
import os


__all__ = ["art", "sart"]


def art(sinogram, angles, initial=None, iterations=1,
        user_interface=None):
    """ 
        The Algebraic Reconstruction Technique (ART) iteratively
        computes the inverse of the Radon transform in two dimensions.
        The reconstruction technique uses *rays* of the diameter of
        one pixel to iteratively solve the system of linear equations
        that describe the projection process. The binary weighting
        factors are
            - 1, if the center of the a pixel is within the *ray*
            - 0, else


        Parameters
        ----------
        sinogram : ndarrayy, shape (A,N)
            Two-dimensional sinogram of line recordings.
        angles : ndarray, length A
            Angular positions of the `sinogram` in radians. The angles
            at which the sinogram slices were recorded do not have to be
            distributed equidistantly as in :func:`backproject`.
            The angles are internaly converted to modulo PI.
        initial : ndarray, shape (N,N), optional
            The initial guess for the solution.
        iterations : int
            Number of iterations to perform.
        user_interface : instance of `ttui.ui`, optional
            The user interface to which progress should be reported.
            The default is to output nothing.


        See Also
        --------
        sart : simultaneous algebraic reconstruction technique


        Notes
        -----
        For theoretical backround, see
        Kak, A. C., & Slaney, M.. *Principles of Computerized
        Tomographic Imaging*, SIAM, (2001)
        
        Sec. 7.2:
        *"ART reconstrutions usually suffer from salt and pepper noise,
        which is caused by the inconsitencies introuced in the set of
        equations by the approximations commonly used for* 
        :math:`w_{ik}` *'s."*
    """
    # make sure `iterations` is an integer
    iterations = int(iterations)
    N = len(sinogram[0])
    # Meshgrid for weigths
    center = N/2.0
    x = np.arange(N) - center +.5
    X, Y = np.meshgrid(x,x)
    # initiate array
    if initial is None:
        f = np.zeros((N,N),dtype=np.dtype(float))
    else:
        f = 1*initial.transpose()[::-1]
    # Make sure all angles are in [0,PI)
    for i in np.arange(len(angles)):
        if angles[i] > np.pi:
            offset = np.floor(angles[i]/np.pi)
            angles[i] -= offset*np.pi
        elif angles[i] < 0:
            offset = np.floor(np.abs((angles[i]/np.pi)))+1
            angles[i] += offset*np.pi
        if angles[i] == np.pi:
            angles[i] = 0

    # These lambda functions return the two x- and y- values of two
    # points projected onto a line along the x- and y-axis, having the
    # angle angle_k in radians.
    # Sourcing this out to here makes things a little faster.
    def GetLambdaLines(angle_k):
        if angle_k == 0:
            # Divide by zero error for tan
            # We have horizontal lines (parallel to x)
            line1 = lambda x2,y2,xpi1,ypi1,xpi2,ypi2,angle_k: (x2-.5, ypi1*np.ones(y2.shape))
            line2 = lambda x2,y2,xpi1,ypi1,xpi2,ypi2,angle_k: (x2+.5, ypi2*np.ones(y2.shape))
        elif angle_k == np.pi/2:
            # We have vertical lines (parallel to y)
            line1 = lambda x2,y2,xpi1,ypi1,xpi2,ypi2,angle_k: (xpi1*np.ones(x2.shape), y2+.5)
            line2 = lambda x2,y2,xpi1,ypi1,xpi2,ypi2,angle_k: (xpi2*np.ones(x2.shape), y2-.5)
        elif angle_k < np.pi/2:
            # CASE 1
            # Compute any other positions on the lines from the given things
            line1 = lambda x2,y2,xpi1,ypi1,xpi2,ypi2,angle_k: ( (y2-ypi1)/np.tan(angle_k) + xpi1,
                                    (x2-xpi1)*np.tan(angle_k) + ypi1)
                #def line2(x2,y2):
                #    y = (x2-xpi2)/np.tan(angle_k) - ypi2
                #    x = (y2-ypi2)*np.tan(angle_k) - xpi2
                #    return x,y
            line2 = lambda x2,y2,xpi1,ypi1,xpi2,ypi2,angle_k: ( (y2-ypi2)/np.tan(angle_k) + xpi2,
                                    (x2-xpi2)*np.tan(angle_k) + ypi2)
        else:
            # CASE 2: Switch x-output - only for speed. Not very obvious - possibly a hack.
            # Compute any other positions on the lines from the given things
            line1 = lambda x2,y2,xpi1,ypi1,xpi2,ypi2,angle_k: ( (y2-ypi2)/np.tan(angle_k) + xpi2,
                                    (x2-xpi1)*np.tan(angle_k) + ypi1)
                #def line2(x2,y2):
                #    y = (x2-xpi2)/np.tan(angle_k) - ypi2
                #    x = (y2-ypi2)*np.tan(angle_k) - xpi2
                #    return x,y
            line2 = lambda x2,y2,xpi1,ypi1,xpi2,ypi2,angle_k: ( (y2-ypi1)/np.tan(angle_k) + xpi1,
                                    (x2-xpi2)*np.tan(angle_k) + ypi2)
        return line1, line2
    # Sort angles?
    # This could increase the speed of convergence.    

    # f[j] is consistent with Kak, Slaney
    f = f.flatten()

    if user_interface is not None:
        pid = user_interface.progress_new(steps=iterations*len(angles),
                                         task="ART.{}".format(os.getpid()))

    for iteration in np.arange(iterations):
        #
        # i iterates the rays
        # j iterates the flattened outarr
        # k iterates the angles
        #
        for k in np.arange(len(angles)):
            # From now on we work in radians
            angle_k = angles[k]
            # p[i] is consistent with Kak, Slaney
            p = sinogram[k]
               # # We do not store the binary weights for each ray:
               # # For large images this thing could get big:
               # w = np.zeros((len(p),N*N), dtype=np.bool)
               # # w[i][j]
            line1, line2 = GetLambdaLines(angle_k)
            # CASES for all 2 quadrants. Angles are modulo PI
            # This case stuff is dependent on the angle. We enumerate
            # pr in negative mathematical angular direction
            #
            # This is the position on the projection, centered around 0:
            pr = np.arange(len(p)) - center + .5 # radial distance,
            #
            if angle_k <= np.pi/2:
                # 0 to PI/2
                case = 1
                # position of each p[i] in the centered *outarr*.
                x_p1 = (pr-.5)*np.sin(angle_k)                # a position of line1 in X
                y_p1 = -(pr-.5)*np.cos(angle_k)                # a position of line1 in Y
                x_p2 = (pr+.5)*np.sin(angle_k)                # a position of line2 in X
                y_p2 = -(pr+.5)*np.cos(angle_k)                # a position of line2 in Y
            else:
                # PI/2 to PI
                # position of each p[i] in the centered *outarr*.
                x_p1 = (pr+.5)*np.sin(angle_k)                # a position of line1 in X
                y_p1 = -(pr+.5)*np.cos(angle_k)                # a position of line1 in Y
                x_p2 = (pr-.5)*np.sin(angle_k)                # a position of line2 in X
                y_p2 = -(pr-.5)*np.cos(angle_k)                # a position of line2 in Y
                case = 2
            for i in np.arange(len(p)):
                # If the angle is zero, then we are looking at the
                # projections onto the right side. The indices are
                # enumerated from bottom to top.
                w_i = np.zeros((N,N), dtype=np.bool)
                       # # where does the ray cut w_i?
                       # xpi1 = x_p1[i]
                       # ypi1 = y_p1[i]
                       # xpi2 = x_p2[i]
                       # ypi2 = y_p2[i]
                # Check if a value is between the two lines (within ray)
                # For each value of X and Y, compute what the values of the
                # line would be.
                xl1, yl1 = line1(X,Y,x_p1[i],y_p1[i],x_p2[i],y_p2[i],angle_k)
                xl2, yl2 = line2(X,Y,x_p1[i],y_p1[i],x_p2[i],y_p2[i],angle_k)
                #
                AND = np.logical_and
                # Values lie between the two lines if the following is True:
                # Furthermore we restrict ourselves to a disk.
                w_i = AND(X**2 + Y**2 < center**2,
                          AND(AND(xl1 < X ,X < xl2),
                              AND(yl1 > Y ,Y > yl2))).flatten()
                #
                # i iterates the rays
                # j iterates the flattened outarr
                # k iterates the angles
                ## In each iteration of the angles the image is changed
                #if np.sum(w_i) != 0:
                #    f += ( p[i] - np.sum(f*w_i) )/np.sum(w_i) * w_i
                # This is faster because we don't a the zeros.
                f[np.where(w_i==True)] += np.divide(
                        ( p[i] - np.sum(f[np.where(w_i==True)]) ),
                        np.sum(w_i))
                del w_i
                #
                 # # Export single steps as bmp files...
                 # #if i%10 == 1:
                 # #    image = f.reshape((N,N))[::-1]
            if user_interface is not None:
                user_interface.progress_iterate(pid)

                 # #    proc_arr2im(image, cut=True).save(os.path.join(DIR,"test/Elephantulus_small_art_%02d_%04d_%08d.bmp" % (iteration,k,i)))
    # By slicing in-place [::-1] we get rid of the inversion of the
    # image along the y-axis.
    if user_interface is not None:
        user_interface.progress_finalize(pid)
    
    return f.reshape((N,N))[::-1].transpose()


def sart(sinogram, angles, initial=None, iterations=1,
         user_interface=None):
    """ The simultaneous algebraic reconstruction technique (SART)
        computes an inverse of the Radon transform in two dimensions.
        The reconstruction technique uses "rays" of the diameter of
        one pixel to iteratively solve the system of linear equations
        that describe the image. The weighting factors are bilinear
        elements. At the beginning and end of each ray, only partial
        weights are used. The pixel values of the image are updated
        only after each iteration is complete.
        
        
        Parameters
        ----------
        sinogram : ndarrayy, shape (A,N)
            Two-dimensional sinogram of line recordings.
        angles : ndarray, length A
            Angular positions of the `sinogram` in radians. The angles
            at which the sinogram slices were recorded do not have to be
            distributed equidistantly as in backprojection techniques.
            The angles are internaly converted to modulo PI.
        initial : ndarray, shape (N,N), optional
            the initial guess for the solution.
        iterations : integer
            Number of iterations to perform.
        user_interface : instance of `ttui.ui`, optional
            The user interface to which progress should be reported.
            The default is to output nothing.
            
        
        See Also
        --------
        art : algebraic reconstruction technique


        Notes
        -----
        Algebraic reconstruction technique (ART) (see `art`):
            Iterations are performed over each ray of each projection.
            Weighting factors are binary (1 if center of pixel is
            within ray, 0 else). This leads to salt and pepper noise.
        
        Simultaneous iterative reconstruction technique (SIRT):
            Same idea as ART, but for each iteration, the change of the
            image f is computed for all rays and projections separately
            and the weights are applied simultaneously after each
            iteration. The result is a slower convergence but the final
            image is also less noisy.
        
        This implementation does NOT use a hamming window to filter
        the data and to emphasize points at the center of the recon-
        struction region.
        
        For theoretical backround, see
        Kak, A. C., & Slaney, M.. *Principles of Computerized
        Tomographic Imaging*, SIAM, (2001)
        
        Sec 7.4:
        *"[SART] seems to combine the best of ART and SIRT. [...] Here
        are the main features if SART: First, [...] the traditional
        pixel basis is abandonded in favor of bilinear elements
        [e.g. interpolation]. Also, for a circular reconstruction
        region, only partial weights are assigned to the first and last
        picture elements on the individual rays. To further reduce the
        noise [...], the correction terms are simultaneously applied
        for all the rays in one projection [...]."*
    """
    N = len(sinogram[0])
    # Meshgrid for weigths
    center = N/2.0
    x = np.arange(N) - center +.5
    # initiate array
    if initial is None:
        g = np.ones((N,N),dtype=np.dtype(float))
    else:
        g = 1*initial[::-1]
    # Make sure all angles are in [0,PI)
    for i in np.arange(len(angles)):
        if angles[i] > np.pi:
            offset = np.floor(angles[i]/np.pi)
            angles[i] -= offset*np.pi
        elif angles[i] < 0:
            offset = np.floor(np.abs((angles[i]/np.pi)))+1
            angles[i] += offset*np.pi
        if angles[i] == np.pi:
            angles[i] = 0
            
    # g^k --> g^[iteration] is consistent with Kak, Slaney
    # k is the number of the current iteration
    # i iterates the rays of the projected image (1 to len(p))
    # j iterates over the entire flattened g
    # A[i,j] holds the weights for each ray i for all? angles l.
    # a[i] denotes the ith row vector of the array A[i,j]
    
    # g^[iteration] += a[i]*(p[i]-a[i]*g^[iteration])/(a[i]*a[i])
    
    g = g.flatten()
    
    if user_interface is not None:
        pid = user_interface.progress_new(steps=iterations*len(angles),
                                        task="SART.{}".format(os.getpid()))

    for k in np.arange(iterations):
        #
        # k iterates the iterations
        # i iterates the rays
        # j iterates the flattened outarr
        # l iterates the angles
        # m iterates the points from which each projection p[i]
        #   is computed
        #
        # initiate array
        dgall = np.zeros((len(angles),N,N))
        for l in np.arange(len(angles)):
            # Differences accumulated over one angle
            #dg = np.zeros((N,N), dtype=np.float64)
            dg = dgall[l]
            ai_sum = np.zeros((N,N))
            
            # From now on we work in radians
            angle_l = angles[l]
            # p[i] is consistent with Kak, Slaney
            p = sinogram[l]
            ## Calculate the weights for each p[i] at the current angle
            # p[i] = sum(g[j]*A[i,j])
            # A[i,j] = sum(d[i,j,m]*ds)
            # ds - distance between points, we let it default to half
            # a pixel size.
            
            # We will go through the procedure of calculating the
            # d[i,j,m]'s and A[i,j]'s.
            #print "angle %d" % l
            
            for i in np.arange(N):
                # Get the coordinates m at which we want to get the 
                # interpolations.
                # The resolution is equal to the input resolution
                # Crop the line according to its length in the
                # circular shaped area:
                # Image radius
                #ir = (N+1)/2.
                ir = center
                # x is centered
                dist = np.sqrt(ir**2-x[i]**2)
                # Delta s: distance between sample points
                # Setting it to half the pixel spacing:
                length = 2.*dist
                num_points = 2*np.ceil(2*dist)+1
                # endpoints are on the reconstruction circle.
                # Therefore "-1"
                ds = length/(num_points-1)
                
                # x-values of the line in the non-rotated coordinate
                # system:
                xline = np.linspace(-dist,dist,num_points,endpoint=True)
                # y-values are constant in the non-rotated system
                yline = x[i] * np.ones(len(xline))
                
                # Rotate the coordinate system of the line
                # These are the actual coordinates in the reconstructed
                # image that where used to sum up p[i]
                xrot = xline*np.cos(angle_l) - yline*np.sin(angle_l)
                yrot = xline*np.sin(angle_l) + yline*np.cos(angle_l)

                # Bispline interpolation at the points xrot and yrot
                # This for-loop slows down computation time. We replaced
                # it (see below). The replacement is a little cryptic.
                # Therefore we keep it here.
                    #dijm = np.zeros((N,N,num_points))
                    #for m in np.arange(num_points):
                    #    # get the four pixels that surround the
                    #    # index positions g[px][py]
                    #    px = xrot[m] + center -.5
                    #    py = yrot[m] + center -.5
                    #    
                    #    # According to quadrants
                    #    P1 = [min(np.ceil(px),N-1), min(np.ceil(py),N-1)]
                    #    P2 = [np.floor(px), min(np.ceil(py),N-1)]
                    #    P3 = [np.floor(px), np.floor(py)]
                    #    P4 = [min(np.ceil(px),N-1), np.floor(py)]
                    #    
                    #    # print px, P1[0], P2[0], P3[0], P4[0]
                    #    # Calculate the weigths ai
                    #    
                    #    # See wikipedia article
                    #    # f(0,0) is at P3
                    #    prelx = px-P3[0]
                    #    prely = py-P3[1]
                    #    ## f(1,1)
                    #    dijm[P1[0]][P1[1]][m] = prelx*prely
                    #    ## f(0,1)
                    #    dijm[P2[0]][P2[1]][m] = (1-prelx)*prely
                    #    ## f(0,0)
                    #    dijm[P3[0]][P3[1]][m] = (1-prelx)*(1-prely)
                    #    ## f(1,0)
                    #    dijm[P4[0]][P4[1]][m] = prelx*(1-prely)
                    # ai = ds*np.sum(dijm,axis=2)
                    
                # We spare us the summation over m by summing directly
                dij = np.zeros((N,N),dtype=np.float64)
                
                px = xrot + center -.5
                py = yrot + center -.5
                
                P1x = np.uint64(np.ceil(px)*(np.ceil(px) < N-1) + (N-1)*(np.ceil(px) >= N-1))
                P1y = np.uint64(np.ceil(py)*(np.ceil(py) < N-1) + (N-1)*(np.ceil(py) >= N-1))
                P2x = np.uint64(np.floor(px)*(np.floor(px) > 0))
                P2y = np.uint64(np.ceil(py)*(np.ceil(py) < N-1) + (N-1)*(np.ceil(py) >= N-1))
                P3x = np.uint64(np.floor(px)*(np.floor(px) > 0))
                P3y = np.uint64(np.floor(py)*(np.floor(py) > 0))
                P4x = np.uint64(np.ceil(px)*(np.ceil(px) < N-1) + (N-1)*(np.ceil(px) >= N-1))
                P4y = np.uint64(np.floor(py)*(np.floor(py) > 0))

                prelx = px-P3x
                prely = py-P3y

                Px = np.array([P1x, P2x, P3x, P4x]).flatten()
                Py = np.array([P2y, P2y, P3y, P4y]).flatten()
                
                Pdelta = np.array([ prelx*prely,
                                    (1-prelx)*prely,
                                    (1-prelx)*(1-prely),
                                    prelx*(1-prely)]).flatten()
                
                # Add a factor of 0.5 for points at beginning and end
                Pdelta[[0, len(prelx)-1, len(prelx), 2*len(prelx)-1,
                       2*len(prelx), 3*len(prelx)-1, 3*len(prelx),
                       4*len(prelx)-1]] *= 0.5
                
                # With the help of the StackOverflow.com community
                # we got rid of the for loop over m:
                # convert yourmulti-dim indices to flat indices
                flat_idx = np.ravel_multi_index((Px, Py), dims=dij.shape)
                # extract the unique indices and their position
                unique_idx, idx_idx = np.unique(flat_idx, return_inverse=True)
                # Aggregate the repeated indices 
                deltas = np.bincount(idx_idx, weights=Pdelta)
                # Sum them to your array
                dij.flat[unique_idx] += deltas

                ai = ds*dij
                del dij
                
                ai_dot = np.sum(np.square(ai))

                # So for each pixel that had just contributed, we will
                # add a value to the difference array dg.
                dg += ai * ( p[i] - np.sum(ai.flatten()*g) )/ai_dot

                ai_sum += ai

                del ai
                del ai_dot
            # This illustrates how the black ring forms on the outside.
            # If a low resolution 128x128 is used, one can see the
            # pixelation error.
            #proc_arr2im((g+dg.flatten()).reshape((N,N))[::-1], cut=False).save(os.path.join(DIR,"test/%08d.bmp" % l))
            #proc_arr2im((ai_sum*255/np.max(ai_sum)).reshape((N,N))[::-1], cut=False).save(os.path.join(DIR,"test/ai%08d.bmp" % l))
            dg[np.where(ai_sum != 0)] /= ai_sum[np.where(ai_sum != 0)]
            del ai_sum
            if user_interface is not None:
                user_interface.progress_iterate(pid)

        ## Only apply the average from all differences dgall
        ## ->leads to slower convergence then ART, but is more accurate
        dgall=np.average(dgall,axis=0).flatten()
        g += dgall
        ##g += dg/ai_sum

        del dgall
         # # Export single steps as bmp files...
         # #if i%10 == 1:
         # #    image = f.reshape((N,N))[::-1]
         # #    proc_arr2im(image, cut=True).save(os.path.join(DIR,"test/Elephantulus_small_art_%02d_%04d_%08d.bmp" % (iteration,k,i)))
         
    # compare input and output:
    #proc_arr2im(g.reshape((N,N)), cut=False).save(os.path.join(DIR,"sart_proc.bmp"))
    #proc_arr2im(initial[::-1], cut=False).save(os.path.join(DIR,"sart_ref.bmp"))

    # By slicing in-place [::-1] we get rid of the inversion of the
    # image along the y-axis.
    if user_interface is not None:
        user_interface.progress_finalize(pid)
    
    return g.reshape((N,N))[::-1]
