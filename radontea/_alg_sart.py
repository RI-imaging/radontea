import numpy as np


def sart(sinogram: np.ndarray, angles: np.ndarray, initial: np.ndarray = None,
         iterations: int = 1, count=None, max_count=None):
    """Simultaneous Algebraic Reconstruction Technique


    SART computes an inverse of the Radon transform in two dimensions.
    The reconstruction technique uses "rays" of the diameter of
    one pixel to iteratively solve the system of linear equations
    that describe the image. The weighting factors are bilinear
    elements. At the beginning and end of each ray, only partial
    weights are used. The pixel values of the image are updated
    only after each iteration is complete.


    Parameters
    ----------
    sinogram: ndarrayy, shape (A,N)
        Two-dimensional sinogram of line recordings.
    angles: ndarray, length A
        Angular positions of the `sinogram` in radians. The angles
        at which the sinogram slices were recorded do not have to be
        distributed equidistantly as in backprojection techniques.
        The angles are internaly converted to modulo PI.
    initial: ndarray, shape (N,N), optional
        the initial guess for the solution.
    iterations: integer
        Number of iterations to perform.
    count, max_count: multiprocessing.Value or `None`
        Can be used to monitor the progress of the algorithm.
        Initially, the value of `max_count.value` is incremented
        by the total number of steps. At each step, the value
        of `count.value` is incremented.


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
    # make sure `iterations` is an integer
    iterations = int(iterations)
    N = sinogram.shape[1]
    A = angles.shape[0]

    if max_count is not None:
        with max_count.get_lock():
            max_count.value += A * iterations + 1
    # Meshgrid for weigths
    center = N / 2.0
    x = np.arange(N) - center + .5
    # initiate array
    if initial is None:
        g = np.ones((N, N), dtype=np.dtype(float))
    else:
        g = 1 * initial[::-1]
    # Make sure all angles are in [0,PI)
    for i in np.arange(A):
        if angles[i] > np.pi:
            offset = np.floor(angles[i] / np.pi)
            angles[i] -= offset * np.pi
        elif angles[i] < 0:
            offset = np.floor(np.abs((angles[i] / np.pi))) + 1
            angles[i] += offset * np.pi
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

    if count is not None:
        with count.get_lock():
            count.value += 1

    for kk in np.arange(iterations):  # @UnusedVariable
        #
        # kk iterates the iterations
        # ii iterates the rays
        # jj iterates the flattened outarr
        # ll iterates the angles
        # mm iterates the points from which each projection p[i]
        #   is computed
        #
        # initiate array
        dgall = np.zeros((A, N, N))
        for ll in np.arange(A):
            # Differences accumulated over one angle
            dg = dgall[ll]
            ai_sum = np.zeros((N, N))

            # From now on we work in radians
            angle_l = angles[ll]
            # p[i] is consistent with Kak, Slaney
            p = sinogram[ll]
            # Calculate the weights for each p[i] at the current angle
            # p[i] = sum(g[j]*A[i,j])
            # A[i,j] = sum(d[i,j,m]*ds)
            # ds - distance between points, we let it default to half
            # a pixel size.

            # We will go through the procedure of calculating the
            # d[i,j,m]'s and A[i,j]'s.
            # print "angle %d" % l

            for ii in np.arange(N):
                # Get the coordinates m at which we want to get the
                # interpolations.
                # The resolution is equal to the input resolution
                # Crop the line according to its length in the
                # circular shaped area:
                # Image radius
                ir = center
                # x is centered
                dist = np.sqrt(ir**2 - x[ii]**2)
                # Delta s: distance between sample points
                # Setting it to half the pixel spacing:
                length = 2. * dist
                num_points = 2 * int(np.ceil(2 * dist)) + 1
                # endpoints are on the reconstruction circle.
                # Therefore "-1"
                ds = length / (num_points - 1)

                # x-values of the line in the non-rotated coordinate
                # system:
                xline = np.linspace(-dist, dist, num_points, endpoint=True)
                # y-values are constant in the non-rotated system
                yline = x[ii] * np.ones(len(xline))

                # Rotate the coordinate system of the line
                # These are the actual coordinates in the reconstructed
                # image that where used to sum up p[ii]
                xrot = xline * np.cos(angle_l) - yline * np.sin(angle_l)
                yrot = xline * np.sin(angle_l) + yline * np.cos(angle_l)

                # Bispline interpolation at the points xrot and yrot
                # This for-loop slows down computation time. We replaced
                # it (see below). The replacement is a little cryptic.
                # Therefore we keep it here.
                # dijm = np.zeros((N,N,num_points))
                # for m in np.arange(num_points):
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
                dij = np.zeros((N, N), dtype=np.float64)

                px = xrot + center - .5
                py = yrot + center - .5

                P1x = np.uint64(np.ceil(px) * (np.ceil(px) < N - 1)
                                + (N - 1) * (np.ceil(px) >= N - 1))
                P1y = np.uint64(np.ceil(py) * (np.ceil(py) < N - 1)
                                + (N - 1) * (np.ceil(py) >= N - 1))
                P2x = np.uint64(np.floor(px) * (np.floor(px) > 0))
                P2y = P1y
                P3x = P2x
                P3y = np.uint64(np.floor(py) * (np.floor(py) > 0))
                P4x = P1x
                P4y = P3y

                prelx = px - P3x
                prely = py - P3y

                Px = np.array([P1x, P2x, P3x, P4x]).flatten()
                Py = np.array([P1y, P2y, P3y, P4y]).flatten()

                Pdelta = np.array([prelx * prely,
                                   (1 - prelx) * prely,
                                   (1 - prelx) * (1 - prely),
                                   prelx * (1 - prely)]).flatten()

                # Add a factor of 0.5 for points at beginning and end
                Pdelta[[0, len(prelx) - 1, len(prelx), 2 * len(prelx) - 1,
                        2 * len(prelx), 3 * len(prelx) - 1, 3 * len(prelx),
                        4 * len(prelx) - 1]] *= 0.5

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

                ai = ds * dij
                del dij

                ai_dot = np.sum(np.square(ai))

                # So for each pixel that had just contributed, we will
                # add a value to the difference array dg.
                dg += ai * (p[ii] - np.sum(ai.flatten() * g)) / ai_dot

                ai_sum += ai

                del ai
                del ai_dot
            # This illustrates how the black ring forms on the outside.
            # If a low resolution 128x128 is used, one can see the
            # pixelation error.
            dg[np.where(ai_sum != 0)] /= ai_sum[np.where(ai_sum != 0)]
            del ai_sum
            if count is not None:
                with count.get_lock():
                    count.value += 1

        # Only apply the average from all differences dgall
        # ->leads to slower convergence then ART, but is more accurate
        dgall = np.average(dgall, axis=0).flatten()
        g += dgall

        del dgall
    # By slicing in-place [::-1] we get rid of the inversion of the
    # image along the y-axis.

    return g.reshape((N, N))[::-1]
