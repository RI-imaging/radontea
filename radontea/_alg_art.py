import numpy as np


def art(sinogram: np.ndarray, angles: np.ndarray, initial: np.ndarray = None,
        iterations: int = 1, count=None, max_count=None) -> np.ndarray:
    """Algebraic Reconstruction Technique

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
    sinogram: ndarrayy, shape (A,N)
        Two-dimensional sinogram of line recordings.
    angles: ndarray, length A
        Angular positions of the `sinogram` in radians. The angles
        at which the sinogram slices were recorded do not have to be
        distributed equidistantly as in :func:`backproject`.
        The angles are internaly converted to modulo PI.
    initial: ndarray, shape (N,N), optional
        The initial guess for the solution.
    iterations: int
        Number of iterations to perform.
    count, max_count: multiprocessing.Value or `None`
        Can be used to monitor the progress of the algorithm.
        Initially, the value of `max_count.value` is incremented
        by the total number of steps. At each step, the value
        of `count.value` is incremented.


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
    N = sinogram.shape[1]
    A = angles.shape[0]

    if max_count is not None:
        with max_count.get_lock():
            max_count.value += A * iterations + 1
    # Meshgrid for weigths
    center = N / 2.0
    x = np.arange(N) - center + .5
    X, Y = np.meshgrid(x, x)
    # initiate array
    if initial is None:
        f = np.zeros((N, N), dtype=np.dtype(float))
    else:
        f = 1 * initial.transpose()[::-1]
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

    # These lambda functions return the two x- and y- values of two
    # points projected onto a line along the x- and y-axis, having the
    # angle angle_k in radians.
    # Sourcing this out to here makes things a little faster.
    def GetLambdaLines(angle_k):
        if angle_k == 0:
            # Divide by zero error for tan
            # We have horizontal lines (parallel to x)
            def line1(x2, y2, xpi1, ypi1, xpi2, ypi2, angle_k): return (
                x2 - .5, ypi1 * np.ones(y2.shape))

            def line2(x2, y2, xpi1, ypi1, xpi2, ypi2, angle_k): return (
                x2 + .5, ypi2 * np.ones(y2.shape))
        elif angle_k == np.pi / 2:
            # We have vertical lines (parallel to y)
            def line1(x2, y2, xpi1, ypi1, xpi2, ypi2, angle_k): return (
                xpi1 * np.ones(x2.shape), y2 + .5)

            def line2(x2, y2, xpi1, ypi1, xpi2, ypi2, angle_k): return (
                xpi2 * np.ones(x2.shape), y2 - .5)
        elif angle_k < np.pi / 2:
            # CASE 1
            # Compute any other positions on the lines from the given things
            def line1(x2, y2, xpi1, ypi1, xpi2, ypi2, angle_k):
                return ((y2 - ypi1) / np.tan(angle_k) + xpi1,
                        (x2 - xpi1) * np.tan(angle_k) + ypi1)
            # def line2(x2,y2):
            #    y = (x2-xpi2)/np.tan(angle_k) - ypi2
            #    x = (y2-ypi2)*np.tan(angle_k) - xpi2
            #    return x,y

            def line2(x2, y2, xpi1, ypi1, xpi2, ypi2, angle_k):
                return ((y2 - ypi2) / np.tan(angle_k) + xpi2,
                        (x2 - xpi2) * np.tan(angle_k) + ypi2)
        else:
            # CASE 2: Switch x-output - only for speed.
            # Not very obvious - possibly a hack.
            # Compute any other positions on the lines from the given things
            def line1(x2, y2, xpi1, ypi1, xpi2, ypi2, angle_k):
                return ((y2 - ypi2) / np.tan(angle_k) + xpi2,
                        (x2 - xpi1) * np.tan(angle_k) + ypi1)
            # def line2(x2,y2):
            #    y = (x2-xpi2)/np.tan(angle_k) - ypi2
            #    x = (y2-ypi2)*np.tan(angle_k) - xpi2
            #    return x,y

            def line2(x2, y2, xpi1, ypi1, xpi2, ypi2, angle_k):
                return ((y2 - ypi1) / np.tan(angle_k) + xpi1,
                        (x2 - xpi2) * np.tan(angle_k) + ypi2)
        return line1, line2
    # Sort angles?
    # This could increase the speed of convergence.

    # f[j] is consistent with Kak, Slaney
    f = f.flatten()

    if count is not None:
        with count.get_lock():
            count.value += 1

    for iteration in np.arange(iterations):  # @UnusedVariable
        #
        # i iterates the rays
        # j iterates the flattened outarr
        # k iterates the angles
        #
        for k in np.arange(A):
            # From now on we work in radians
            angle_k = angles[k]
            # p[i] is consistent with Kak, Slaney
            p = sinogram[k]
            # # We do not store the binary weights for each ray:
            # # For large images this thing could get big:
            # w = np.zeros((len(p),N*N), dtype=bool)
            # # w[i][j]
            line1, line2 = GetLambdaLines(angle_k)
            # CASES for all 2 quadrants. Angles are modulo PI
            # This case stuff is dependent on the angle. We enumerate
            # pr in negative mathematical angular direction
            #
            # This is the position on the projection, centered around 0:
            pr = np.arange(len(p)) - center + .5  # radial distance,
            #
            if angle_k <= np.pi / 2:
                # 0 to PI/2
                # case == 1
                # position of each p[i] in the centered *outarr*.
                # a position of line1 in X
                x_p1 = (pr - .5) * np.sin(angle_k)
                # a position of line1 in Y
                y_p1 = -(pr - .5) * np.cos(angle_k)
                # a position of line2 in X
                x_p2 = (pr + .5) * np.sin(angle_k)
                # a position of line2 in Y
                y_p2 = -(pr + .5) * np.cos(angle_k)
            else:
                # PI/2 to PI
                # case == 2
                # position of each p[i] in the centered *outarr*.
                # a position of line1 in X
                x_p1 = (pr + .5) * np.sin(angle_k)
                # a position of line1 in Y
                y_p1 = -(pr + .5) * np.cos(angle_k)
                # a position of line2 in X
                x_p2 = (pr - .5) * np.sin(angle_k)
                # a position of line2 in Y
                y_p2 = -(pr - .5) * np.cos(angle_k)
            for i in np.arange(len(p)):
                # If the angle is zero, then we are looking at the
                # projections onto the right side. The indices are
                # enumerated from bottom to top.
                w_i = np.zeros((N, N), dtype=bool)
                # # where does the ray cut w_i?
                # xpi1 = x_p1[i]
                # ypi1 = y_p1[i]
                # xpi2 = x_p2[i]
                # ypi2 = y_p2[i]
                # Check if a value is between the two lines (within ray)
                # For each value of X and Y, compute what the values of the
                # line would be.
                xl1, yl1 = line1(
                    X, Y, x_p1[i], y_p1[i], x_p2[i], y_p2[i], angle_k)
                xl2, yl2 = line2(
                    X, Y, x_p1[i], y_p1[i], x_p2[i], y_p2[i], angle_k)
                #
                AND = np.logical_and
                # Values lie between the two lines if the following is True:
                # Furthermore we restrict ourselves to a disk.
                w_i = AND(X**2 + Y**2 < center**2,
                          AND(AND(xl1 < X, X < xl2),
                              AND(yl1 > Y, Y > yl2))).flatten()
                #
                # i iterates the rays
                # j iterates the flattened outarr
                # k iterates the angles
                # In each iteration of the angles the image is changed
                # if np.sum(w_i) != 0:
                #    f += ( p[i] - np.sum(f*w_i) )/np.sum(w_i) * w_i
                # This is faster because we don't a the zeros.
                valid = np.where(w_i)
                f[valid] += (p[i] - np.sum(f[valid])) / np.sum(w_i)
            if count is not None:
                with count.get_lock():
                    count.value += 1
    # By slicing in-place [::-1] we get rid of the inversion of the
    # image along the y-axis.

    return f.reshape((N, N))[::-1].transpose()
