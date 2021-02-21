import multiprocessing as mp
import numpy as np
import time

from typing import Callable


def do_work(in_queue, out_list, count, max_count):
    while True:
        item = in_queue.get()
        # exit signal
        if item == "STOP":
            return

        func2d, kwargs, index = item
        result = func2d(count=count, max_count=max_count, **kwargs)

        out_list.append((index, result))
        time.sleep(.01)


def volume_recon(func2d: Callable, sinogram: np.ndarray = None,
                 angles: np.ndarray = None,
                 count=None, max_count=None, ncpus=None,
                 **kwargs) -> np.ndarray:
    """Slice-wise 3D inversion of the Radon transform

    Computes the slice-wise 3D inverse of the Radon transform using
    multiprocessing.


    Parameters
    ----------
    func2d: callable
        A method for the slice-wise reconstruction
        (e.g. :func:`backproject`).
    sinogram: ndarray, shape (A,M,N)
        Three-dimensional sinogram of line recordings. The axis `1`
        iterates through the `M` slices. The rotation takes place through
        axis `1`.
    angles: (A,) ndarray
        Angular positions of the `sinogram` in radians in the
        interval [0, PI).
    count, max_count: multiprocessing.Value or `None`
        Can be used to monitor the progress of this function. The
        value of `max_count.value` is set initially and the value
        of `count.value` is incremented until it reaches the end
        of the algorithm (`max_count.value`).
    **kwargs: dict
        Additional keyword arguments to `func2d`.

    Returns
    -------
    out: ndarray, shape (N,M,N)
        The reconstructed volume.

        .. versionchanged:: 0.4.0

            Output indexing now follows the ODTbrain convention. For the
            the old behavior, use ``out.transpose(1, 0, 2)``.
    """
    if sinogram.shape[0] != angles.shape[0]:
        msg = "First dimension of `sinogram` must match size of `angles`"
        raise ValueError(msg)

    if len(sinogram.shape) != 3:
        msg = "`sinogram` must have three dimensions."
        raise ValueError(msg)

    if ncpus is None:
        ncpus = mp.cpu_count()

    fkw = kwargs.copy()
    fkw["angles"] = angles

    manager = mp.Manager()
    results = manager.list()
    work = manager.Queue()

    # kick-off working processes
    pool = []
    max_counts = [mp.Value("I", 0, lock=True) for _ii in range(ncpus)]
    for ii in range(ncpus):
        p = mp.Process(target=do_work, args=(work, results,
                                             count, max_counts[ii]))
        p.start()
        pool.append(p)

    # add first slice job (we need to do this to get an estimate of max_count)
    fkwj = fkw.copy()
    fkwj["sinogram"] = sinogram[:, 0, :]
    work.put((func2d, fkwj, 0))

    # determine max_count for a single slice and set it globally
    if max_count is not None:
        for _ii in range(60):  # wait max 6s
            time.sleep(.01)
            initial_max_count = np.max([c.value for c in max_counts])
            if initial_max_count != 0:
                break
        with max_count.get_lock():
            max_count.value = sinogram.shape[1] * initial_max_count

    # add other slice jobs
    for jj in range(1, sinogram.shape[1]):
        fkwj = fkw.copy()
        fkwj["sinogram"] = sinogram[:, jj, :]
        work.put((func2d, fkwj, jj))

    # put stop signal at end of queue to let workers know they may stop
    for _kk in range(ncpus):
        work.put("STOP")

    for p in pool:
        p.join()

    sh = sinogram.shape
    out = np.zeros((sh[2], sh[1], sh[2]))
    for ii in range(len(results)):
        idx, res = results[ii]
        out[:, idx, :] = res

    return out
