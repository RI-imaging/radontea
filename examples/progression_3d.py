"""Progress monitoring with progression

The progress of the reconstruction algorithms in radontea
can be tracked with other packages such as
`progression <https://cimatosa.github.io/progression/>`_.
"""
from multiprocessing import cpu_count

import numpy as np
import progression as pr

import radontea as rt


A = 55    # number of angles
N = 128   # detector size x
M = 24    # detector size y (number of slices)

# generate random data
sino0 = np.random.random((A, N))
sino = np.random.random((A, M, N))
sino[:, 0, :] = sino0
angles = np.linspace(0, np.pi, A)

count = pr.UnsignedIntValue()
max_count = pr.UnsignedIntValue()


with pr.ProgressBar(count=count,
                    max_count=max_count,
                    interval=0.3
                    ) as pb:
    pb.start()
    rt.volume_recon(func2d=rt.sart, sinogram=sino, angles=angles,
                    ncpus=cpu_count(), count=count, max_count=max_count)
