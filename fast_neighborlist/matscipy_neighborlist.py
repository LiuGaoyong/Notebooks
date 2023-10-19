from typing import List, Union
from numbers import Real

import numpy as np
from ase import Atoms
from matscipy.neighbours import neighbour_list

from mylib.ase_neighborlist import _get_cutoffs


def matscipy_nb(atoms: Atoms, r_cutoffs: Union[float, List[float]]):
    r_cutoffs = np.asfarray(_get_cutoffs(atoms.numbers, r_cutoffs))
    i, j, S = neighbour_list('ijS', atoms, cutoff=r_cutoffs)
    ijS = np.column_stack([i, j, S])
    return ijS[np.lexsort((i, j))]


if __name__ == '__main__':
    from ase.build import bulk
    from ase.utils.timing import Timer
    from mylib.ase_neighborlist import ase_3loop, ase_kdtree, ase_kdtree_out
    timer = Timer()

    atoms = bulk('Cu', 'bcc', 3.5) * (30, 30, 30)
    cutoff = 12

    print(f"Natoms: {len(atoms)}\n" +
          f"Cutoff: {float(cutoff)}")

    ijS = []
    for func in (
            # ase_3loop,
            # ase_kdtree,
            ase_kdtree_out,
            matscipy_nb):
        with timer("{}".format(func.__name__)):
            ijS.append(func(atoms, cutoff))
    timer.write()

    # for i in range(1, len(ijS)):
    #     np.testing.assert_equal(ijS[i], ijS[0])


# ================================================================
# Natoms: 27
# Cutoff: 12.0
# Timing:             incl.     excl.
# ------------------------------------------
# ase_3loop:          1.621     1.621  52.1% |--------------------|
# ase_kdtree:         1.143     1.143  36.8% |--------------|
# ase_kdtree_out:     0.311     0.311  10.0% |---|
# matscipy_nb:        0.033     0.033   1.1% |
# Other:              0.002     0.002   0.0% |
# ------------------------------------------
# Total:                        3.111 100.0%
# =================================================================
# Natoms: 2700
# Cutoff: 12.0
# Timing:             incl.     excl.
# ------------------------------------------
# ase_kdtree_out:     7.635     7.635  65.2% |-------------------------|
# matscipy_nb:        4.018     4.018  34.3% |-------------|
# Other:              0.052     0.052   0.4% |
# ------------------------------------------
# Total:                       11.705 100.0%
# =================================================================
# Natoms: 27000
# Cutoff: 12.0
# Timing:             incl.     excl.
# ------------------------------------------
# ase_kdtree_out:    57.237    57.237  56.8% |----------------------|
# matscipy_nb:       42.997    42.997  42.7% |----------------|
# Other:              0.506     0.506   0.5% |
# ------------------------------------------
# Total:                      100.741 100.0%
