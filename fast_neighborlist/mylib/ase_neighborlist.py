from typing import List, Union
from numbers import Real

import numpy as np
from ase import Atoms
from ase.data import covalent_radii as COV_R
from ase.neighborlist import PrimitiveNeighborList, primitive_neighbor_list

from .ase_neighborlist_fast import FastPrimitiveNeighborList


def _get_cutoffs(Z: List[int],
                 x: Union[float, List[float]] = None,
                 ) -> List[float]:
    """Get the cutoffs radii for the given atoms(alias Z).

    Args:
        Z (List[int]): the atomic number of atoms
        x (Union[float, List[float]], optional): the cutoff input value.
            If it is a float number, it will be repeated len(Z) times.
            If it is a list, it will be returned directly.
            If it is none, covalent radii will be returned.
            Defaults to None.

    Returns:
        List[float]: the cutoffs radii for the given atoms
    """
    if x is None:
        return COV_R[Z]
    elif isinstance(x, Real):
        return [float(x), ] * len(Z)
    else:
        if len(Z) != len(x):
            raise KeyError("The x must have same length with Z, "
                           "if x is a list.")
        return [float(i) for i in x]


def ase_3loop(atoms: Atoms, r_cutoffs: Union[float, List[float]]):
    r_cutoffs = np.asfarray(_get_cutoffs(atoms.numbers, r_cutoffs))
    i, j, S, d, D = primitive_neighbor_list(
        'ijSdD', atoms.pbc, atoms.cell, atoms.positions,
        cutoff=r_cutoffs, self_interaction=False,
        use_scaled_positions=False)
    ijS = np.column_stack([i, j, S])
    return ijS[np.lexsort((i, j))]


def ase_kdtree(atoms: Atoms, r_cutoffs: Union[float, List[float]]):
    r_cutoffs = np.asfarray(_get_cutoffs(atoms.numbers, r_cutoffs))
    nl = PrimitiveNeighborList(cutoffs=r_cutoffs, skin=0,
                               self_interaction=False, bothways=True,
                               use_scaled_positions=False)
    nl.build(atoms.pbc, atoms.cell, atoms.positions)
    ijS = []
    for i in range(len(atoms)):
        nb, disp = nl.get_neighbors(i)
        for ii in range(len(nb)):
            ijS.append(tuple([i, nb[ii]] + disp[ii].tolist()))
    np.asarray(ijS, dtype=int)
    return np.unique(ijS, axis=0)


def ase_kdtree_out(atoms: Atoms, r_cutoffs: Union[float, List[float]]):
    r_cutoffs = np.asfarray(_get_cutoffs(atoms.numbers, r_cutoffs))
    nl = FastPrimitiveNeighborList(cutoffs=r_cutoffs, skin=0,
                                   self_interaction=False, bothways=True,
                                   use_scaled_positions=False)
    nl.build(atoms.pbc, atoms.cell, atoms.positions)
    ijS = []
    for i in range(len(atoms)):
        nb, disp = nl.get_neighbors(i)
        for ii in range(len(nb)):
            ijS.append(tuple([i, nb[ii]] + disp[ii].tolist()))
    np.asarray(ijS, dtype=int)
    return np.unique(ijS, axis=0)
