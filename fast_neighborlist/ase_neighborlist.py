from dataclasses import dataclass
from typing import List, Union
from numbers import Real

import numpy as np
from ase import Atoms
from ase.data import covalent_radii as COV_R
from ase.neighborlist import (
    PrimitiveNeighborList,
    primitive_neighbor_list)


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


@dataclass(eq=True, order=True)
class ijS_Data:
    i: int
    j: int
    S: List[int]

    def __post_init__(self):
        self.S = tuple(i for i in self.S)

    def __hash__(self) -> int:
        vars = (self.i, self.j) + self.S
        return hash(vars)


def nb_3loop(atoms: Atoms, r_cutoffs: Union[float, List[float]]):
    r_cutoffs = np.asfarray(_get_cutoffs(atoms.numbers, r_cutoffs))
    i, j, S, d, D = primitive_neighbor_list(
        'ijSdD', atoms.pbc, atoms.cell, atoms.positions,
        cutoff=r_cutoffs, self_interaction=False,
        use_scaled_positions=False)
    ijS = [ijS_Data(i[ii], j[ii], S[ii]) for ii in range(len(S))]
    return sorted(set(ijS))


def nb_kdtree(atoms: Atoms, r_cutoffs: Union[float, List[float]]):
    r_cutoffs = np.asfarray(_get_cutoffs(atoms.numbers, r_cutoffs))
    nl = PrimitiveNeighborList(cutoffs=r_cutoffs, skin=0,
                               self_interaction=False, bothways=False,
                               use_scaled_positions=False)
    nl.build(atoms.pbc, atoms.cell, atoms.positions)
    ijS = []
    for i in range(len(atoms)):
        nb, disp = nl.get_neighbors(i)
        for ii in range(len(nb)):
            ijS.append(ijS_Data(i, nb[ii], disp[ii]))
    return sorted(set(ijS))


if __name__ == '__main__':
    from pprint import pprint
    from ase.build import bulk

    atoms = bulk('Cu', 'fcc', 3.5)
    ijS_1 = nb_3loop(atoms, 1.3)
    ijS_2 = nb_kdtree(atoms, 1.3)

    # assert set(ijS_1) == set(ijS_2)

    print(f"Result:\n"
          f"  length_1: {len(ijS_1)}"
          f"  length_2: {len(ijS_2)}")
    pprint(ijS_1)
    pprint(ijS_2)
