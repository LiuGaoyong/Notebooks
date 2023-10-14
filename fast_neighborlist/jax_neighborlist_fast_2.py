import itertools
from typing import List, Union

import jax.numpy as jnp
from ase.atoms import Atoms
from ase.geometry import (minkowski_reduce,
                          wrap_positions)
from scipy.spatial import cKDTree

from ase_neighborlist import _get_cutoffs


def _nb_prepare(atoms: Atoms, r_cutoffs: Union[float, List[float]]):
    r_cutoffs = jnp.asarray(_get_cutoffs(atoms.numbers, r_cutoffs))
    positions0, rcmax = atoms.positions, r_cutoffs.max()
    rcell, op = minkowski_reduce(atoms.cell, atoms.pbc)
    positions = wrap_positions(positions0, cell=rcell,
                               pbc=atoms.pbc, eps=0)
    offsets = atoms.cell.scaled_positions(
        positions - positions0
    ).round().astype(int)

    N = []
    rcell = jnp.asarray(op @ atoms.cell.array)
    ircell = jnp.linalg.pinv(rcell)
    for i in range(3):
        if atoms.pbc[i]:
            v = ircell[:, i]
            h = 1 / jnp.linalg.norm(v)
            n = int(2 * rcmax / h) + 1
        else:
            n = 0
        N.append(n)

    n123 = jnp.asarray([
        (n1, n2, n3)
        for n1, n2, n3 in itertools.product(
            range(0, N[0] + 1),
            range(-N[1], N[1] + 1),
            range(-N[2], N[2] + 1)
        ) if not (n1 == 0 and (n2 < 0 or n2 == 0 and n3 < 0))])
    return rcell, op, offsets, n123, positions, r_cutoffs


def nb_jax(atoms: Atoms, r_cutoffs: Union[float, List[float]]):
    rcell, r_op, offsets, n123, pos_mkwsk, r_cutoffs = _nb_prepare(
        atoms=atoms, r_cutoffs=r_cutoffs)

    disp = jnp.asarray(n123) @ rcell
    kk_pos = (pos_mkwsk[:, None] - disp).reshape(-1, 3)

    tree = cKDTree(pos_mkwsk, copy_data=False)
    kk_indices = tree.query_ball_point(
        kk_pos, jnp.max(r_cutoffs)*2, workers=-1)

    natoms = len(pos_mkwsk)
    i, kk = jnp.asarray([(i, kk)
                         for kk, i_list in enumerate(kk_indices)
                         for i in i_list], dtype=int).T
    j, k = kk % natoms, kk // natoms

    # assert jnp.allclose(pos_mkwsk[j] - disp[k], kk_pos[kk])
    cutoff = r_cutoffs[i] + r_cutoffs[j]
    delta = pos_mkwsk[i] - kk_pos[kk]
    d = jnp.linalg.norm(delta, axis=1)

    cond = jnp.logical_and(
        d < cutoff,
        jnp.logical_not(
            jnp.logical_and(
                (n123[k] == 0).all(1),
                i == j)))
    kij = jnp.column_stack([k, i, j])
    k, i, j = kij[cond].T

    S = n123[k] @ r_op + offsets[i] - offsets[j]
    ijS = jnp.row_stack([
        jnp.column_stack([i, j, S]),
        jnp.column_stack([j, i, -S])])
    return jnp.unique(ijS, axis=0)


if __name__ == '__main__':
    import time
    from ase.build import bulk
    from ase_neighborlist import nb_3loop, nb_kdtree

    atoms = bulk('Cu', 'bcc', 3.5)
    a = time.time()
    ijS_1 = nb_3loop(atoms, 12.3)
    b = time.time()
    ijS_2 = nb_jax(atoms, 12.3)
    c = time.time()
    ijS_3 = nb_kdtree(atoms, 12.3)
    d = time.time()

    print(f"Result:\n"
          f"  length_1: {len(ijS_1)}\n"
          f"  length_2: {len(ijS_2)}\n")
    print(f"Time:\n",
          "  3loop          time_1: {:8.3f} s\n".format(b-a),
          "  kdtree+loop    time_2: {:8.3f} s\n".format(d-c),
          "  kdtree+jax     time_2: {:8.3f} s\n".format(c-b),)
    print("================================")

    assert jnp.allclose(ijS_1, ijS_2)
    assert jnp.allclose(ijS_1, ijS_3)
