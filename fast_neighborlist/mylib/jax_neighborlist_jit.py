import os
import itertools
from typing import List, Union

import jax
import jax.numpy as jnp
from ase.atoms import Atoms
from ase.geometry import (minkowski_reduce,
                          wrap_positions)

from .ase_neighborlist import _get_cutoffs


DEFAULT_PLATFORM = 'cpu'
DEFAULT_CORE_NUM = os.cpu_count()

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', DEFAULT_PLATFORM)
os.environ['XLA_FLAGS'] = '--{:s}={:d}'.format(
    "xla_force_host_platform_device_count",
    DEFAULT_CORE_NUM)


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

    @jax.jit
    def _f(i, j, kk, cutoffs, pos, n123, rcell):
        disp = n123[kk] @ rcell
        delta = pos[i] - (pos[j] + disp)
        cutoff = cutoffs[i] + cutoffs[j]
        d = jnp.linalg.norm(delta)
        is_nb = d <= cutoff
        return is_nb

    @jax.jit
    def f(cutoffs, pos, n123, rcell):
        cutoffs = jnp.asarray(cutoffs, dtype=float)
        rcell = jnp.asarray(rcell, dtype=float)
        pos = jnp.asarray(pos, dtype=float)
        n123 = jnp.asarray(n123, dtype=int)

        ij_arr = jnp.arange(pos.shape[0])
        kk_arr = jnp.arange(n123.shape[0])

        result = jax.vmap(jax.vmap(jax.vmap(
            _f, in_axes=(0, None, None, None, None, None, None)
        ), in_axes=(None, 0, None, None, None, None, None)
        ), in_axes=(None, None, 0, None, None, None, None)
        )(ij_arr, ij_arr, kk_arr, cutoffs, pos, n123, rcell)
        return result

    @jax.jit
    def kij2ijS(k, i, j, n123, offsets):
        S = n123[k] @ r_op + offsets[i] - offsets[j]
        ijS = jnp.row_stack([
            jnp.column_stack([i, j, S]),
            jnp.column_stack([j, i, -S])])
        return ijS

    tmp = f(r_cutoffs, pos_mkwsk, n123, rcell)
    k, i, j = jnp.where(tmp)
    self_interaction = jnp.logical_and(
        (n123[k] == 0).all(1), i == j)
    kij = jnp.column_stack([k, i, j])
    kij = kij[~self_interaction]
    k, i, j = kij.T

    ijS = kij2ijS(k, i, j, n123, offsets)
    return jnp.unique(ijS, axis=0)


if __name__ == '__main__':
    import time
    from ase.build import bulk
    from ase_neighborlist import nb_3loop, nb_kdtree

    atoms = bulk('Cu', 'bcc', 3.5) * (10, 10, 1)
    ijS_2 = nb_jax(atoms, 12.3)

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
          "  all+jax        time_2: {:8.3f} s\n".format(c-b),)
    print("================================")

    assert jnp.allclose(jnp.unique(ijS_1, axis=0),
                        jnp.unique(ijS_2, axis=0))
    assert jnp.allclose(ijS_1, ijS_3)
