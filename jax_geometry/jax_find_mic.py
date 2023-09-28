import os
import itertools
from functools import partial
from typing import Callable

import jax
import numpy as np
from jax import numpy as jnp
from ase.geometry import minkowski_reduce
from ase.geometry.cell import complete_cell
from ase.cell import Cell

DEFAULT_PLATFORM = 'cpu'
DEFAULT_CORE_NUM = os.cpu_count()

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', DEFAULT_PLATFORM)
os.environ['XLA_FLAGS'] = '--{:s}={:d}'.format(
    "xla_force_host_platform_device_count",
    DEFAULT_CORE_NUM)


@jax.jit
def pbc2pbc(pbc):
    result = jnp.zeros(3, dtype=bool)
    return result.at[:].set(pbc)


@jax.jit
def frc2car(f, cell):
    """Calculate Cartesian positions from scaled positions."""
    return jnp.asarray(f) @ jnp.asarray(cell)


@jax.jit
def car2frc(v, cell):
    """Calculate scaled positions from Cartesian positions."""
    return jnp.linalg.solve(jnp.asarray(cell).T, jnp.asarray(v).T).T


@jax.jit
def fn_mic_nopbc(v, cell):
    return v


@jax.jit
def fn_mic_naive(v, cell):
    """Finds the minimum-image representation of vector(s) v.

    Safe to use for (pbc.all() and (norm(v_mic) < 0.5 * min(cell.lengths()))).
    Can otherwise fail for non-orthorhombic cells.
    Described in:
    W. Smith, "The Minimum Image Convention in Non-Cubic MD Cells", 1989,
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1696.
    """
    frac = car2frc(v, cell)
    frac = frac - jnp.floor(frac + 0.5)
    return frac @ cell


@jax.jit
def wrap_positions(v, cell, pbc=True):
    frac = car2frc(jnp.asarray(v), jnp.asarray(cell))
    frac_wrapped = jnp.where(pbc2pbc(pbc), frac % 1.0, frac)
    return frac_wrapped @ jnp.asarray(cell)


@jax.jit
def fn_mic_general(v, cell, pbc, hkls, r_op):
    v = jnp.asarray(v, dtype=float)
    cell = jnp.asarray(cell, dtype=float)
    r_op = jnp.asarray(r_op, dtype=float)
    hkls = jnp.asarray(hkls, dtype=float)

    rcell = r_op @ cell
    vrvecs = hkls @ rcell
    v = wrap_positions(v, rcell, pbc=pbc)

    if v.ndim == 1:
        x = v + vrvecs
        lengths = jnp.linalg.norm(x, axis=1)
        indices = jnp.argmin(lengths, axis=0)
        return x[indices, :]
    elif v.ndim == 2:
        x = v + vrvecs[:, None]
        lengths = jnp.linalg.norm(x, axis=2)
        indices = jnp.argmin(lengths, axis=0)
        return x[indices, jnp.arange(len(v)), :]
    else:
        raise KeyError("v.ndim must <= 2.")


class CellWithPBC:

    def __init__(self, cell=None, pbc=None,
                 cell_as_param: bool = False):
        self.pbc = pbc2pbc(pbc if pbc else False)
        self._cell_as_param = cell_as_param
        self._cell = Cell.new(cell)

    def __repr__(self):
        if self._cell.orthorhombic:
            numbers = self._cell.lengths().tolist()
        else:
            numbers = self._cell.tolist()
        return 'Cell({})'.format(numbers)

    @property
    def fn_mic(self) -> Callable:
        cell = jnp.asarray(self._cell.array)
        dim = jnp.sum(self._cell.any(1) & self.pbc)
        if dim == 0:
            func = fn_mic_nopbc
        elif dim == 3 and self._cell.orthorhombic:
            func = fn_mic_naive
        else:
            pbc = self.pbc.tolist()
            cell = complete_cell(self._cell)
            r_op = minkowski_reduce(cell, pbc)[1]
            r_op = jnp.asarray(r_op, dtype=float)
            ranges = [np.arange(-1 * p, p + 1) for p in pbc]
            hkls = [(0, 0, 0)] + list(itertools.product(*ranges))
            hkls = jnp.asarray(hkls, dtype=float)
            func = partial(fn_mic_general, pbc=self.pbc,
                           hkls=hkls, r_op=r_op)
        if not self._cell_as_param:
            cell = jnp.asarray(cell, dtype=float)
            return partial(func, cell=cell)
        else:
            return func