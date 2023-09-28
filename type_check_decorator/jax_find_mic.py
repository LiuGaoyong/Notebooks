import os
import itertools
from functools import partial, wraps
from typing import Callable, Literal

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


################################################################
#                           一些装饰器
################################################################

def convert_to_jax(*input_names, dtype=float,
                   check_shape: Literal['coords', 'cell'] = None,
                   allow_1D_coords: bool = True,
                   ) -> Callable:
    n_input_names = len(input_names)
    if not isinstance(dtype, (list, tuple)):
        dtype = [dtype] * n_input_names
    else:
        assert len(dtype) == n_input_names
        dtype = list(dtype)
    if not isinstance(check_shape, (list, tuple)):
        check_shape = [check_shape] * n_input_names
    else:
        assert len(check_shape) == n_input_names
        check_shape = list(check_shape)
    if not isinstance(allow_1D_coords, (list, tuple)):
        allow_1D_coords = [allow_1D_coords] * n_input_names
    else:
        assert len(allow_1D_coords) == n_input_names
        allow_1D_coords = list(allow_1D_coords)

    def f(x, i: int, name: str):
        result = jnp.asarray(x, dtype=dtype[i])
        if check_shape[i] == 'cell':
            if result.shape != (3, 3):
                raise KeyError(f"The input of {name} must be 3x3 2D array.")
        elif check_shape[i] == 'coords':
            if result.ndim > 2:
                raise KeyError(f"The input of {name} cannot be "
                               "{:d}D array.".format(result.ndim))
            elif result.ndim == 2 and result.shape[1] != 3:
                raise KeyError(f"The input of {name} must have "
                               "Nx3 shape if it is 2D array.")
            elif (allow_1D_coords[i]
                    and result.ndim == 1
                    and result.shape[0] != 3):
                raise KeyError(f"The input of {name} must have "
                               "3 elements if it is 1D array.")
        return result

    def decorator(func):
        code = func.__code__
        argnames = code.co_varnames
        ndefaults = len(func.__defaults__) if func.__defaults__ else 0
        nposargs = code.co_argcount - ndefaults
        posargnames = argnames[:nposargs]
        for name in input_names:
            if name not in argnames:
                raise ValueError("In decorator convert_to_numpy(): Name '{}' "
                                 "doesn't correspond to any positional "
                                 "argument of the decorated function {}()."
                                 "".format(name, func.__name__))

        @wraps(func)
        def wrapper(*args, **kwargs):
            args = list(args)
            for i, name in enumerate(posargnames):
                if name in input_names:
                    args[i] = f(args[i], name=name,
                                i=input_names.index(name))
            for k in kwargs.keys():
                if name in input_names:
                    kwargs[k] = f(kwargs[k], name=name,
                                  i=input_names.index(name))
            return func(*args, **kwargs)
        return wrapper
    return decorator


################################################################
#                       装饰器定义结束
################################################################


@jax.jit
def pbc2pbc(pbc):
    result = jnp.zeros(3, dtype=bool)
    return result.at[:].set(pbc)


@jax.jit
@convert_to_jax('f', 'cell', check_shape=['coords', 'cell'])
def frc2car(f, cell):
    """Calculate Cartesian positions from scaled positions."""
    return f @ cell


@jax.jit
@convert_to_jax('v', 'cell', check_shape=['coords', 'cell'])
def car2frc(v, cell):
    """Calculate scaled positions from Cartesian positions."""
    return jnp.linalg.solve(cell.T, v.T).T


@jax.jit
@convert_to_jax('v', check_shape='coords')
def fn_mic_nopbc(v, cell):
    return v


@jax.jit
@convert_to_jax('v', 'cell', check_shape=['coords', 'cell'])
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
@convert_to_jax('v', 'cell', check_shape=['coords', 'cell'])
def wrap_positions(v, cell, pbc=True):
    frac = car2frc(v, cell)
    frac_wrapped = jnp.where(
        pbc2pbc(pbc), frac % 1.0, frac)
    return frac_wrapped @ cell


@jax.jit
@convert_to_jax('v', 'cell', 'hkls', 'r_op',
                check_shape=['coords', 'cell',
                             'coords', 'cell'])
def fn_mic_general(v, cell, pbc, hkls, r_op):
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
