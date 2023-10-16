import numba
import numpy as np

from scipy.spatial import cKDTree as SciKdtree
from numba_kdtree.kd_tree import KDTree as NumbaKdTree


def scipy_neighbor(data1, data2, r) -> np.ndarray:
    result = np.zeros([data2.shape[0], data1.shape[0]], dtype=bool)
    indices = SciKdtree(data1).query_ball_point(data2, r=r, workers=-1)
    for i, jlst in enumerate(indices):
        result[i, jlst] = True
    return result


@numba.njit(parallel=True)
def numba_neighbor_1(data1, data2, r) -> np.ndarray:
    result = np.full((data2.shape[0], data1.shape[0]), False, dtype=np.bool_)
    indices = NumbaKdTree(data1).query_radius(data2, r, workers=-1)
    for i in numba.prange(data2.shape[0]):
        result[i, indices[i]] = True
    return result


@numba.njit(parallel=False)
def numba_neighbor_2(data1, data2, r) -> np.ndarray:
    result = np.full((data2.shape[0], data1.shape[0]), False, dtype=np.bool_)
    indices = NumbaKdTree(data1).query_radius(data2, r, workers=-1)
    for i in numba.prange(data2.shape[0]):
        result[i, indices[i]] = True
    return result


if __name__ == "__main__":
    import os
    import pickle
    from ase.utils.timing import Timer
    timer = Timer()

    this_dir = os.path.dirname(__file__)
    with open(os.path.join(this_dir, "./nl_cache_Cu8000_rc12.pkl"), "rb") as f:
        data = pickle.load(f)
        POS = data['positions']
        KK_POS = data['kk_pos']

    numba_neighbor_1(POS, KK_POS[:2], 5)
    numba_neighbor_2(POS, KK_POS[:2], 5)

    print(f"RUN for {POS.shape[0]}x{KK_POS.shape[0]}")
    with timer("scipy_neighbor"):
        a1 = scipy_neighbor(POS, KK_POS, 5)
    with timer("numba_neighbor parallel"):
        a2 = numba_neighbor_1(POS, KK_POS, 5)
    with timer("numba_neighbor"):
        a3 = numba_neighbor_2(POS, KK_POS, 5)
    timer.write()

    assert np.all(a1 == a2)
