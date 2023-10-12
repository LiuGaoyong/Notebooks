import itertools

import numpy as np
from ase.cell import Cell
from ase.geometry import wrap_positions, minkowski_reduce
from ase.neighborlist import PrimitiveNeighborList
from scipy.spatial import cKDTree


class FastPrimitiveNeighborList(PrimitiveNeighborList):

    def build(self, pbc, cell, coordinates):
        """Build the list.

        Coordinates are taken to be scaled or not according
        to self.use_scaled_positions.
        """
        self.pbc = pbc = np.array(pbc, copy=True)
        self.cell = cell = Cell(cell)
        self.coordinates = coordinates = np.array(coordinates, copy=True)

        if len(self.cutoffs) != len(coordinates):
            raise ValueError('Wrong number of cutoff radii: {0} != {1}'
                             .format(len(self.cutoffs), len(coordinates)))

        if len(self.cutoffs) > 0:
            rcmax = self.cutoffs.max()
        else:
            rcmax = 0.0

        if self.use_scaled_positions:
            positions0 = cell.cartesian_positions(coordinates)
        else:
            positions0 = coordinates

        rcell, op = minkowski_reduce(cell, pbc)
        positions = wrap_positions(positions0, rcell, pbc=pbc, eps=0)

        N = []
        ircell = np.linalg.pinv(rcell)
        for i in range(3):
            if self.pbc[i]:
                v = ircell[:, i]
                h = 1 / np.linalg.norm(v)
                n = int(2 * rcmax / h) + 1
            else:
                n = 0
            N.append(n)

        tree = cKDTree(positions, copy_data=True)
        offsets = cell.scaled_positions(positions - positions0)
        offsets = offsets.round().astype(int)

        n123 = [(n1, n2, n3) for n1, n2, n3 in itertools.product(
            range(0, N[0] + 1),
            range(-N[1], N[1] + 1),
            range(-N[2], N[2] + 1)
        ) if not (n1 == 0 and (n2 < 0 or n2 == 0 and n3 < 0))]
        displacement = np.array(n123) @ rcell

        kk_pos = (positions[:, None] - displacement).reshape(-1, 3)
        kk_indices = tree.query_ball_point(kk_pos, r=2.0*rcmax, workers=-1)

        self._cache = {
            'positions': positions,
            'N': N,
            'n123': n123,
            'tree': tree,
            'kk_pos': kk_pos,
            'kk_indices': kk_indices
        }

        natoms = len(positions)
        self.nneighbors = 0
        self.npbcneighbors = 0
        self.neighbors = [np.empty(0, int) for a in range(natoms)]
        self.displacements = [np.empty((0, 3), int) for a in range(natoms)]
        self.nupdates += 1
        if natoms == 0:
            return

        for kk, indices in enumerate(kk_indices):
            a, k = kk % natoms, kk // natoms
            if not len(indices):
                continue

            indices = np.array(indices)
            delta = positions[indices] + displacement[k] - positions[a]
            cutoffs = self.cutoffs[indices] + self.cutoffs[a]
            i = indices[np.linalg.norm(delta, axis=1) < cutoffs]

            n1, n2, n3 = n123[k]
            if n1 == 0 and n2 == 0 and n3 == 0:
                if self.self_interaction:
                    i = i[i >= a]
                else:
                    i = i[i > a]

            self.nneighbors += len(i)
            self.neighbors[a] = np.concatenate((self.neighbors[a], i))
            disp = (n1, n2, n3) @ op + offsets[i] - offsets[a]
            self.npbcneighbors += disp.any(1).sum()
            self.displacements[a] = np.concatenate(
                (self.displacements[a], disp))

        if self.bothways:
            neighbors2 = [[] for a in range(natoms)]
            displacements2 = [[] for a in range(natoms)]
            for a in range(natoms):
                for b, disp in zip(self.neighbors[a], self.displacements[a]):
                    neighbors2[b].append(a)
                    displacements2[b].append(-disp)
            for a in range(natoms):
                nbs = np.concatenate((self.neighbors[a], neighbors2[a]))
                disp = np.array(
                    list(self.displacements[a]) + displacements2[a])
                # Force correct type and shape for case of no neighbors:
                self.neighbors[a] = nbs.astype(int)
                self.displacements[a] = disp.astype(int).reshape((-1, 3))

        if self.sorted:
            for a, i in enumerate(self.neighbors):
                mask = (i < a)
                if mask.any():
                    j = i[mask]
                    offsets = self.displacements[a][mask]
                    for b, offset in zip(j, offsets):
                        self.neighbors[b] = np.concatenate(
                            (self.neighbors[b], [a]))
                        self.displacements[b] = np.concatenate(
                            (self.displacements[b], [-offset]))
                    mask = np.logical_not(mask)
                    self.neighbors[a] = self.neighbors[a][mask]
                    self.displacements[a] = self.displacements[a][mask]
