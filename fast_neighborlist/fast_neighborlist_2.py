import numpy as np
from ase.data import atomic_numbers
from ase.geometry import complete_cell


def primitive_neighbor_list(pbc, cell, positions, cutoff):

    quantities = 'ijS'
    numbers=None
    self_interaction=False
    use_scaled_positions=False
    max_nbins=1e6

    # Naming conventions: Suffixes indicate the dimension of an array. The
    # following convention is used here:
    #     c: Cartesian index, can have values 0, 1, 2
    #     i: Global atom index, can have values 0..len(a)-1
    #     xyz: Bin index, three values identifying x-, y- and z-component of a
    #          spatial bin that is used to make neighbor search O(n)
    #     b: Linearized version of the 'xyz' bin index
    #     a: Bin-local atom index, i.e. index identifying an atom *within* a
    #        bin
    #     p: Pair index, can have value 0 or 1
    #     n: (Linear) neighbor index

    # Return empty neighbor list if no atoms are passed here
    if len(positions) == 0:
        empty_types = dict(i=(int, (0, )),
                           j=(int, (0, )),
                           D=(float, (0, 3)),
                           d=(float, (0, )),
                           S=(int, (0, 3)))
        retvals = []
        for i in quantities:
            dtype, shape = empty_types[i]
            retvals += [np.array([], dtype=dtype).reshape(shape)]
        if len(retvals) == 1:
            return retvals[0]
        else:
            return tuple(retvals)

    # Compute reciprocal lattice vectors.
    b1_c, b2_c, b3_c = np.linalg.pinv(cell).T

    # Compute distances of cell faces.
    l1 = np.linalg.norm(b1_c)
    l2 = np.linalg.norm(b2_c)
    l3 = np.linalg.norm(b3_c)
    face_dist_c = np.array([1 / l1 if l1 > 0 else 1,
                            1 / l2 if l2 > 0 else 1,
                            1 / l3 if l3 > 0 else 1])

    if isinstance(cutoff, dict):
        max_cutoff = max(cutoff.values())
    else:
        if np.isscalar(cutoff):
            max_cutoff = cutoff
        else:
            cutoff = np.asarray(cutoff)
            max_cutoff = 2*np.max(cutoff)

    # We use a minimum bin size of 3 A
    bin_size = max(max_cutoff, 3)
    # Compute number of bins such that a sphere of radius cutoff fits into
    # eight neighboring bins.
    nbins_c = np.maximum((face_dist_c / bin_size).astype(int), [1, 1, 1])
    nbins = np.prod(nbins_c)
    # Make sure we limit the amount of memory used by the explicit bins.
    while nbins > max_nbins:
        nbins_c = np.maximum(nbins_c // 2, [1, 1, 1])
        nbins = np.prod(nbins_c)

    # Compute over how many bins we need to loop in the neighbor list search.
    neigh_search_x, neigh_search_y, neigh_search_z = \
        np.ceil(bin_size * nbins_c / face_dist_c).astype(int)

    # If we only have a single bin and the system is not periodic, then we
    # do not need to search neighboring bins
    neigh_search_x = 0 if nbins_c[0] == 1 and not pbc[0] else neigh_search_x
    neigh_search_y = 0 if nbins_c[1] == 1 and not pbc[1] else neigh_search_y
    neigh_search_z = 0 if nbins_c[2] == 1 and not pbc[2] else neigh_search_z

    # Sort atoms into bins.
    if use_scaled_positions:
        scaled_positions_ic = positions
        positions = np.dot(scaled_positions_ic, cell)
    else:
        scaled_positions_ic = np.linalg.solve(complete_cell(cell).T,
                                              positions.T).T
    bin_index_ic = np.floor(scaled_positions_ic*nbins_c).astype(int)
    cell_shift_ic = np.zeros_like(bin_index_ic)

    for c in range(3):
        if pbc[c]:
            # (Note: np.divmod does not exist in older numpies)
            cell_shift_ic[:, c], bin_index_ic[:, c] = \
                divmod(bin_index_ic[:, c], nbins_c[c])
        else:
            bin_index_ic[:, c] = np.clip(bin_index_ic[:, c], 0, nbins_c[c]-1)

    # Convert Cartesian bin index to unique scalar bin index.
    bin_index_i = (bin_index_ic[:, 0] +
                   nbins_c[0] * (bin_index_ic[:, 1] +
                                 nbins_c[1] * bin_index_ic[:, 2]))

    # atom_i contains atom index in new sort order.
    atom_i = np.argsort(bin_index_i)
    bin_index_i = bin_index_i[atom_i]

    # Find max number of atoms per bin
    max_natoms_per_bin = np.bincount(bin_index_i).max()

    # Sort atoms into bins: atoms_in_bin_ba contains for each bin (identified
    # by its scalar bin index) a list of atoms inside that bin. This list is
    # homogeneous, i.e. has the same size *max_natoms_per_bin* for all bins.
    # The list is padded with -1 values.
    atoms_in_bin_ba = -np.ones([nbins, max_natoms_per_bin], dtype=int)
    for i in range(max_natoms_per_bin):
        # Create a mask array that identifies the first atom of each bin.
        mask = np.append([True], bin_index_i[:-1] != bin_index_i[1:])
        # Assign all first atoms.
        atoms_in_bin_ba[bin_index_i[mask], i] = atom_i[mask]

        # Remove atoms that we just sorted into atoms_in_bin_ba. The next
        # "first" atom will be the second and so on.
        mask = np.logical_not(mask)
        atom_i = atom_i[mask]
        bin_index_i = bin_index_i[mask]

    # Make sure that all atoms have been sorted into bins.
    assert len(atom_i) == 0
    assert len(bin_index_i) == 0

    # Now we construct neighbor pairs by pairing up all atoms within a bin or
    # between bin and neighboring bin. atom_pairs_pn is a helper buffer that
    # contains all potential pairs of atoms between two bins, i.e. it is a list
    # of length max_natoms_per_bin**2.
    atom_pairs_pn = np.indices((max_natoms_per_bin, max_natoms_per_bin),
                               dtype=int)
    atom_pairs_pn = atom_pairs_pn.reshape(2, -1)

    # Initialized empty neighbor list buffers.
    first_at_neightuple_nn = []
    secnd_at_neightuple_nn = []
    cell_shift_vector_x_n = []
    cell_shift_vector_y_n = []
    cell_shift_vector_z_n = []

    # This is the main neighbor list search. We loop over neighboring bins and
    # then construct all possible pairs of atoms between two bins, assuming
    # that each bin contains exactly max_natoms_per_bin atoms. We then throw
    # out pairs involving pad atoms with atom index -1 below.
    binz_xyz, biny_xyz, binx_xyz = np.meshgrid(np.arange(nbins_c[2]),
                                               np.arange(nbins_c[1]),
                                               np.arange(nbins_c[0]),
                                               indexing='ij')
    # The memory layout of binx_xyz, biny_xyz, binz_xyz is such that computing
    # the respective bin index leads to a linearly increasing consecutive list.
    # The following assert statement succeeds:
    #     b_b = (binx_xyz + nbins_c[0] * (biny_xyz + nbins_c[1] *
    #                                     binz_xyz)).ravel()
    #     assert (b_b == np.arange(np.prod(nbins_c))).all()

    # First atoms in pair.
    _first_at_neightuple_n = atoms_in_bin_ba[:, atom_pairs_pn[0]]
    for dz in range(-neigh_search_z, neigh_search_z+1):
        for dy in range(-neigh_search_y, neigh_search_y+1):
            for dx in range(-neigh_search_x, neigh_search_x+1):
                # Bin index of neighboring bin and shift vector.
                shiftx_xyz, neighbinx_xyz = divmod(binx_xyz + dx, nbins_c[0])
                shifty_xyz, neighbiny_xyz = divmod(biny_xyz + dy, nbins_c[1])
                shiftz_xyz, neighbinz_xyz = divmod(binz_xyz + dz, nbins_c[2])
                neighbin_b = (neighbinx_xyz + nbins_c[0] *
                              (neighbiny_xyz + nbins_c[1] * neighbinz_xyz)
                              ).ravel()

                # Second atom in pair.
                _secnd_at_neightuple_n = \
                    atoms_in_bin_ba[neighbin_b][:, atom_pairs_pn[1]]

                # Shift vectors.
                _cell_shift_vector_x_n = \
                    np.resize(shiftx_xyz.reshape(-1, 1),
                              (max_natoms_per_bin**2, shiftx_xyz.size)).T
                _cell_shift_vector_y_n = \
                    np.resize(shifty_xyz.reshape(-1, 1),
                              (max_natoms_per_bin**2, shifty_xyz.size)).T
                _cell_shift_vector_z_n = \
                    np.resize(shiftz_xyz.reshape(-1, 1),
                              (max_natoms_per_bin**2, shiftz_xyz.size)).T

                # We have created too many pairs because we assumed each bin
                # has exactly max_natoms_per_bin atoms. Remove all surperfluous
                # pairs. Those are pairs that involve an atom with index -1.
                mask = np.logical_and(_first_at_neightuple_n != -1,
                                      _secnd_at_neightuple_n != -1)
                if mask.sum() > 0:
                    first_at_neightuple_nn += [_first_at_neightuple_n[mask]]
                    secnd_at_neightuple_nn += [_secnd_at_neightuple_n[mask]]
                    cell_shift_vector_x_n += [_cell_shift_vector_x_n[mask]]
                    cell_shift_vector_y_n += [_cell_shift_vector_y_n[mask]]
                    cell_shift_vector_z_n += [_cell_shift_vector_z_n[mask]]

    # Flatten overall neighbor list.
    first_at_neightuple_n = np.concatenate(first_at_neightuple_nn)
    secnd_at_neightuple_n = np.concatenate(secnd_at_neightuple_nn)
    cell_shift_vector_n = np.transpose([np.concatenate(cell_shift_vector_x_n),
                                        np.concatenate(cell_shift_vector_y_n),
                                        np.concatenate(cell_shift_vector_z_n)])

    # Add global cell shift to shift vectors
    cell_shift_vector_n += cell_shift_ic[first_at_neightuple_n] - \
        cell_shift_ic[secnd_at_neightuple_n]

    # Remove all self-pairs that do not cross the cell boundary.
    if not self_interaction:
        m = np.logical_not(np.logical_and(
            first_at_neightuple_n == secnd_at_neightuple_n,
            (cell_shift_vector_n == 0).all(axis=1)))
        first_at_neightuple_n = first_at_neightuple_n[m]
        secnd_at_neightuple_n = secnd_at_neightuple_n[m]
        cell_shift_vector_n = cell_shift_vector_n[m]

    # For nonperiodic directions, remove any bonds that cross the domain
    # boundary.
    for c in range(3):
        if not pbc[c]:
            m = cell_shift_vector_n[:, c] == 0
            first_at_neightuple_n = first_at_neightuple_n[m]
            secnd_at_neightuple_n = secnd_at_neightuple_n[m]
            cell_shift_vector_n = cell_shift_vector_n[m]

    # Sort neighbor list.
    i = np.argsort(first_at_neightuple_n)
    first_at_neightuple_n = first_at_neightuple_n[i]
    secnd_at_neightuple_n = secnd_at_neightuple_n[i]
    cell_shift_vector_n = cell_shift_vector_n[i]

    # Compute distance vectors.
    distance_vector_nc = positions[secnd_at_neightuple_n] - \
        positions[first_at_neightuple_n] + \
        cell_shift_vector_n.dot(cell)
    abs_distance_vector_n = \
        np.sqrt(np.sum(distance_vector_nc*distance_vector_nc, axis=1))

    # We have still created too many pairs. Only keep those with distance
    # smaller than max_cutoff.
    mask = abs_distance_vector_n < max_cutoff
    first_at_neightuple_n = first_at_neightuple_n[mask]
    secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
    cell_shift_vector_n = cell_shift_vector_n[mask]
    distance_vector_nc = distance_vector_nc[mask]
    abs_distance_vector_n = abs_distance_vector_n[mask]

    if isinstance(cutoff, dict) and numbers is not None:
        # If cutoff is a dictionary, then the cutoff radii are specified per
        # element pair. We now have a list up to maximum cutoff.
        per_pair_cutoff_n = np.zeros_like(abs_distance_vector_n)
        for (atomic_number1, atomic_number2), c in cutoff.items():
            try:
                atomic_number1 = atomic_numbers[atomic_number1]
            except KeyError:
                pass
            try:
                atomic_number2 = atomic_numbers[atomic_number2]
            except KeyError:
                pass
            if atomic_number1 == atomic_number2:
                mask = np.logical_and(
                    numbers[first_at_neightuple_n] == atomic_number1,
                    numbers[secnd_at_neightuple_n] == atomic_number2)
            else:
                mask = np.logical_or(
                    np.logical_and(
                        numbers[first_at_neightuple_n] == atomic_number1,
                        numbers[secnd_at_neightuple_n] == atomic_number2),
                    np.logical_and(
                        numbers[first_at_neightuple_n] == atomic_number2,
                        numbers[secnd_at_neightuple_n] == atomic_number1))
            per_pair_cutoff_n[mask] = c
        mask = abs_distance_vector_n < per_pair_cutoff_n
        first_at_neightuple_n = first_at_neightuple_n[mask]
        secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
        cell_shift_vector_n = cell_shift_vector_n[mask]
        distance_vector_nc = distance_vector_nc[mask]
        abs_distance_vector_n = abs_distance_vector_n[mask]
    elif not np.isscalar(cutoff):
        # If cutoff is neither a dictionary nor a scalar, then we assume it is
        # a list or numpy array that contains atomic radii. Atoms are neighbors
        # if their radii overlap.
        mask = abs_distance_vector_n < \
            cutoff[first_at_neightuple_n] + cutoff[secnd_at_neightuple_n]
        first_at_neightuple_n = first_at_neightuple_n[mask]
        secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
        cell_shift_vector_n = cell_shift_vector_n[mask]
        distance_vector_nc = distance_vector_nc[mask]
        abs_distance_vector_n = abs_distance_vector_n[mask]

    # Assemble return tuple.
    retvals = []
    for q in quantities:
        if q == 'i':
            retvals += [first_at_neightuple_n]
        elif q == 'j':
            retvals += [secnd_at_neightuple_n]
        elif q == 'D':
            retvals += [distance_vector_nc]
        elif q == 'd':
            retvals += [abs_distance_vector_n]
        elif q == 'S':
            retvals += [cell_shift_vector_n]
        else:
            raise ValueError('Unsupported quantity specified.')
    if len(retvals) == 1:
        return retvals[0]
    else:
        return tuple(retvals)
