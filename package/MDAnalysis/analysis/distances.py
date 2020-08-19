# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
#


"""
Distance analysis --- :mod:`MDAnalysis.analysis.distances`
==========================================================

This module provides functions to rapidly compute distances between
atoms or groups of atoms.

:func:`dist` and :func:`between` can take atom groups that do not even
have to be from the same :class:`~MDAnalysis.core.universe.Universe`.

See Also
--------
:mod:`MDAnalysis.lib.distances`
"""

__all__ = ['distance_array', 'self_distance_array',
           'contact_matrix', 'dist', 'between']

import numpy as np
import scipy.sparse

from ..lib.distances import (capped_distance,
                             self_distance_array,
                             distance_array,  # legacy reasons
                             apply_PBC
                             )
from ..lib.c_distances import contact_matrix_no_pbc, contact_matrix_pbc
from ..lib.NeighborSearch import AtomNeighborSearch
from ..lib.distances import calc_bonds
from ..lib.mdamath import vector_of_best_fit


import warnings
import logging
logger = logging.getLogger("MDAnalysis.analysis.distances")


def contact_matrix(coord, cutoff=15.0, returntype="numpy", box=None):
    '''Calculates a matrix of contacts.

    There is a fast, high-memory-usage version for small systems
    (*returntype* = 'numpy'), and a slower, low-memory-usage version for
    larger systems (*returntype* = 'sparse').

    If *box* dimensions are passed then periodic boundary conditions
    are applied.

    Parameters
    ---------
    coord : array
       Array of coordinates of shape ``(N, 3)`` and dtype float32.
    cutoff : float, optional, default 15
       Particles within `cutoff` are considered to form a contact.
    returntype : string, optional, default "numpy"
       Select how the contact matrix is returned.
       * ``"numpy"``: return as an ``(N. N)`` :class:`numpy.ndarray`
       * ``"sparse"``: return as a :class:`scipy.sparse.lil_matrix`
    box : array-like or ``None``, optional, default ``None``
       Simulation cell dimensions in the form of
       :attr:`MDAnalysis.trajectory.base.Timestep.dimensions` when
       periodic boundary conditions should be taken into account for
       the calculation of contacts.

    Returns
    -------
    array or sparse matrix
       The contact matrix is returned in a format determined by the `returntype`
       keyword.

    See Also
    --------
    :mod:`MDAnalysis.analysis.contacts` for native contact analysis


    .. versionchanged:: 0.11.0
       Keyword *suppress_progmet* and *progress_meter_freq* were removed.
    '''

    if returntype == "numpy":
        adj = np.full((len(coord), len(coord)), False, dtype=bool)
        pairs = capped_distance(coord, coord, max_cutoff=cutoff,
                                box=box, return_distances=False)

        idx, idy = np.transpose(pairs)
        adj[idx, idy] = True

        return adj
    elif returntype == "sparse":
        # Initialize square List of Lists matrix of dimensions equal to number
        # of coordinates passed
        sparse_contacts = scipy.sparse.lil_matrix((len(coord), len(coord)),
                                                  dtype='bool')
        if box is not None:
            # with PBC
            contact_matrix_pbc(coord, sparse_contacts, box, cutoff)
        else:
            # without PBC
            contact_matrix_no_pbc(coord, sparse_contacts, cutoff)
        return sparse_contacts


def dist(A, B, offset=0, box=None):
    """Return distance between atoms in two atom groups.

    The distance is calculated atom-wise. The residue ids are also
    returned because a typical use case is to look at CA distances
    before and after an alignment. Using the `offset` keyword one can
    also add a constant offset to the resids which facilitates
    comparison with PDB numbering.

    Arguments
    ---------
    A, B : AtomGroup
       :class:`~MDAnalysis.core.groups.AtomGroup` with the
       same number of atoms
    offset : integer or tuple, optional, default 0
       An integer `offset` is added to *resids_A* and *resids_B* (see
       below) in order to produce PDB numbers.

       If `offset` is :class:`tuple` then ``offset[0]`` is added to
       *resids_A* and ``offset[1]`` to *resids_B*. Note that one can
       actually supply numpy arrays of the same length as the atom
       group so that an individual offset is added to each resid.

    Returns
    -------
    resids_A : array
        residue ids of the `A` group (possibly changed with `offset`)
    resids_B : array
       residue ids of the `B` group (possibly changed with `offset`)
    distances : array
       distances between the atoms
    """

    if A.atoms.n_atoms != B.atoms.n_atoms:
        raise ValueError("AtomGroups A and B do not have the "
                         "same number of atoms")
    try:
        off_A, off_B = offset
    except (TypeError, ValueError):
        off_A = off_B = int(offset)
    residues_A = np.array(A.resids) + off_A
    residues_B = np.array(B.resids) + off_B

    d = calc_bonds(A.positions, B.positions, box)
    return np.array([residues_A, residues_B, d])


def between(group, A, B, distance):
    """Return sub group of `group` that is within `distance` of both `A` and `B`

    This function is not aware of periodic boundary conditions.

    Can be used to find bridging waters or molecules in an interface.

    Similar to "*group* and (AROUND *A* *distance* and AROUND *B* *distance*)".

    Parameters
    ----------
    group : AtomGroup
        Find members of `group` that are between `A` and `B`
    A : AtomGroup
    B : AtomGroup
        `A` and `B` are the groups of atoms between which atoms in
        `group` are searched for.  The function works is more
        efficient if `group` is bigger than either `A` or `B`.
    distance : float
        maximum distance for an atom to be counted as in the vicinity of
        `A` or `B`

    Returns
    -------
    AtomGroup
        :class:`~MDAnalysis.core.groups.AtomGroup` of atoms that
        fulfill the criterion


    .. versionadded: 0.7.5

    """
    ns_group = AtomNeighborSearch(group)
    resA = set(ns_group.search(A, distance))
    resB = set(ns_group.search(B, distance))
    return sum(sorted(resB.intersection(resA)))


def group_coordinates_by_spectralclustering(coordinates, orientations,
                                            n_groups=2, delta=20,
                                            cutoff=1e2, box=None,
                                            return_predictor=False,
                                            angle_threshold=0.8,
                                            **kwargs):
    """Cluster coordinates into groups using spectral clustering

    If the optional argument `box` is supplied, the minimum image convention
    is applied when calculating distances.

    Parameters
    ----------
    coordinates: numpy.ndarray
        Coordinate array with shape ``(n, 3)``
    n_groups: int (optional)
        Number of resulting groups
    cutoff: float (optional)
        Cutoff for computing distances in the matrix used for clustering
    box: numpy.ndarray (optional)
        The unitcell dimensions of the system or ``None``. If ``None``,
        we do not use the minimum image convention.
    return_predictor: bool (optional)
        whether to return the cluster class
    **kwargs:
        ignored (available to provide a similar interface to other
        grouping functions)

    Returns
    -------
    indices: list of numpy.ndarray
        List of indices for each group, corresponding to the order
        of ``coordinates``. ``indices[i]`` is the array of indices
        for the i-th cluster. ``k = indices[i][j]`` means that the
        k-th entry in ``coordinates`` is in cluster ``i``.
        The groups are sorted by size.

    clusterer: :class:`sklearn.cluster.SpectralClustering` (optional)
        The object used to cluster the groups
    """
    try:
        import sklearn.cluster as skc
    except ImportError:
        raise ImportError('scikit-learn is required to use this method '
                          'but is not installed. Install it with `conda '
                          'install scikit-learn` or `pip install '
                          'scikit-learn`.') from None
    orientations = np.asarray(orientations)
    coordinates = np.asarray(coordinates)

    # first merge by similar neighbours
    norm = np.linalg.norm(orientations, axis=1)
    orientations /= norm.reshape(-1, 1)
    angles = np.dot(orientations, orientations.T) #/ np.outer(norm, norm.T)
    n_coordinates = len(coordinates)
    indices = np.arange(n_coordinates)

    dist_mat = np.zeros((n_coordinates, n_coordinates))
    dist_mat[:] = 2*cutoff
    pairs, distances = capped_distance(coordinates, coordinates, cutoff,
                                       box=box, return_distances=True)
    pi, pj = tuple(pairs.T)
    # dist_mat[pi, pj] = dist_mat[pj, pi] = distances

    # calculate normals
    splix = np.where(np.ediff1d(pairs[:, 0]))[0] + 1
    plist = np.split(pairs, splix)
    dlist = np.split(distances, splix)

    for p, d in zip(plist, dlist):
        i = p[0, 0]
        js_ = p[:, 1]
        i_coord = coordinates[i]
        neigh_ = coordinates[js_]
        vec = orientations[i]
        if box is not None:
            diff = neigh_ - i_coord
            move = np.where(np.abs(diff) > box[:3]/2)
            neigh_[move] -= box[move[1]] * np.sign(diff[move])
        neigh_ -= i_coord
        ang_ = [np.dot(x, vec)/np.linalg.norm(x) for x in neigh_]
        ang_ = np.nan_to_num(ang_, nan=1)
        proj = np.abs(d * ang_)
        dist_mat[i, js_] = dist_mat[js_, i] = proj + d


    if delta is None:
        delta = np.max(dist_mat[pi, pj]) / 3

    gau = np.exp(- dist_mat ** 2 / (2. * delta ** 2))
    # reasonably acute/obtuse angles are acute/obtuse anough
    cos = np.clip(angles, -angle_threshold, angle_threshold)
    cos += angle_threshold
    cos /= (2*angle_threshold)
    # cos = 1 / (1 + np.exp(-beta*cos))
    ker = gau * cos

    ker[np.diag_indices(n_coordinates)] = 1

    try:
        sc = skc.SpectralClustering(n_clusters=n_groups, affinity="precomputed")
    except ValueError as exc:
        if "Unknown kernel" in str(exc):
            raise ValueError("'precomputed_nearest_neighbors' not "
                             "recognised as a kernel. Try upgrading "
                             "scikit-learn >= 0.23.1")
        raise exc from None

    clusters = sc.fit_predict(ker)
    ix = np.argsort(clusters)
    groups = np.split(indices[ix], np.where(np.ediff1d(clusters[ix]))[0]+1)
    groups = [np.sort(x) for x in groups]
    groups.sort(key=lambda x: len(x), reverse=True)
    if return_predictor:
        return (groups, sc)
    return groups


def group_coordinates_by_cog(coordinates, centers=[], box=None,
                             return_predictor=False,
                             **kwargs):
    """Separate coordinates into groups by distance to points

    For example, separate a bilayer into leaflets depending on distance
    to center-of-geometry

    Parameters
    ----------
    coordinates: numpy.ndarray
        Coordinate array with shape ``(n, 3)``
    centers: array-like (optional)
        List or numpy array (shape ``(n, 3)``) of xyz coordinates
    box: numpy.ndarray (optional)
        The unitcell dimensions of the system or ``None``. If ``None``,
        we do not use the minimum image convention.
    return_predictor: bool (optional)
        whether to return the distance matrix used to cluster the groups
    **kwargs:
        ignored (available to provide a similar interface to other
        grouping functions)

    Returns
    -------
    indices: list of numpy.ndarray
        List of indices for each group, corresponding to the order
        of ``coordinates``. ``indices[i]`` is the array of indices
        for the i-th cluster. ``k = indices[i][j]`` means that the
        k-th entry in ``coordinates`` is in cluster ``i``.
        The groups are *not* sorted by size, unlike other grouping
        metrics.

    distance_matrix: :class:`numpy.ndarray` (optional)
        Distance matrix of each coordinate to each center
    """
    coordinates = coordinates.astype(np.float64)
    centers = np.asarray(centers)
    dist_arr = distance_array(coordinates, centers, box=box)
    indices = np.arange(len(coordinates))
    cluster_i = np.argmin(dist_arr, axis=1)
    ix = np.argsort(cluster_i)
    groups = np.split(indices[ix], np.where(np.ediff1d(cluster_i[ix]))[0]+1)
    groups = [np.sort(x) for x in groups]
    if return_predictor:
        return (groups, dist_arr)
    return groups


def group_coordinates_by_graph(coordinates, cutoff=15.0, box=None,
                               sparse=None, return_predictor=False,
                               **kwargs):
    """Cluster coordinates into groups using an adjacency matrix

    If the optional argument `box` is supplied, the minimum image convention
    is applied when calculating distances.

    Parameters
    ----------
    coordinates: numpy.ndarray
        Coordinate array with shape ``(n, 3)``
    cutoff: float (optional)
        Cutoff distance for determining if two coordinates are in the same
        group
    box: numpy.ndarray (optional)
        The unitcell dimensions of the system or ``None``. If ``None``,
        we do not use the minimum image convention.
    return_predictor: bool (optional)
        whether to return the graph
    **kwargs:
        ignored (available to provide a similar interface to other
        grouping functions)

    Returns
    -------
    indices: list of numpy.ndarray
        List of indices for each group, corresponding to the order
        of ``coordinates``. ``indices[i]`` is the array of indices
        for the i-th cluster. ``k = indices[i][j]`` means that the
        k-th entry in ``coordinates`` is in cluster ``i``.
        The groups are sorted by size.

    graph: :class:`networkx.Graph` (optional)
        Graph used to group the coordinates
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required to use this method "
                          "but is not installed. Install it with "
                          "`conda install networkx` or "
                          "`pip install networkx`.") from None
    returntype = "numpy" if not sparse else "sparse"
    coordinates = np.asarray(coordinates).astype(np.float32)
    try:
        adj = contact_matrix(coordinates, cutoff=cutoff, box=box,
                             returntype=returntype)
    except ValueError as exc:
        if sparse is None:
            warnings.warn("NxN matrix is too big. Switching to sparse matrix "
                          "method")
            adj = contact_matrix(coordinates, cutoff=cutoff, box=box,
                                 returntype="sparse")
        elif sparse is False:
            raise ValueError("NxN matrix is too big. "
                             "Use `sparse=True`") from None
        else:
            raise exc

    graph = nx.Graph(adj)
    groups = [np.sort(list(c)) for c in nx.connected_components(graph)]
    if return_predictor:
        return (groups, graph)
    else:
        return groups


def group_vectors_by_orientation(coordinates, orientations,
                                 return_predictor=False, n_groups=2,
                                 cutoff=30.0, box=None,
                                 min_group=1,
                                 return_ungrouped=False,
                                 max_neighbors=15,
                                 angle_threshold=0.5,
                                 angle_relax=0.2, **kwargs):
    """Group vectors into 2 groups by angles to first vector

    If the optional argument `box` is supplied, the minimum image convention
    is applied when calculating distances.

    Parameters
    ----------
    vectors: numpy.ndarray
        Coordinate array with shape ``(n, 3)``
    n_groups: int (optional)
        Number of resulting groups. It is not tested n_groups != 2.
    return_predictor: bool (optional)
        whether to return the angles
    **kwargs:
        ignored (available to provide a similar interface to other
        grouping functions)

    Returns
    -------
    indices: list of numpy.ndarray
        List of indices for each group, corresponding to the order
        of ``coordinates``. ``indices[i]`` is the array of indices
        for the i-th cluster. ``k = indices[i][j]`` means that the
        k-th entry in ``coordinates`` is in cluster ``i``.
        The groups are sorted by size.

    angles: :class:`numpy.ndarray` (optional)
        Array of angles to the first vector
    """
    vectors = np.asarray(orientations)
    coordinates = np.asarray(coordinates)

    # first merge by similar neighbours
    norm = np.linalg.norm(vectors, axis=1)
    angles = np.dot(vectors, vectors.T) / np.outer(norm, norm.T)
    pairs, dists = capped_distance(coordinates, coordinates,
                                   cutoff, box=box, return_distances=True)
    splix = np.where(np.ediff1d(pairs[:, 0]))[0] + 1
    plist = np.split(pairs, splix)
    dlist = np.split(dists, splix)
    groups_ = np.arange(len(orientations))

    dix = [np.argsort(x) for x in dlist]
    lengths = [len(x) for x in dix]
    dlist = [d[x] for d, x in zip(dlist, dix)]
    plist = [p[x] for p, x in zip(plist, dix)]
    n_neighbors = int(max_neighbors/2) + 1

    # first pass: merge similar groups
    for p in plist:
        i = p[0, 0]
        js = p[:, 1][:max_neighbors*2]
        similar = angles[i, js] > angle_threshold
        ijs_ = js[similar][:max_neighbors]
        common = np.bincount(groups_[ijs_]).argmax()
        groups_[ijs_[:n_neighbors]] = common

    ids, counts = np.unique(groups_, return_counts=True)

    n_squash = len(ids) - n_groups
    # print("n_squash:", n_squash)

    for p in plist:
        i = p[0, 0]
        js = p[:, 1]
        similar = angles[i, js] > angle_threshold #(angle_threshold-angle_relax)
        ijs_ = js[similar][:n_neighbors]
        common = np.bincount(groups_[ijs_]).argmax()
        groups_[ijs_[:n_neighbors]] = common

    # second pass: clean up
    ids, counts = np.unique(groups_, return_counts=True)

    n_squash = len(ids) - n_groups
    # print("n_squash:", n_squash)
    indices = [np.where(groups_ == i)[0] for i in ids[np.argsort(counts)]]
    ignored = []

    if n_squash > 0:
        keep = indices[n_squash:]
        ditch = indices[:n_squash]
        groups_ = [list(x) for x in keep]

        mean_cog = np.array([coordinates[x].mean(axis=0) for x in keep])
        mean_orients = np.array([vectors[x].mean(axis=0) for x in keep])
        std_orients = np.array([np.std(vectors[x], axis=0) for x in keep])
        unsquashed = []



        while len(indices) > 2:
            if len(indices) + len(unsquashed) == n_groups:
                break
            current_gp = indices.pop(0)
            p_neighbors = np.concatenate([plist[x] for x in current_gp])
            d_neighbors = np.concatenate([dlist[x] for x in current_gp])
            a_neighbors = np.array([angles[x[0], x[1]]
                                    for x in p_neighbors])
            not_self = np.array(
                [x not in current_gp for x in p_neighbors[:, 1]])
            similar = a_neighbors > (angle_threshold-angle_relax)
            p_neighbors = p_neighbors[not_self & similar]
            d_neighbors = d_neighbors[not_self & similar]
            a_neighbors = a_neighbors[not_self & similar]
            dix_ = np.argsort(d_neighbors)
            p_neighbors = p_neighbors[dix_]
            d_neighbors = d_neighbors[dix_]
            a_neighbors = a_neighbors[dix_]
            ids, u_ix = np.unique(p_neighbors[:, 1], return_index=True)
            ud_ix = np.sort(u_ix)#[:n_neighbors]
            # print(d_neighbors[ud_ix])

            friends, neighbors = p_neighbors[ud_ix].T
            inter_angles = a_neighbors[ud_ix]

            members, comm1 = [], []
            for x in indices:
                m, c, _ = np.intersect1d(neighbors, x, assume_unique=True,
                                         return_indices=True)
                # print(m, c)
                c = np.sort(c)#[:n_neighbors]
                members.append(neighbors[c])
                # members.append(m)
                comm1.append(c)

            n_members = [len(x) for x in members]
            # print([inter_angles[x] if len(x) else 0 for x in comm1])
            a_members = [inter_angles[x][:n_neighbors].mean() if len(x)
                         else 0 for x in comm1]
            # print(len(current_gp), a_members, n_members)
            # a_std = [inter_angles[x].std() for x in comm1]

            if any(n_members):
                _most = np.argmax(n_members)
                _closest = np.argmax(a_members)

                
                _mean = a_members[_most]
                # _std = a_std[_most]
                if n_members[_closest] == np.max(n_members): #_mean > (angle_threshold-angle_relax):
                    indices[_most] = np.concatenate([indices[_most],
                                                     current_gp])
                    indices.sort(key=lambda x: len(x))
                    continue

            if len(current_gp) < min_group:
                ignored.append(current_gp)
            else:
                # print("unsquashed")
                unsquashed.append(current_gp)

        indices = unsquashed + indices
        n_squash = len(indices) - n_groups
        indices = sorted(indices, key=lambda x: len(x))

        if n_squash > 0:
            # just throw it into the nearest one...
            keep = indices[n_squash:]
            ditch = indices[:n_squash]

            coord_keep = [coordinates[x] for x in keep]
            coord_ditch = [coordinates[x] for x in ditch]

            # cog_keep = np.array([coordinates[x].mean(axis=0)
            #                      for x in keep])
            # cog_ditch = np.array([coordinates[x].mean(axis=0)
            #                       for x in ditch])
            _min = []

            for cd in coord_ditch:
                _dists = []
                for ck in coord_keep:
                    _da = distance_array(cd, ck, box=box)
                    _dists.append(np.min(_da))
                _min.append(np.argmin(_dists))

            _min = np.array(_min)

            # _dist = distance_array(cog_ditch, cog_keep, box=box)
            # _min = np.argmin(_dist, axis=1)
            for i in range(n_groups):
                found = np.where(_min == i)[0]
                extra = [ditch[f] for f in found]
                keep[i] = np.concatenate([keep[i]] + extra)

            indices = keep

        n_squash = len(indices) - n_groups

    while n_squash < 0:
        # split leaflets out, largest first
        ix = indices[-1]
        n_ix_ = len(ix)
        vectors_ = vectors[ix]
        # first group by angle to first atom
        vdot = np.dot(vectors_[0], vectors_.T)
        vix = np.argsort(vdot)  # angle to first vector
        vdot = vdot[vix]
        ix_ = ix[vix]
        vectors_ = vectors_[vix]
        splix = np.where(vdot >= 0)[0][0]
        halves = np.split(ix_, [splix], axis=0)

        if len(halves[0]) and len(halves[1]):
            # re-group by anti/parallel
            mean_orient = [vectors[x].mean(axis=0) for x in halves]
            mean_orient = np.array(mean_orient)
            mags = np.einsum('ij,ij,kl,kl->ik', vectors_, vectors_,
                             mean_orient, mean_orient, optimize='optimal')
            angles_ = np.dot(vectors_, mean_orient.T) / (mags ** 0.5)
            args = np.argmax(angles_, axis=1)
            anti_groups = [ix_[np.where(args == i)[0]] for i in (0, 1)]
            indices = indices[:-1] + anti_groups

        else:
            # re-group by distance. This is SLOW
            coords_ = coordinates[ix]
            cog = coords_.mean(axis=0)
            mean_orient_ = vectors[ix].mean(axis=0)
            mean_orient_ /= np.linalg.norm(mean_orient_)
            mean_orient_ *= np.linalg.norm(cog)
            lower = cog - mean_orient_
            upper = cog + mean_orient_

            options = np.array([lower, upper])
            dist_ = distance_array(coords_, options, box=box)
            min_dist = np.argmin(dist_, axis=1)
            init_sep = [ix[np.where(min_dist == i)[0]] for i in (0, 1)]

            cogs = np.array([coordinates[x].mean(axis=0)
                             for x in init_sep])
            dist_ = distance_array(coords_, cogs, box=box)
            min_dist = np.argmin(dist_, axis=1)
            new_groups_ = [ix[np.where(min_dist == i)[0]] for i in (0, 1)]

            indices = new_groups_ + indices[:-1]

        # indices = sorted(indices, key=lambda x: len(x))
        n_squash = len(indices) - n_groups

    groups = sorted(indices, key=lambda x: len(x), reverse=True)

    if return_ungrouped:
        groups += ignored

    groups = [np.sort(x) for x in groups]

    if return_predictor:
        return (groups, angles)
    return groups
