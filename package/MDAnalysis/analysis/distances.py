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


def group_coordinates_by_spectralclustering(coordinates, n_groups=2,
                                            cutoff=1e2, box=None,
                                            return_predictor=False):
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
    coordinates = coordinates.astype(np.float64)
    n_coordinates = len(coordinates)
    indices = np.arange(n_coordinates)
    if n_groups == 1:
        if return_predictor:
            return ([indices], None)
        return indices

    dist_mat = np.zeros((n_coordinates, n_coordinates))
    dist_mat[:] = cutoff
    pairs, distances = capped_distance(coordinates, coordinates, cutoff,
                                       box=box, return_distances=True)
    pi, pj = tuple(pairs.T)
    dist_mat[pi, pj] = dist_mat[pj, pi] = distances

    try:
        sc = skc.SpectralClustering(n_clusters=n_groups,
                                    affinity='precomputed_nearest_neighbors')
    except ValueError as exc:
        if "Unknown kernel" in str(exc):
            raise ValueError("'precomputed_nearest_neighbors' not "
                             "recognised as a kernel. Try upgrading "
                             "scikit-learn >= 0.23.1")
        raise exc

    clusters = sc.fit_predict(dist_mat)
    ix = np.argsort(clusters)
    groups = np.split(indices[ix], np.where(np.ediff1d(clusters[ix]))[0]+1)
    groups = [np.sort(x) for x in groups]
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
    dist_ = distance_array(coordinates, centers, box=box)
    indices = np.arange(len(coordinates))
    cluster_i = np.argmin(dist, axis=1)
    ix = np.argsort(cluster_i)
    groups = np.split(indices[ix], np.where(np.ediff1d(cluster_i[ix]))[0]+1)
    groups = [np.sort(x) for x in groups]
    if return_predictor:
        return (groups, dist_)
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


def group_coordinates_by_normal(coordinates, orientations,
                                n_groups=2, cutoff=15.0, box=None,
                                **kwargs):
    pairs, dists = capped_distance(coordinates, coordinates,
                                   cutoff, box=box, return_distances=True)
    vectors = np.asarray(orientations)
    mags = np.einsum('ij,ij,kl,kl->ik', vectors, vectors, vectors, vectors)
    angles = np.dot(vectors, vectors.T) / (mags ** 0.5)

    ix = np.sort(pairs[:, 0])
    pairs = pairs[ix]
    dists = dists[ix]
    splix = np.where(np.ediff1d(pairs[:, 0]))[0] + 1
    plist = np.split(pairs)
    dlist = np.split(dists)

    for p, d in zip(plist, dlist):
        if len(p) == 1:  # self to self
            continue
        i = p[0, 0]
        js = p[:, 1]
        similar = angles[i, js] < 0
        coords = coordinates[js][similar]
        origin = coordinates[i][similar]
        


def group_vectors_by_orientation(coordinates, orientations,
                                 return_predictor=False, n_groups=2,
                                 cutoff=15.0, box=None, **kwargs):
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

    # first pass: merge similar groups
    for p, d in zip(plist, dlist):
    # for p, d in zip(plist, dlist):
        i = p[0, 0]
        js = p[:, 1]
        similar = angles[i, js] > 0.2
        dist_ix = np.argsort(d[similar])[:20]
        ijs_ = np.unique(p[similar][dist_ix])
        common = np.bincount(groups_[ijs_]).argmax()
        groups_[ijs_] = common


    # second pass: clean up
    ids, counts = np.unique(groups_, return_counts=True)
    
    n_squash = len(ids) - n_groups
    indices = [np.where(groups_ == i)[0] for i in ids[np.argsort(counts)]]

    while n_squash != 0:
        if n_squash > 0:
            keep = indices[n_squash:]
            ditch = indices[:n_squash]
            groups_ = [list(x) for x in keep]

            mean_cog = np.array([coordinates[x].mean(axis=0) for x in keep])
            mean_orients = np.array([vectors[x].mean(axis=0) for x in keep])

            # extra = np.concatenate(ditch)
            # extra_vec = vectors[extra]
            # extra_mag = np.einsum('ij,ij,kl,kl->ik', extra_vec, extra_vec,
            #                     mean_orients, mean_orients, optimize='optimal')
            # extra_angles = np.dot(extra_vec, mean_orients.T) / extra_mag

            # extra_dots = np.einsum('ij,ij->i', extra_vec, extra_vec)
            # for i, idx in enumerate(extra):
            #     _vec = vectors[idx]
            #     js = plist[idx][:, 1]
            #     friends = [np.intersect1d(js, x, assume_unique=True) for x in keep]
            #     f_orients = np.array([vectors[x].mean(axis=0) for x in friends])
            #     extra_mags = np.einsum('ij,ij->i', f_orients, f_orients) * extra_dots[i]
            #     extra_angles = np.dot(_vec, f_orients.T) / (extra_mags ** 0.5)
            #     most = np.argmax([len(x) for x in friends])
            #     closest = np.argmax(extra_angles)
            #     if extra_angles[most] > 0 or closest == most:
            #         groups_[most].append(idx)  # acute and max neighbours
            #     else:
            #         options = [most, closest]
            #         dist_ = distance_array(_vec, mean_cog[options],
            #                                box=box)
            #         g = options[int(np.argmin(dist_) % 2 == 0)]
            #         groups_[g].append(idx)

            extra_vec = np.array([vectors[x].mean(axis=0) for x in ditch])
            extra_mag = np.einsum('ij,ij,kl,kl->ik', extra_vec, extra_vec,
                                mean_orients, mean_orients, optimize='optimal')
            extra_angles = np.dot(extra_vec, mean_orients.T) / extra_mag

            for i, ix_ in enumerate(ditch):
                js = np.unique(np.concatenate([plist[i_][:, 1] for i_ in ix_]))
                other = [np.intersect1d(js, x, assume_unique=True) for x in keep]
                most = np.argmax([len(x) for x in other])
                closest = np.argmax(extra_angles[i])
                if extra_angles[i, most] > 0 or closest == most:
                    groups_[most].extend(ix_)  # acute and max neighbours
                else:
                    extra_xyz = coordinates[ix_]
                    options = [most, closest]
                    dist_ = distance_array(extra_xyz, mean_cog[options],
                                           box=box)
                    g = options[int(np.argmin(dist_) % 2 == 0)]
                    groups_[g].extend(ix_)

            indices = [np.array(x) for x in groups_]

        else:
            # split leaflets out, largest first
            ix = indices[-1]
            n_ix_ = len(ix)
            vectors_ = vectors[ix]
            # first group by angle to first atom
            vdot = np.dot(vectors_[0], vectors_.T)
            vix = np.argsort(vdot)  # angle to first vector
            vdot = vdot[vix]
            ix_ = ix[vix]
            splix = np.where(vdot >= 0)[0][0]
            halves = np.split(ix_, [splix], axis=0)

            if len(halves[0]) and len(halves[1]):
                # re-group by anti/parallel
                mean_orient = np.array([vectors[x].mean(axis=0) for x in halves])
                mags = np.einsum('ij,ij,kl,kl->ik', vectors_, vectors_, 
                                mean_orient, mean_orient, optimize='optimal')
                angles_ = np.dot(vectors_, mean_orient.T) / (mags ** 0.5)
                args = np.argmax(angles_, axis=1)
                anti_groups = [ix_[np.where(args == i)[0]] for i in (0, 1)]
                indices = indices[:-1] + anti_groups

            else:
                # re-group by distance. This is SLOW
                coords_ = coordinates[ix]
                mean_orient_ = vectors[ix].mean(axis=0)
                cog = coords_.mean(axis=0)
                lower = cog - mean_orient_/2
                upper = cog + mean_orient_/2

                options = np.array([lower, upper])
                dist_ = distance_array(coords_, options, box=box)
                min_dist = np.argmin(dist_, axis=1)
                new_groups_ = [ix[np.where(min_dist == i)[0]] for i in (0, 1)]
                indices = new_groups_ + indices[:-1]


        # indices = sorted(indices, key=lambda x: len(x))
        n_squash = len(indices) - n_groups

    groups = sorted(indices, key=lambda x: len(x))

    if return_predictor:
        return (groups, angles)
    return groups
