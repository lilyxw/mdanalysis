# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
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


"""
Leaflet analysis --- :mod:`MDAnalysis.analysis.leaflet`
=======================================================

This module implements leaflet-based lipid analysis.

Finding leaflets
----------------

:class:`~MDAnalysis.analysis.leaflet.LeafletFinder` implements three
algorithms:

* the *LeafletFinder* algorithm, described in
  [Michaud-Agrawal2011]_. It can identify the lipids in a bilayer of
  arbitrary shape and topology, including planar and undulating bilayers
  under periodic boundary conditions or vesicles. It follows the
  algorithm below (for further details see [Michaud-Agrawal2011]_.)

    1. build a graph of all phosphate distances < cutoff
    2. identify the largest connected subgraphs
    3. analyse first and second largest graph, which correspond to the leaflets

* the spectral clustering algorithm. This clusters lipids by headgroups
  into the number of clusters specified by the user, according to
  pairwise distance.

* partitioning lipids by how close they are to a *known* center of
  geometry. This is *not recommended* unless the leaflets are planar
  and well-defined. It will not work well on vesicles.

* You can also pass in your own function, which must take an input array of
  coordinates and return a list of indices

One can use this information to identify

* the upper and lower leaflet of a *planar membrane* by comparing the
  the :meth:`~MDAnalysis.core.groups.AtomGroup.center_of_geometry` of
  the leaflet groups, or

* the outer and inner leaflet of a *vesicle* by comparing histograms
  of distances from the centre of geometry (or possibly simply the
  :meth:`~MDAnalysis.core.groups.AtomGroup.radius_of_gyration`).

See example scripts in the MDAnalysisCookbook_ on how to use
:class:`LeafletFinder`. The function :func:`optimize_cutoff` implements a
(slow) heuristic method to find the best cut off for the LeafletFinder
algorithm.

.. _MDAnalysisCookbook: https://github.com/MDAnalysis/MDAnalysisCookbook/tree/master/examples


Classes and Functions
---------------------

.. autoclass:: LeafletFinder
   :members:

.. autofunction:: optimize_cutoff

"""

import numpy as np

from ..core import groups
from ..lib.mdamath import vector_of_best_fit, make_whole
from .. import selections, lib
from . import distances

from ..due import due, Doi

due.cite(Doi("10.1002/jcc.21787"),
         description="LeafletFinder 'graph' algorithm",
         path="MDAnalysis.analysis.leaflet.LeafletFinder")

del Doi


def lipid_orientation(headgroup, residue=None, unit=False, box=None):
    """Determine lipid orientation from headgroup.

    Parameters
    ----------
    headgroup: Atom or AtomGroup or numpy.ndarray
        Headgroup atom or atoms. The center of geometry is used
        as the origin.
    residue: Residue or AtomGroup (optional)
        Residue atoms. If not given, this is taken as the first residue
        of the headgroup atoms. Even if the residue is guessed correctly,
        it is more efficient to provide as an argument if known.
    unit: bool (optional)
        Whether to return the unit vector

    Returns
    -------
    vector: numpy.ndarray (3,)
        Vector of orientation
    """
    if isinstance(headgroup, groups.Atom):
        headgroup = groups.AtomGroup([headgroup])
    try:
        ra = residue.atoms
    except AttributeError:
        ra = headgroup.residues[0].atoms
    n_hg = len(headgroup)
    atoms = ra - headgroup
    all_xyz = np.concatenate([headgroup.positions, atoms.positions])
    if box is not None:  # slow.....
        diff = all_xyz - all_xyz[0]
        move = np.where(np.abs(diff) > box[:3] / 2)
        all_xyz[move] -= box[move[1]] * np.sign(diff[move])
    hg_xyz = all_xyz[:n_hg]
    at_xyz = all_xyz[n_hg:]
    cog_hg = hg_xyz.mean(axis=0)
    cog_atoms = at_xyz.mean(axis=0)
    vec = cog_atoms - cog_hg
    if unit:
        vec = vec / np.dot(vec, vec)
    return vec


class LeafletFinder(object):
    """Identify atoms in the same leaflet of a lipid bilayer.

    You can use a predefined method ("graph", "spectralclustering" or
    "center_of_geometry"). Alternatively, you can pass in your own function
    as a method. This *must* accept an array of coordinates as the first
    argument, and *must* return either a list of numpy arrays (the
    ``components`` attribute) or a tuple of (list of numpy arrays,
    predictor object). The numpy arrays should be arrays of indices of the
    input coordinates, such that ``k = components[i][j]`` means that the
    ``k``th coordinate belongs to the ``i-th`` leaflet.
    The class will also pass the following keyword arguments to your function:
    ``cutoff``, ``box``, ``return_predictor``.

    Parameters
    ----------
    universe : Universe or AtomGroup
        Atoms to apply the algorithm to
    select : str
        A :meth:`Universe.select_atoms` selection string
        for atoms that define the lipid head groups, e.g.
        universe.atoms.PO4 or "name PO4" or "name P*"
    cutoff : float (optional)
        cutoff distance for computing distances (for the spectral clustering
        method) or determining connectivity in the same leaflet (for the graph
        method). In spectral clustering, it just has to be suitably large to
        cover a significant part of the leaflet, but lower values increase
        computational efficiency. Please see the :func:`optimize_cutoff`
        function for help with values for the graph method. A cutoff is not
        used for the "center_of_geometry" method.
    pbc : bool (optional)
        If ``False``, does not follow the minimum image convention when
        computing distances
    method: str or function (optional)
        method to use to assign groups to leaflets. Choose
        "graph" for :func:`~distances.group_coordinates_by_graph`;
        "spectralclustering" for
        :func:`~distances.group_coordinates_by_spectralclustering`;
        "center_of_geometry" for
        :func:`~distances.group_coordinates_by_cog`;
        "orientation" to calculate orientations for each lipid and
        use :func:`~distances.group_vectors_by_orientation`
        or alternatively, pass in your own method. This *must* accept an
        array of coordinates as the first argument, and *must*
        return either a list of numpy arrays (the ``components``
        attribute) or a tuple of (list of numpy arrays, predictor object).
    calculate_orientations: bool (optional)
        if your custom method requires the orientation vector of each lipid,
        set ``calculate_orientations=True`` and an Nx3 array of orientation
        vectors will get passed into your function with the keyword
        ``orientation``. This is set to ``True`` for ``method="orientation"``.
    resort_cog: bool (optional)
        Whether to re-check leaflet membership by distance to
        center-of-geometry after assigning.
    **kwargs:
        Passed to ``method``


    Attributes
    ----------
    universe: Universe
    select: str
        Selection string
    selection: AtomGroup
        Atoms that the analysis is applied to
    headgroups: List of AtomGroup
        Atoms that the analysis is applied to, grouped by residue.
    pbc: bool
        Whether to use PBC or not
    box: numpy.ndarray or None
        Cell dimensions to use in calculating distances
    predictor:
        The object used to group the leaflets. :class:`networkx.Graph` for
        ``method="graph"``; :class:`sklearn.cluster.SpectralClustering` for
        ``method="spectralclustering"``; or :class:`numpy.ndarray` for
        ``method="center_of_geometry"``.
    positions: numpy.ndarray (N x 3)
        Array of positions of headgroups to use. If your selection has
        multiple atoms for each residue, this is the center of geometry.
    orientations: numpy.ndarray (N x 3) or None
        Array of orientation vectors calculated with ``lipid_orientation``.
    components: list of numpy.ndarray
        List of indices of atoms in each leaflet, corresponding to the
        order of `selection`. ``components[i]`` is the array of indices
        for the ``i``-th leaflet. ``k = components[i][j]`` means that the
        ``k``-th atom in `selection` is in the ``i``-th leaflet.
        The components are sorted by size for the "spectralclustering" and
        "graph" methods. For the "center_of_geometry" method, they are
        sorted by the order that the centers are passed into the class.
    groups: list of AtomGroups
        List of AtomGroups in each leaflet. ``groups[i]`` is the ``i``-th
        leaflet. The components are sorted by size for the "spectralclustering"
        and "graph" methods. For the "center_of_geometry" method, they are
        sorted by the order that the centers are passed into the class.
    leaflets: list of AtomGroup
        List of AtomGroups in each leaflet. ``groups[i]`` is the ``i``-th
        leaflet. The leaflets are sorted by z-coordinate so that the
        upper-most leaflet is first.
    sizes: list of ints
        List of the size of each leaflet in ``groups``.


    Example
    -------
    The components of the graph are stored in the list
    :attr:`LeafletFinder.components`; the atoms in each component are numbered
    consecutively, starting at 0. To obtain the atoms in the input structure
    use :attr:`LeafletFinder.groups`::

       L = LeafletFinder(PDB, 'name P*')
       leaflet_1 = L.groups[0]
       leaflet_2 = L.groups[1]

    The residues can be accessed through the standard MDAnalysis mechanism::

       leaflet_1.residues

    provides a :class:`~MDAnalysis.core.groups.ResidueGroup`
    instance. Similarly, all atoms in the first leaflet are then ::

       leaflet_1.residues.atoms


    See also
    --------
    :func:`~MDAnalysis.analysis.distances.group_coordinates_by_graph`
    :func:`~MDAnalysis.analysis.distances.group_coordinates_by_spectralclustering`


    .. versionchanged:: 2.0.0
        Refactored to move grouping code into ``distances`` and use
        multiple methods. Added the "spectralclustering" and
        "center_of_geometry" methods.

    .. versionchanged:: 1.0.0
       Changed `selection` keyword to `select`
    """

    @staticmethod
    def _unwrap(coord_list: list, box: np.ndarray,
                centers: np.ndarray = None):
        if centers is None:
            centers = [x[0] for x in coord_list]

        xyz = np.concatenate(coord_list)
        n_xyz = np.cumsum([len(x) for x in coord_list[:-1]])
        diff = np.concatenate([x - c for x, c in zip(coord_list, centers)])
        move = np.where(np.abs(diff) > box[:3]/2)
        xyz[move] -= box[move[1]] * np.sign(diff[move])
        return np.split(xyz, n_xyz)

    def _get_positions_by_residue(self, selection, centers=None):
        if self.box is None:
            return selection.center(None, compound='residues', pbc=self.pbc)
        sel = [x.positions.copy() for x in selection.split('residue')]
        uw = self._unwrap(sel, box=self.box, centers=centers)
        return np.array([x.mean(axis=0) for x in uw])

    def _guess_cutoff(self, cutoff, min_cutoff=15):
        # get box
        box = self.universe.dimensions
        if box is None:
            xyz = self.selection.positions
            box = box.max(axis=0) - box.min(axis=0)
        else:
            box = box[:3]

        per_leaflet = self.n_residues / self.n_groups
        spread = box / (per_leaflet ** 0.5)

        if self.method == "graph":
            # not too low
            guessed = 2 * min(spread)
            return max(guessed, min_cutoff)

        if cutoff == "guess":
            guessed = 3 * max(spread)
            return max(guessed, min_cutoff)

        dist = np.linalg.norm(box)
        guessed = dist / 2
        return max(guessed, min_cutoff)

    def __init__(self, universe, select='all', cutoff="guess", pbc=True,
                 method="graph", n_groups=2,
                 calculate_orientations=False, **kwargs):
        self.universe = universe.universe
        self.select = select
        self.selection = universe.atoms.select_atoms(select, periodic=pbc)
        self.headgroups = self.selection.split('residue')
        self.residues = self.selection.residues
        self.n_residues = len(self.residues)
        self.n_groups = n_groups
        self.pbc = pbc
        self.box = self.universe.dimensions if pbc else None
        self.positions = self._get_positions_by_residue(self.selection)

        if isinstance(method, str):
            method = method.lower().replace('_', '')
            self.method = method
        if method == "graph":
            self._method = distances.group_coordinates_by_graph
        elif method == "spectralclustering":
            self._method = distances.group_coordinates_by_spectralclustering
        elif method == "centerofgeometry":
            self._method = distances.group_coordinates_by_cog
        elif method == "orientation":
            self._method = distances.group_vectors_by_orientation
            calculate_orientations = True
        else:
            self._method = self.method = method

        if cutoff == "guess" or cutoff < 0:
            cutoff = self._guess_cutoff(cutoff)
        elif not isinstance(cutoff, (int, float)):
            raise ValueError("cutoff must be an int, float, or 'guess'. "
                             f"Given: {cutoff}")
        self.cutoff = cutoff

        self.orientations = None
        if calculate_orientations:
            # faster than doing lipid_orientations over and over
            not_hg = self.residues.atoms - self.selection
            cog_other = self._get_positions_by_residue(not_hg,
                                                       centers=self.positions)
            orients = cog_other - self.positions
            self.orientations = orients

        results = self._method(self.positions,
                               cutoff=self.cutoff, box=self.box,
                               return_predictor=True,
                               orientations=self.orientations,
                               n_groups=n_groups, **kwargs)

        if isinstance(results, tuple):
            self.components, self.predictor = results
        else:
            self.components = results
            self.predictor = None

        if len(self.headgroups) == len(self.selection):
            self.groups = [self.selection[x] for x in self.components]
        else:
            self.groups = [sum(self.headgroups[y] for y in x)
                           for x in self.components]

        self.sizes = [len(ag) for ag in self.groups]

        self.leaflets = sorted(self.groups,
                               key=lambda x: x.center_of_geometry()[-1],
                               reverse=True)

    def groups_iter(self):
        """Iterator over all leaflet :meth:`groups`"""
        for group in self.groups:
            yield group

    def write_selection(self, filename, mode="w", format=None, **kwargs):
        """Write selections for the leaflets to *filename*.

        The format is typically determined by the extension of *filename*
        (e.g. "vmd", "pml", or "ndx" for VMD, PyMol, or Gromacs).

        See :class:`MDAnalysis.selections.base.SelectionWriter` for all
        options.
        """
        sw = selections.get_writer(filename, format)
        with sw(filename, mode=mode,
                preamble=f"Leaflets found by {repr(self)}\n",
                **kwargs) as writer:
            for i, ag in enumerate(self.groups, 1):
                writer.write(ag, name=f"leaflet_{i:d}")

    def __repr__(self):
        return (f"LeafletFinder(select='{self.select}', "
                f"cutoff={self.cutoff:.1f} Å, pbc={self.pbc})")


def optimize_cutoff(universe, select, dmin=10.0, dmax=20.0, step=0.5,
                    max_imbalance=0.2, **kwargs):
    r"""Find cutoff that minimizes number of disconnected groups.

    Applies heuristics to find best groups:

    1. at least two groups (assumes that there are at least 2 leaflets)
    2. reject any solutions for which:

       .. math::

              \frac{|N_0 - N_1|}{|N_0 + N_1|} > \mathrm{max_imbalance}

       with :math:`N_i` being the number of lipids in group
       :math:`i`. This heuristic picks groups with balanced numbers of
       lipids.

    Parameters
    ----------
    universe : Universe
        :class:`MDAnalysis.Universe` instance
    select : AtomGroup or str
        AtomGroup or selection string as used for :class:`LeafletFinder`
    dmin : float (optional)
    dmax : float (optional)
    step : float (optional)
        scan cutoffs from `dmin` to `dmax` at stepsize `step` (in Angstroms)
    max_imbalance : float (optional)
        tuning parameter for the balancing heuristic [0.2]
    kwargs : other keyword arguments
        other arguments for  :class:`LeafletFinder`

    Returns
    -------
    (cutoff, N)
         optimum cutoff and number of groups found


    .. Note:: This function can die in various ways if really no
              appropriate number of groups can be found; it ought  to be
              made more robust.

    .. versionchanged:: 1.0.0
       Changed `selection` keyword to `select`
    """
    kwargs.pop('cutoff', None)  # not used, so we filter it
    _sizes = []
    for cutoff in np.arange(dmin, dmax, step):
        LF = LeafletFinder(universe, select, cutoff=cutoff, **kwargs)
        # heuristic:
        #  1) N > 1
        #  2) no imbalance between large groups:
        sizes = LF.sizes
        if len(sizes) < 2:
            continue
        n0 = float(sizes[0])  # sizes of two biggest groups ...
        n1 = float(sizes[1])  # ... assumed to be the leaflets
        imbalance = np.abs(n0 - n1) / (n0 + n1)
        # print "sizes: %(sizes)r; imbalance=%(imbalance)f" % vars()
        if imbalance > max_imbalance:
            continue
        _sizes.append((cutoff, len(LF.sizes)))
    results = np.rec.fromrecords(_sizes, names="cutoff,N")
    del _sizes
    results.sort(order=["N", "cutoff"])  # sort ascending by N, then cutoff
    return results[0]  # (cutoff,N) with N>1 and shortest cutoff
