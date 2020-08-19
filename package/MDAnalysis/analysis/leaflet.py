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


Calculating lipid enrichment
----------------------------

The lipid environment around a protein can be enriched or depleted in certain
lipid species, relative to the proportions of these species in the entire
membrane. One metric to quantify this is with a lipid depletion-enrichment
index, or DEI.

:class:`~MDAnalysis.analysis.leaflet.LipidEnrichment` implements a number of
options to calculate a DEI:

* the original method of [Corradi2018]_
  (``distribution='gaussian', buffer=0``).
  This uses a hard distance cut-off to determine membership in the
  lipid shell around the protein; fits the array of DEIs of each frame
  to a Gaussian distribution; and uses a two-tailed T-test to
  calculate p-values. This means that the mean and SD of the DEI is
  calculated directly from frame-by-frame DEI values. The mean is the
  arithmetic mean and the variance is the average of squared deviations
  from the mean.

* fitting a binomial distribution (``distribution='binomial'``).
  The number of lipids present around a protein more accurately follows
  a hypergeometric distribution with :math:`k` lipids of a species around
  the protein, :math:`K` specific lipids in the leaflet, :math:`n` total
  lipids around the protein, and :math:`N` total lipids in the leaflet:

  .. math::
    p = \frac{K}/{N}
    \text{Pr}(k; n, p) = \binom{n}{k} p^k (1-p)^{n-k}

  The convolution of the hypergeometric distributions in each frame is
  non-trivial, so instead this method approximates a trajectory of
  samples as a convolution of binomial distributions. This is itself
  a binomial:

  .. math::

    \sum^n_{i=1} B(n_i, p) ~ B(\sum^n{i=1} n_i, p) \ 0 < p < 1

  The mean and SD DEI for the overall trajectory is then calculated
  from the log-normal distribution that arises from the ratio of two
  binomial distributions :math:`T = X/Y`, where :math:`X` is the
  binomial distribution of lipids found around the protein, and
  :math:`Y` is the binomial distribution of the null hypothesis. The
  p-value is calculated from :math:`Y`.

  Edge cases are handled in the following ways:

  * :math:`p_{i}=0`: the lipid is totally depleted and never around
    the protein. Mean DEI = 0; SD = 0; Median = 0; p-value calculated
    as normal.
  * :math:`p_{0}=0`: there is none of this lipid species in the leaflet.
    In this case :math:`p_{i}=0` must also be true. Mean DEI = 1; SD = 0;
    Median = 1; p-value = 1
  * :math:`p_{0}=1`: the leaflet is comprised entirely of this lipid species.
    In this case :math:`p_{i}=1` must also be true. Mean DEI = 1; SD = 0;
    Median = 1; p-value = 1

* using a soft cutoff (``cutoff>0, buffer>0``).
  Using a hard-cutoff may result in extreme DEI values for lipid species
  with low concentration in the leaflet, or miss near-contact events,
  especially in short trajectories with sampling. Add a buffer to your
  ``cutoff`` to add fractional values for near-contact events.
  This follows a normal distribution, where the standard deviation is set
  such that 99.7% (3 sigma) of the probability distribution is contained
  within the buffer.

  In this case, we are no longer working with discrete events.
  If ``distribution='binomial'`` is selected, the class will use a continuous
  approximation of the hypergeometric distribution to compute frame-by-frame
  p-values (using the gamma function for factorials). 
  If ``distribution='gaussian'``, there is no change.

* using a soft potential (``cutoff=0, buffer>0``)
  You can choose to use a soft potential entirely by simply setting the
  hard cutoff region to 0. 


Classes and Functions
---------------------

.. autoclass:: LeafletFinder
   :members:

.. autofunction:: optimize_cutoff

.. autoclass:: LipidEnrichment
    :members:

"""
import warnings
import functools

import numpy as np
try:
    import scipy.stats
except ImportError:
    found_scipy = False
else:
    found_scipy = True

from .. import core, selections, _TOPOLOGY_ATTRNAMES, _TOPOLOGY_ATTRS
from . import distances
from .base import AnalysisBase
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

    def run(self):
        """
        Actually run the analysis.

        This is its own function to avoid repeating expensive setup in
        __init__ when it's called on every frame.
        """
        self.positions = self._get_positions_by_residue(self.selection)
        if self.calculate_orientations:
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
                               n_groups=self.n_groups, **self.kwargs)

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

        self.group_positions = [self.positions[x] for x in self.components]
        self.sizes = [len(ag) for ag in self.groups]

        z = [x.center_of_geometry()[-1] for x in self.groups]
        ix = np.argsort(z)
        self.leaflets = [self.groups[i] for i in ix[::-1]]
        self.leaflet_positions = [self.group_positions[i] for i in ix[::-1]]

        


    def __init__(self, universe, select='all', cutoff=None, pbc=True,
                 method="spectralclustering", n_groups=2,
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

        if isinstance(method, str):
            method = method.lower().replace('_', '')
            self.method = method
        if method == "graph":
            self._method = distances.group_coordinates_by_graph
        elif method == "spectralclustering":
            self._method = distances.group_coordinates_by_spectralclustering
            calculate_orientations = True
        elif method == "centerofgeometry":
            self._method = distances.group_coordinates_by_cog
        elif method == "orientation":
            self._method = distances.group_vectors_by_orientation
            calculate_orientations = True
        else:
            self._method = self.method = method

        if cutoff is None or cutoff < 0:
            cutoff = self._guess_cutoff(cutoff)
        elif not isinstance(cutoff, (int, float)):
            raise ValueError("cutoff must be an int, float, or 'guess'. "
                             f"Given: {cutoff}")
        self.cutoff = cutoff
        self.kwargs = kwargs
        self.orientations = None
        self.calculate_orientations = calculate_orientations
        self.run()

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


def binomial_gamma(top, bottom):
    try:
        from scipy.special import gamma
    except ImportError:
        raise ImportError("scipy is needed for this analysis but "
                          "is not installed. Please install with "
                          "`conda install scipy` or "
                          "`pip install scipy`") from None
    return gamma(top+1) / (gamma(bottom+1) * gamma(top-bottom+1))


class LipidEnrichment(AnalysisBase):
    r"""Calculate the lipid depletion-enrichment index around a protein
    by leaflet.

    The depletion-enrichment index (DEI) of a lipid around a protein
    indicates how enriched or depleted that lipid in the lipid annulus
    around the protein, with respect to the density of the lipid in the bulk
    membrane. If more than one leaflet is specified, then the index is
    calculated for each leaflet.

    The DEI for lipid species or group :math:`L` at cutoff :math:`r` is as
    follows:

    .. math::

        \text{DEI}_{L, r} = \frac{n(x_{(L, r)})}{n(x_r)} \times \frac{n(x)}{n(x_L)}

    where :math:`n(x_{(L, r)})` is the number of lipids :math:`L` within
    distance :math:`r` of the protein; :math:`n(x_r)` is total number of
    lipids within distance :math:`r` of the protein; :math:`n(x_L)`
    is the total number of lipids :math:`L` in that membrane or leaflet; and
    :math:`n(x)` is the total number of all lipids in that membrane or
    leaflet.

    The results of this analysis contain these values:

    * 'Near protein': the number of lipids :math:`L` within
        cutoff :math:`r` of the protein
    * 'Fraction near protein': the fraction of the lipids :math:`L`
        with respect to all lipids within cutoff :math:`r` of the
        protein: :math:`\frac{n(x_{(L, r)})}{n(x_r)}`
    * 'Enrichment': the depletion-enrichment index.

    This metric was obtained from [Corradi2018]_. Please cite them if you
    use this analysis in published work.

    There are

    .. note::

        This analysis requires ``scikit-learn`` to be installed to
        determine leaflets.


    Parameters
    ----------
    universe: Universe or AtomGroup
        The atoms to apply this analysis to.
    select_protein: str (optional)
        Selection string for the protein.
    select_residues: str (optional)
        Selection string for the group of residues / lipids to analyse.
    select_headgroup: str (optional)
        Selection string for the lipid headgroups. This is used to determine
        leaflets and compute distances, unless ``compute_headgroup_only`` is
        ``False``.
    n_leaflets: int (optional)
        How many leaflets to split the membrane into.
    count_by_attr: str (optional)
        How to split the lipid species. By default, this is `resnames` so
        each lipid species is computed individually. However, any topology
        attribute can be used. For example, you could add a new attribute
        describing tail saturation.
    cutoff: float (optional)
        Cutoff in ångström
    delta: float (optional)
        Leaflets are determined via spectral clustering on a similarity matrix.
        This similarity matrix is created from a pairwise distance matrix by
        applying the gaussian kernel
        :math:`\exp{\frac{-\text{dist_matrix}^2}{2*\delta^2}}`. If your
        leaflets are incorrect, try finetuning the `delta` value.
    compute_headgroup_only: bool (optional)
        By default this analysis only uses distance from the lipid headgroup
        to determine whether a lipid is within the `cutoff`
        distance from the protein. This choice was made to save computation.
        Compute the distance from every atom by toggling this option off.
    **kwargs
        Passed to :class:`~MDAnalysis.analysis.base.AnalysisBase`.


    Attributes
    ----------
    protein: :class:`~MDAnalysis.core.groups.AtomGroup`
        Protein atoms.
    residues: :class:`~MDAnalysis.core.groups.ResidueGroup`
        Lipid residues.
    headgroups: :class:`~MDAnalysis.core.groups.AtomGroup`
        Headgroup atoms.
    leaflet_residues: list of :class:`~MDAnalysis.core.groups.ResidueGroup`
        Residues per leaflet.
    leaflet_headgroups: list of :class:`~MDAnalysis.core.groups.AtomGroup`
        Headgroup atoms per leaflet.
    residue_counts: list of :class:`numpy.ndarray` (n_lipid_groups, n_frames)
        Number of each residue around protein for each frame, per leaflet.
    total_counts: list of :class:`numpy.ndarray` (n_frames,)
        Total number of residues around protein for each frame, per leaflet.
    leaflets: list of dict of dicts
        Counts, fraction, and enrichment index of each residue per frame, per
        leaflet.
    leaflets_summary: list of dict of dicts
        Counts, fraction, and enrichment index as averages and standard
        deviations, per leaflet.
    box: :class:`numpy.ndarray` or ``None``
        Universe dimensions.
    n_leaflets: int
        Number of leaflets
    delta: float
        delta used Gaussian kernel to transform pairwise distances into a
        similarity matrix for clustering.
    compute_headgroup_only: bool
        whether to compute distances only using the headgroup.
    attrname: str
        The topology attribute used to group lipid species.


    .. versionadded:: 2.0.0

    """

    def __init__(self, universe, select_protein: str = 'protein',
                 select_residues: str = 'all',
                 select_headgroup: str = 'name PO4',
                 n_leaflets: int = 2, count_by_attr: str = 'resnames',
                 cutoff: float = 6, pbc: bool = True,
                 compute_headgroup_only: bool = True,
                 update_leaflet_step: int = 1,
                 group_method: str = "spectralclustering",
                 group_kwargs: dict = {},
                 distribution: str = "binomial",
                 compute_p_value: bool = True,
                 buffer: float = 0, beta=4, **kwargs):
        super(LipidEnrichment, self).__init__(universe.universe.trajectory,
                                              **kwargs)
        if n_leaflets < 1:
            raise ValueError('Must have at least one leaflet')
        self.n_leaflets = n_leaflets

        self.group_method = group_method.lower()
        self.group_kwargs = group_kwargs
        self.update_leaflet_step = update_leaflet_step

        self.distribution = distribution.lower()
        if self.distribution == "binomial":
            self._fit_distribution = self._fit_binomial
        elif self.distribution == "gaussian":
            self._fit_distribution = self._fit_gaussian
        else:
            raise ValueError("`distribution` should be either "
                             "'binomial' or 'gaussian'")

        self.compute_p_value = compute_p_value

        if self.compute_p_value:  # look for scipy once
            if not found_scipy:
                raise ImportError("scipy is needed for this analysis but "
                                  "is not installed. Please install with "
                                  "`conda install scipy` or "
                                  "`pip install scipy`") from None

            if buffer:
                self._compute_p = self._compute_p_hypergeom_gamma
            elif self.distribution == "gaussian":
                self._compute_p = self._compute_p_gaussian
            else:
                self._compute_p = self._compute_p_hypergeom

        self.pbc = pbc
        self.box = universe.dimensions if pbc else None
        self.protein = universe.select_atoms(select_protein)
        ag = universe.select_atoms(select_residues)
        self.residues = ag.residues
        self.n_residues = len(ag.residues)
        self.headgroups = self.residues.atoms.select_atoms(select_headgroup)
        assert len(self.headgroups.residues) == self.n_residues

        if compute_headgroup_only:
            self._compute_atoms = self.headgroups
        else:
            self._compute_atoms = self.residues.atoms
        self._resindices = self._compute_atoms.resindices
        self.cutoff = cutoff
        self.buffer = buffer
        self.beta = beta
        self.leaflets = []
        self.leaflets_summary = []

        # process this a bit to remove errors like "alt_locs"
        attrname = _TOPOLOGY_ATTRNAMES[count_by_attr.lower().replace('_', '')]
        self.attrname = _TOPOLOGY_ATTRS[attrname].attrname

    def _prepare(self):
        # in case of change + re-run
        self.mid_buffer = self.buffer / 2.0
        self.max_cutoff = self.cutoff + self.buffer
        self._buffer_sigma = self.buffer / self.beta
        if self._buffer_sigma:
            self._buffer_coeff = 1 / (self._buffer_sigma * np.sqrt(2 * np.pi))
        self.ids = np.unique(getattr(self.residues, self.attrname))

        # results
        self.near_counts = np.zeros((self.n_leaflets, len(self.ids),
                                     self.n_frames))
        self.residue_counts = np.zeros((self.n_leaflets, len(self.ids),
                                        self.n_frames))
        self.total_counts = np.zeros((self.n_leaflets, self.n_frames))
        self.leaflet_residue_masks = np.zeros((self.n_frames, self.n_leaflets,
                                               self.n_residues), dtype=bool)
        self.leaflet_residues = np.zeros((self.n_frames, self.n_leaflets),
                                         dtype=object)
        # self._update_leaflets()

    def _update_leaflets(self):
        lf = LeafletFinder(self.headgroups, method=self.group_method,
                           n_groups=self.n_leaflets, **self.group_kwargs)
        self._current_leaflets = [l.residues for l in lf.leaflets[:self.n_leaflets]]
        lengths = [len(x) for x in self._current_leaflets]
        self._current_ids = [getattr(r, self.attrname)
                             for r in self._current_leaflets]

    def _single_frame(self):
        if not self._frame_index % self.update_leaflet_step:
            self._update_leaflets()

        # initial scoop for nearby groups
        coords_ = self._compute_atoms.positions
        pairs = distances.capped_distance(self.protein.positions,
                                          coords_,
                                          self.cutoff, box=self.box,
                                          return_distances=False)
        if pairs.size > 0:
            indices = np.unique(pairs[:, 1])
        else:
            indices = []

        # print("indices", indices)

        # now look for groups in the buffer
        if len(indices) and self.buffer:
            pairs2, dist = distances.capped_distance(self.protein.positions,
                                                     coords_, self.max_cutoff,
                                                     min_cutoff=self.cutoff,
                                                     box=self.box,
                                                     return_distances=True)
            
            # don't count things in inner cutoff
            mask = [x not in indices for x in pairs2[:, 1]]
            pairs2 = pairs2[mask]
            dist = dist[mask]

            if pairs2.size > 0:
                _ix = np.argsort(pairs2[:, 1])
                indices2 = pairs2[_ix][:, 1]
                dist = dist[_ix] - self.cutoff# - self.mid_buffer

                init_resix2 = self._resindices[indices2]
                # sort through for minimum distance
                ids2, splix = np.unique(init_resix2, return_index=True)
                resix2 = init_resix2[splix]
                split_dist = np.split(dist, splix[1:])
                min_dist = np.array([x.min() for x in split_dist])
                # print('min dist', len(min_dist), resix2)

                # logistic function
                for i, leaf in enumerate(self._current_leaflets):
                    ids = self._current_ids[i]
                    match, rix, lix = np.intersect1d(resix2, leaf.residues.resindices,
                                                     assume_unique=True,
                                                     return_indices=True)
                    subdist = min_dist[rix]
                    subids = ids[lix]
                    for j, x in enumerate(self.ids):
                        mask = (subids == x)
                        xdist = subdist[mask]
                        exp = -0.5 * ((xdist/self._buffer_sigma) ** 2)
                        n = self._buffer_coeff * np.exp(exp)
                        self.near_counts[i, j, self._frame_index] += n.sum()

        soft = self.near_counts[:, :, self._frame_index].sum()

        init_resix = self._resindices[indices]
        resix = np.unique(init_resix)
        for i, leaf in enumerate(self._current_leaflets):
            ids = self._current_ids[i]
            _, ix1, ix2 = np.intersect1d(resix, leaf.residues.resindices,
                                         assume_unique=True,
                                         return_indices=True)
            self.total_counts[i, self._frame_index] = len(ix1)
            subids = ids[ix2]
            for j, x in enumerate(self.ids):
                self.residue_counts[i, j, self._frame_index] += sum(ids == x)
                self.near_counts[i, j, self._frame_index] += sum(subids == x)

        both = self.near_counts[:, :, self._frame_index].sum()
        # print("cutoffs : ", soft, both, both-soft)


    def _fit_gaussian(self, data, *args, **kwargs):
        """Treat each frame as an independent observation in a gaussian
        distribution.

        Appears to be original method of [Corradi2018]_.

        .. note::

            The enrichment p-value is calculated from a two-tailed
            sample T-test, following [Corradi2018]_.

        """
        near = data['Near protein']
        frac = data['Fraction near protein']
        dei = data['Enrichment']
        summary = {
            'Mean near protein': near.mean(),
            'SD near protein': near.std(),
            'Mean fraction near protein': frac.mean(),
            'SD fraction near protein': frac.std(),
            'Mean enrichment': dei.mean(),
            'Median enrichment': np.median(dei),
            'SD enrichment': dei.std()
        }
        if self.compute_p_value:
            # sample T-test, 2-tailed
            t, p = scipy.stats.ttest_1samp(dei, 1)
            summary['Enrichment p-value'] = p

        return summary

    def _fit_binomial(self, data: dict, n_near_species: np.ndarray,
                      n_near: np.ndarray, n_species: np.ndarray,
                      n_all: np.ndarray, n_near_tot: int, n_all_tot: int):
        """
        This function computes the following approximate probability
        distributions and derives statistics accordingly.

        * The number of lipids near the protein is represented as a
        normal distribution.
        * The fraction of lipids near the protein follows a
        hypergeometric distribution.
        * The enrichment is represented as the log-normal distribution
        derived from the ratio of two binomial convolutions of the
        frame-by-frame binomial distributions.

        All these approximations assume that each frame or observation is
        independent. The binomial approximation assumes that:

        * the number of the lipid species near the protein is
        small compared to the total number of that lipid species
        * the total number of all lipids is large
        * the fraction (n_species / n_all) is not close to 0 or 1.

        .. note::

            The enrichment p-value is calculated from the log-normal
            distribution of the null hypothesis: that the average
            enrichment is representative of the ratio of
            n_species : n_all

        """

        summary = {}
        p_time = data['Fraction near protein']
        if n_near_tot:  # catch zeros
            p_real = n_near_species.sum() / n_near_tot
        else:
            p_real = 0
        if n_all_tot:
            p_exp = n_species.sum() / n_all_tot
        else:
            p_exp = 0

        # n events: assume normal
        summary['Mean near protein'] = n_near_species.mean()
        summary['SD near protein'] = n_near_species.std()

        # actually hypergeometric, but binomials are easier
        # X ~ B(n_near_tot, p_real)
        summary['Mean fraction near protein'] = p_real
        s2 = np.mean((p_time - p_real) ** 2)
        var_frac = ((p_real * (1-p_real)) - s2) * n_near_tot
        summary['SD fraction near protein'] = (var_frac ** 0.5) / n_near_tot

        # Now compute ratio of binomials
        # Let X ~ B(n, p_real),  Y ~ B(n, p_exp), T = (X/Y)
        # log T is approx. normally distributed. T is log-normal distribution
        # FIRST catch edge cases where this is not suitable
        if p_exp == 0:  # trivial case
            summary['Mean enrichment'] = 1  # 0 / 0
            summary['SD enrichment'] = 0
            summary['Median enrichment'] = 1
            if self.compute_p_value:
                summary['Enrichment p-value'] = 1
            return summary

        if p_real == 0:  # totally depleted
            summary['Mean enrichment'] = 0
            summary['SD enrichment'] = 0
            summary['Median enrichment'] = 0
            if self.compute_p_value:
                # X ~ B(n, p_exp)(0)
                p = scipy.stats.binom.cdf(0, n_near_tot, p_exp)
                summary['Enrichment p-value'] = p
            return summary

        if p_exp == 1:  # totally this thing
            summary['Mean enrichment'] = 1  # 1 / 1
            summary['SD enrichment'] = 0
            summary['Median enrichment'] = 1
            if self.compute_p_value:
                summary['Enrichment p-value'] = 1
            return summary

        mean_logT = np.log(p_real / p_exp)
        var_logT = ((1/p_real) + (1/p_exp) - 2) / n_near_tot
        summary['Mean log T'] = mean_logT
        summary['Var log T'] = var_logT
        # summary['1/p_real'] = 1/p_real
        # summary['1/p_exp'] = 1/p_exp
        # summary['Near near, total'] = n_near_tot
        summary['Mean enrichment'] = mean = np.exp(mean_logT) * np.exp(var_logT/2)
        summary['Variance enrichment'] = (np.exp(var_logT) - 1) * np.exp(2*mean_logT + var_logT)
        # var_enr = (np.exp(var_logT) - 1) * np.exp(2*mean_logT + var_logT)
        diff = (data['Enrichment'] - mean) ** 2
        var_pop = diff.sum() / (self.n_frames - 1)
        summary['Population SD enrichment'] = var_pop ** 0.5
        summary['Median enrichment'] = p_real / p_exp
        if self.compute_p_value:
            v_logT_ = 2 * (1/p_exp - 1) / n_near_tot
            s = v_logT_ ** 0.5  # sd
            avg = summary['Median enrichment']
            # if avg > 1:
            #     summary['Enrichment p-value'] = scipy.stats.lognorm.sf(avg, s)
            # else:
            #     summary['Enrichment p-value'] = scipy.stats.lognorm.cdf(avg, s)
            if avg <= 1:
                summary['Enrichment p-value'] = scipy.stats.lognorm.cdf(avg, s)
            else:
                summary['Enrichment p-value'] = scipy.stats.lognorm.sf(avg, s)
        return summary

    def _collate(self, n_near_species: np.ndarray, n_near: np.ndarray,
                 n_species: np.ndarray, n_all: np.ndarray,
                 n_near_tot: int, n_all_tot: int):
        data = {}
        data['Near protein'] = n_near_species
        frac = np.nan_to_num(n_near_species / n_near, nan=0.0)
        data['Fraction near protein'] = frac
        data['Total number'] = n_species
        n_total = np.nan_to_num(n_species / n_all, nan=0.0)
        data['Fraction total'] = n_total
        n_total[n_total == 0] = np.nan
        dei = np.nan_to_num(frac / n_total, nan=0.0)
        data['Enrichment'] = dei
        if self.compute_p_value:
            pval = np.zeros(len(frac))
            for i, args in enumerate(zip(dei, n_near_species, n_all,
                                         n_species, n_near_species)):
                pval[i] = self._compute_p(*args)
            data['Enrichment p-value'] = pval

        summary = self._fit_distribution(data, n_near_species, n_near,
                                         n_species, n_all, n_near_tot,
                                         n_all_tot)
        return data, summary

    def _conclude(self):
        self.leaflets = []
        self.leaflets_summary = []

        for i in range(self.n_leaflets):
            timeseries = {}
            summary = {}
            res_counts = self.residue_counts[i]
            near_counts = self.near_counts[i]

            near_all = near_counts.sum(axis=0)
            total_all = res_counts.sum(axis=0)
            n_near_tot = near_all.sum()
            n_all_tot = total_all.sum()
            d, s = self._collate(near_all, near_all, total_all,
                                 total_all, n_near_tot, n_all_tot)
            timeseries['all'] = d
            summary['all'] = s
            for j, resname in enumerate(self.ids):
                near_species = near_counts[j]
                total_species = res_counts[j]
                d, s = self._collate(near_species, near_all, total_species,
                                     total_all, n_near_tot, n_all_tot)
                timeseries[resname] = d
                summary[resname] = s
            self.leaflets.append(timeseries)
            self.leaflets_summary.append(summary)

    def summary_as_dataframe(self):
        """Convert the results summary into a pandas DataFrame.

        This requires pandas to be installed.
        """

        if not self.leaflets_summary:
            raise ValueError('Call run() first to get results')
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('pandas is required to use this function '
                              'but is not installed. Please install with '
                              '`conda install pandas` or '
                              '`pip install pandas`.') from None

        dfs = [pd.DataFrame.from_dict(d, orient='index')
               for d in self.leaflets_summary]
        for i, df in enumerate(dfs, 1):
            df['Leaflet'] = i
        df = pd.concat(dfs)
        return df

    def _compute_p_ttest(self, dei, *args):
        # sample T-test, 2-tailed
        t, p = scipy.stats.ttest_1samp(dei, 1)
        return p

    def _compute_p_hypergeom(self, dei, k, N, K, n):
        kn = k/n
        KN = K/N
        if kn <= KN:
            return scipy.stats.hypergeom.cdf(k, N, K, n)
        return scipy.stats.hypergeom.sf(k, N, K, n)
        # return scipy.stats.hypergeom.pmf(k, N, K, n)

    def _compute_p_hypergeom_gamma(self, dei, k, N, K, n):
        K_k = binomial_gamma(K, k)
        N_k = binomial_gamma(N - K, n - k)
        N_n = binomial_gamma(N, n)
        return K_k * N_k / N_n

    def _compute_p_gaussian(self, dei, k, N, K, n):
        kn = k/n
        KN = K/N
        sigma = KN / 2.5
        if kn <= KN:
            return scipy.stats.norm.cdf(kn, KN, sigma)
        return scipy.stats.norm.sf(kn, KN, sigma)

        # return scipy.stats.norm.pdf(k/n, K/N, sigma)
