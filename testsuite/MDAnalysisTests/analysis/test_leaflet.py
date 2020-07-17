# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
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
import MDAnalysis as mda
import pytest

from numpy.testing import assert_equal, assert_almost_equal
import numpy as np
import networkx as NX

from MDAnalysis.analysis.leaflet import (LeafletFinder, optimize_cutoff,
                                         LipidEnrichment)
from MDAnalysis.analysis import distances
from MDAnalysisTests.datafiles import (Martini_membrane_gro,
                                       GRO_MEMPROT,
                                       XTC_MEMPROT,
                                       )

try:
    import sklearn
except ImportError:
    sklearn_version = None
else:
    sklearn_version = sklearn.__version__

try:
    import pandas
except ImportError:
    has_pandas = False
else:
    has_pandas = True

skip_spectralclustering = pytest.mark.skipif((sklearn_version is None or
                                              sklearn_version < "0.23.1"),
                                             reason=("at least scikit-learn"
                                                     " >= 0.23.1 required"))


def lines2one(lines):
    """Join lines and squash all whitespace"""
    return " ".join(" ".join(lines).split())


@pytest.fixture()
def martini_universe():
    return mda.Universe(Martini_membrane_gro)


def test_optimize_cutoff():
    universe = mda.Universe(Martini_membrane_gro)
    cutoff, N = optimize_cutoff(universe, select="name PO4", pbc=True)
    assert N == 2
    assert_almost_equal(cutoff, 10.5, decimal=4)


def test_optimize_cutoff(martini_universe):
    cutoff, N = optimize_cutoff(martini_universe, select="name PO4",
                                pbc=True)
    assert N == 2
    assert_almost_equal(cutoff, 10.5, decimal=4)


class BaseTestLeafletFinder(object):


@pytest.mark.parametrize('n,sel,vec', [
    (0, "name PO4", [-0.5472627,  4.8972744, -8.8254544]),
    (180, "name ROH", [-0.0157176,  3.7500022, -6.7571378])
])
def test_orientation(martini_universe, n, sel, vec):
    # vectors checked by visual validation "does this look about right"
    res = martini_universe.residues[n]
    headgroup = res.atoms.select_atoms(sel)
    orientation = lipid_orientation(headgroup)
    assert_almost_equal(orientation, vec, decimal=5)

    @pytest.fixture()
    def universe(self):
        return mda.Universe(*self.files).select_atoms(self.select)


class BaseTestLeafletFinder(object):
    kwargs = {}
    n_lipids = 360
    select = "all"

    @pytest.fixture()
    def universe(self):
        return mda.Universe(*self.files).select_atoms(self.select)

    @pytest.fixture()
    def lipid_heads(self, universe):
        return universe.select_atoms(self.LIPID_HEAD_STRING)

    @pytest.fixture()
    def lfls(self, universe):
        return LeafletFinder(universe, select=self.LIPID_HEAD_STRING,
                             pbc=True, method=self.method, **self.kwargs)

    def test_leaflet_finder(self, universe, lfls):
        top_heads, bottom_heads = lfls.leaflets
        assert len(top_heads+bottom_heads) == len(lfls.selection.residues)
        assert_equal(top_heads.resids, self.leaflet_resids[0],
                     err_msg="Found wrong leaflet lipids")
        assert_equal(bottom_heads.resids, self.leaflet_resids[1],
                     err_msg="Found wrong leaflet lipids")

    def test_string_vs_atomgroup_proper(self, universe, lipid_heads, lfls):
        lfls_ag = LeafletFinder(lipid_heads, pbc=True, method=self.method,
                                **self.kwargs)
        groups_ag = lfls_ag.leaflets
        groups_string = lfls.leaflets
        assert_equal(groups_string[0].indices, groups_ag[0].indices)
        assert_equal(groups_string[1].indices, groups_ag[1].indices)


class BaseTestLeafletFinderMartini(BaseTestLeafletFinder):
    files = [Martini_membrane_gro]
    LIPID_HEAD_STRING = "name PO4"
    leaflet_resids = [np.arange(1, 181), np.arange(226, 406)]


class TestLeafletFinderByGraph(BaseTestLeafletFinderMartini):
    method = "graph"

    def test_pbc_on_off(self, universe, lfls):
        lfls_pbc_off = LeafletFinder(universe, select=self.LIPID_HEAD_STRING,
                                     pbc=False, method=self.method)
        assert lfls.predictor.size() > lfls_pbc_off.predictor.size()

    def test_pbc_on_off_difference(self, universe):
        lfls_pbc_on = LeafletFinder(universe, select=self.LIPID_HEAD_STRING,
                                    cutoff=7, pbc=True, method=self.method)
        lfls_pbc_off = LeafletFinder(universe, select=self.LIPID_HEAD_STRING,
                                     cutoff=7, pbc=False, method=self.method)
        pbc_on_graph = lfls_pbc_on.predictor
        pbc_off_graph = lfls_pbc_off.predictor
        diff_graph = NX.difference(pbc_on_graph, pbc_off_graph)
        assert_equal(set(diff_graph.edges), {(69, 153), (73, 79),
                                             (206, 317), (313, 319)})

    @pytest.mark.parametrize("sparse", [True, False, None])
    def test_sparse_on_off_none(self, universe, sparse):
        lfls_ag = LeafletFinder(universe, select=self.LIPID_HEAD_STRING,
                                cutoff=15.0, pbc=True, method=self.method,
                                sparse=sparse)
        assert_almost_equal(len(lfls_ag.predictor.edges), 1903, decimal=4)

    # def test_write_selection(self, universe, tmpdir, lfls):
    #     with tmpdir.as_cwd():
    #         filename = lfls.write_selection('leaflet.vmd')
    #         expected_output = lines2one(["""
    #         # Leaflets found by LeafletFinder(select='name PO4',
    #         cutoff=11.3 Å, pbc=True)
    #         # MDAnalysis VMD selection
    #         atomselect macro leaflet_1
    #         {index 1 13 25 37 49 61 73 85 \\
    #         97 109 121 133 145 157 169 181 \\
    #         193 205 217 229 241 253 265 277 \\
    #         289 301 313 325 337 349 361 373 \\
    #         385 397 409 421 433 445 457 469 \\
    #         481 493 505 517 529 541 553 565 \\
    #         577 589 601 613 625 637 649 661 \\
    #         673 685 697 709 721 733 745 757 \\
    #         769 781 793 805 817 829 841 853 \\
    #         865 877 889 901 913 925 937 949 \\
    #         961 973 985 997 1009 1021 1033 1045 \\
    #         1057 1069 1081 1093 1105 1117 1129 1141 \\
    #         1153 1165 1177 1189 1201 1213 1225 1237 \\
    #         1249 1261 1273 1285 1297 1309 1321 1333 \\
    #         1345 1357 1369 1381 1393 1405 1417 1429 \\
    #         1441 1453 1465 1477 1489 1501 1513 1525 \\
    #         1537 1549 1561 1573 1585 1597 1609 1621 \\
    #         1633 1645 1657 1669 1681 1693 1705 1717 \\
    #         1729 1741 1753 1765 1777 1789 1801 1813 \\
    #         1825 1837 1849 1861 1873 1885 1897 1909 \\
    #         1921 1933 1945 1957 1969 1981 1993 2005 \\
    #         2017 2029 2041 2053 2065 2077 2089 2101 \\
    #         2113 2125 2137 2149 }
    #         # MDAnalysis VMD selection
    #         atomselect macro leaflet_2
    #         {index 2521 2533 2545 2557 2569 2581 2593 2605 \\
    #         2617 2629 2641 2653 2665 2677 2689 2701 \\
    #         2713 2725 2737 2749 2761 2773 2785 2797 \\
    #         2809 2821 2833 2845 2857 2869 2881 2893 \\
    #         2905 2917 2929 2941 2953 2965 2977 2989 \\
    #         3001 3013 3025 3037 3049 3061 3073 3085 \\
    #         3097 3109 3121 3133 3145 3157 3169 3181 \\
    #         3193 3205 3217 3229 3241 3253 3265 3277 \\
    #         3289 3301 3313 3325 3337 3349 3361 3373 \\
    #         3385 3397 3409 3421 3433 3445 3457 3469 \\
    #         3481 3493 3505 3517 3529 3541 3553 3565 \\
    #         3577 3589 3601 3613 3625 3637 3649 3661 \\
    #         3673 3685 3697 3709 3721 3733 3745 3757 \\
    #         3769 3781 3793 3805 3817 3829 3841 3853 \\
    #         3865 3877 3889 3901 3913 3925 3937 3949 \\
    #         3961 3973 3985 3997 4009 4021 4033 4045 \\
    #         4057 4069 4081 4093 4105 4117 4129 4141 \\
    #         4153 4165 4177 4189 4201 4213 4225 4237 \\
    #         4249 4261 4273 4285 4297 4309 4321 4333 \\
    #         4345 4357 4369 4381 4393 4405 4417 4429 \\
    #         4441 4453 4465 4477 4489 4501 4513 4525 \\
    #         4537 4549 4561 4573 4585 4597 4609 4621 \\
    #         4633 4645 4657 4669 }

    # """])
    #         with open('leaflet.vmd', 'r') as f:
    #             lines = f.readlines()
    #         assert lines2one(lines) == expected_output


@skip_spectralclustering
class TestLeafletFinderBySC(BaseTestLeafletFinderMartini):
    method = "spectralclustering"


class TestLeafletFinderByOrientation(BaseTestLeafletFinderMartini):
    method = "orientation"

    def test_half_chol(self, universe):
        ag = universe.residues[::2].atoms.select_atoms('resname CHOL')
        lf = LeafletFinder(ag, select="name ROH", method="orientation",
                           cutoff="guess", n_groups=2)
        assert_equal(lf.leaflets[0].resids,
                     [181, 183, 185, 187, 189, 191, 193, 195, 197, 199,
                      201, 203, 207, 209, 211, 213, 215, 217, 219, 221,
                      223, 225])
        assert_equal(lf.leaflets[1].resids,
                     [205, 407, 409, 411, 413, 415, 417, 419, 421, 423,
                      425, 427, 429, 431, 433, 435, 437, 439, 441, 443,
                      445, 447, 449])

    def test_fifth_chol(self, universe):
        ag = universe.residues[::5].atoms.select_atoms('resname CHOL')
        lf = LeafletFinder(ag, select="name ROH", method="orientation",
                           cutoff="guess", n_groups=2)
        assert_equal(lf.leaflets[0].resids,
                     [181, 186, 191, 196, 201, 206, 211, 221])
        assert_equal(lf.leaflets[1].resids,
                     [216, 406, 411, 416, 421, 426, 431, 436, 441, 446])


class TestLeafletFinderByCOG(BaseTestLeafletFinderMartini):
    method = "center_of_geometry"
    kwargs = {'centers': [[55.63316663, 56.79550008, 73.80222244],
                          [56.81394444, 55.90877751, 33.33372219]]}


class TestLeafletFinderByCOGDirect(BaseTestLeafletFinderMartini):
    kwargs = {'centers': [[55.63316663, 56.79550008, 73.80222244],
                          [56.81394444, 55.90877751, 33.33372219]]}

    @staticmethod
    def method(*args, **kwargs):
        return distances.group_coordinates_by_cog(*args, **kwargs)


class BaseTestLFMemProt(BaseTestLeafletFinder):
    files = [GRO_MEMPROT, XTC_MEMPROT]
    select = "resname POPE POPG"
    LIPID_HEAD_STRING = "name P*"
    leaflet_resids = [list(range(297, 410)) + list(range(518, 546)),
                      list(range(410, 518)) + list(range(546, 573))]


class TestLFMemProtGraph(BaseTestLFMemProt):
    method = "graph"


class TestLFMemProtOrientation(BaseTestLFMemProt):
    method = "orientation"


@skip_spectralclustering
class TestLFMemProtSC(BaseTestLFMemProt):
    method = "spectralclustering"


class BaseTestOrientationLeafletFinder(object):
    n_groups = 2
    LIPID_HEAD_STRING = "name ROH GL1 PO4"
    kwargs = {}

    @pytest.fixture()
    def universe(self):
        return mda.Universe(self.file)

    def test_full(self, universe):
        lf = LeafletFinder(universe.atoms, select=self.LIPID_HEAD_STRING,
                           n_groups=self.n_groups, method="orientation",
                           **self.kwargs)
        assert len(lf.leaflets) == self.n_groups
        assert_equal(lf.sizes, self.n_full_leaflets)
        for found, known in zip(lf.leaflets, self.full_leaflets_20):
            assert_equal(found.residues.resindices[::20], known)

    def test_half(self, universe):
        ag = universe.residues[::2]
        lf = LeafletFinder(ag, select=self.LIPID_HEAD_STRING,
                           n_groups=self.n_groups, method="orientation",
                           **self.kwargs)
        assert len(lf.leaflets) == self.n_groups
        assert_equal(lf.sizes, self.n_half_leaflets)
        for found, known in zip(lf.leaflets, self.half_leaflets_20):
            assert_equal(found.residues.resindices[::20], known)

    def test_fifth(self, universe):
        ag = universe.residues[::5]
        lf = LeafletFinder(ag, select=self.LIPID_HEAD_STRING,
                           n_groups=self.n_groups, method="orientation",
                           **self.kwargs)
        assert len(lf.leaflets) == self.n_groups
        assert_equal(lf.sizes, self.n_fifth_leaflets)
        for found, known in zip(lf.leaflets, self.fifth_leaflets_20):
            assert_equal(found.residues.resindices[::20], known)


class TestOrientationDoubleBilayer(BaseTestOrientationLeafletFinder):
    file = Martini_double_membrane
    n_groups = 4
    n_full_leaflets = [407, 407, 403, 403]
    n_half_leaflets = [203, 203, 202, 202]
    n_half_chol = [23, 23, 22, 22]
    n_fifth_leaflets = [82, 82, 80, 80]
    n_fifth_chol = [10, 10, 8, 8]
    full_leaflets_20 = ([450, 470, 490, 510, 530, 550, 570, 590,
                         610, 630, 650, 672],
                        [654, 693, 713, 733, 753, 773, 793, 813,
                         833, 853, 873, 893],
                        [0,  20,  40,  60,  80, 100, 120, 140,
                         160, 180, 200, 222],
                        [204, 243, 263, 283, 303, 323, 343, 363,
                         383, 403, 423, 443])

    half_leaflets_20 = ([450, 490, 530, 570, 610, 650],
                        [654, 714, 754, 794, 834, 874],
                        [0,  40,  80, 120, 160, 200],
                        [204, 264, 304, 344, 384, 424])

    fifth_leaflets_20 = ([450, 550, 650], [665, 770, 870],
                         [0, 100, 200], [215, 320, 420])

    half_chol_2 = ([630, 634, 638, 642, 646, 650, 656, 660,
                    664, 668, 672],
                   [654, 858, 862, 866, 870, 874, 878, 882,
                    886, 890, 894, 898],
                   [180, 184, 188, 192, 196, 200, 206, 210,
                    214, 218, 222],
                   [204, 408, 412, 416, 420, 424, 428, 432,
                    436, 440, 444, 448])

    fifth_chol = ([630, 635, 640, 645, 650, 655, 660, 670],
                  [665, 855, 860, 865, 870, 875, 880, 885,
                   890, 895],
                  [180, 185, 190, 195, 200, 205, 210, 220],
                  [215, 405, 410, 415, 420, 425, 430, 435,
                   440, 445])

    def test_half_chol(self, universe):
        ag = universe.residues[::2].atoms.select_atoms('resname CHOL')
        lf = LeafletFinder(ag, select=self.LIPID_HEAD_STRING,
                           n_groups=self.n_groups, method="orientation",
                           **self.kwargs)
        assert len(lf.leaflets) == self.n_groups
        assert_equal(lf.sizes, self.n_half_chol)
        for found, known in zip(lf.leaflets, self.half_chol_2):
            assert_equal(found.residues.resindices[::2], known)

    def test_fifth_chol(self, universe):
        ag = universe.residues[::5].atoms.select_atoms('resname CHOL')
        lf = LeafletFinder(ag, select=self.LIPID_HEAD_STRING,
                           n_groups=self.n_groups, method="orientation",
                           **self.kwargs)
        assert len(lf.leaflets) == self.n_groups
        assert_equal(lf.sizes, self.n_fifth_chol)
        for found, known in zip(lf.leaflets, self.fifth_chol):
            assert_equal(found.residues.resindices, known)


class TestOrientationSingleBilayer(TestOrientationDoubleBilayer):
    file = Martini_membrane_gro
    n_groups = 2
    n_full_leaflets = [407, 403]
    n_half_leaflets = [203, 202]
    n_half_chol = [23, 22]
    n_fifth_leaflets = [82, 80]
    n_fifth_chol = [10, 8]
    full_leaflets_20 = ([0,  20,  40,  60,  80, 100, 120, 140,
                         160, 180, 200, 222],
                        [204, 243, 263, 283, 303, 323, 343, 363,
                         383, 403, 423, 443])

    half_leaflets_20 = ([0,  40,  80, 120, 160, 200],
                        [204, 264, 304, 344, 384, 424])

    fifth_leaflets_20 = ([0, 100, 200], [215, 320, 420])

    half_chol_2 = ([180, 184, 188, 192, 196, 200, 206, 210,
                    214, 218, 222],
                   [204, 408, 412, 416, 420, 424, 428, 432,
                    436, 440, 444, 448])

    fifth_chol = ([180, 185, 190, 195, 200, 205, 210, 220],
                  [215, 405, 410, 415, 420, 425, 430, 435,
                   440, 445])


class TestOrientationVesicle(BaseTestOrientationLeafletFinder):
    file = DPPC_vesicle_only
    n_full_leaflets = [3702, 2358]
    n_half_leaflets = [1892, 1138]
    n_fifth_leaflets = [724, 488]
    full_leaflets_20 = ([0,   43,   76,  112,  141,  172,  204,
                         234,  270,  301,  342,  377,  409,  441,
                         474,  513,  544,  579,  621,  647,  677,
                         715,  747,  771,  811,  847,  882,  914,
                         951,  982, 1016, 1046, 1084, 1116, 1150,
                         1181, 1210, 1246, 1278, 1312, 1351, 1375,
                         1401, 1440, 1476, 1505, 1549, 1582, 1618,
                         1648, 1680, 1713, 1740, 1780, 1810, 1841,
                         1864, 1899, 1936, 1974, 1999, 2033, 2066,
                         2095, 2127, 2181, 2207, 2243, 2278, 2311,
                         2336, 2368, 2400, 2427, 2456, 2482, 2515,
                         2547, 2575, 2608, 2636, 2665, 2693, 2720,
                         2748, 2792, 2822, 2860, 2891, 2936, 2960,
                         2992, 3017],
                        [3,   36,   89,  139,  198,  249,  298,
                         340,  388,  435,  491,  528,  583,  620,
                         681,  730,  794,  831,  877,  932,  979,
                         1032, 1073, 1132, 1180, 1238, 1286, 1328,
                         1396, 1441, 1490, 1528, 1577, 1625, 1688,
                         1742, 1782, 1839, 1910, 1945, 2005, 2057,
                         2111, 2153, 2180, 2236, 2286, 2342, 2401,
                         2470, 2528, 2584, 2649, 2722, 2773, 2818,
                         2861, 2905, 2961])

    half_leaflets_20 = ([0,   74,  134,  188,  250,  306,  362,
                         452,  524,  588,  660,  736,  796,  872,
                         928,  996, 1066, 1120, 1190, 1252, 1304,
                         1374, 1434, 1512, 1576, 1638, 1686, 1750,
                         1818, 1872, 1954, 2008, 2078, 2146, 2222,
                         2296, 2346, 2398, 2460, 2524, 2590, 2646,
                         2702, 2756, 2836, 2900, 2958, 3012],
                        [4,   98,  228,  350,  434,  518,  614,
                         696,  806,  912, 1006, 1124, 1220, 1328,
                         1452, 1528, 1666, 1776, 1892, 1972, 2088,
                         2174, 2264, 2410, 2520, 2626, 2766, 2854,
                         2972])

    fifth_leaflets_20 = ([0,  175,  355,  540,  735,  890, 1105,
                          1270, 1430, 1580, 1735, 1885, 2095, 2300,
                          2445, 2585, 2720, 2885, 3020],
                         [5,  265,  465,  650,  915, 1095, 1325,
                             1675, 1920, 2115, 2305, 2640, 2945])


class TestOrientationVesicleMessy(TestOrientationVesicle):
    file = DPPC_vesicle_plus
    n_full_leaflets = [3702, 2358]
    n_half_leaflets = [1868, 1162]
    n_fifth_leaflets = [754, 464]
    n_groups = 2
    kwargs = {'min_group': 20, 'cutoff': 20}

    full_leaflets_20 = ([0,   44,   77,  113,  142,  174,  206,
                         238,  274,  305,  350,  385,  419,  452,
                         485,  525,  556,  591,  633,  659,  689,
                         728,  760,  785,  825,  862,  897,  929,
                         966,  998, 1033, 1063, 1101, 1133, 1167,
                         1199, 1228, 1265, 1297, 1333, 1373, 1398,
                         1424, 1464, 1500, 1529, 1574, 1608, 1644,
                         1674, 1706, 1739, 1767, 1809, 1839, 1870,
                         1893, 1929, 1966, 2005, 2030, 2064, 2097,
                         2126, 2159, 2214, 2240, 2276, 2311, 2345,
                         2370, 2402, 2435, 2462, 2491, 2517, 2550,
                         2583, 2612, 2646, 2674, 2703, 2731, 2758,
                         2786, 2831, 2862, 2900, 2931, 2977, 3002,
                         3034, 3059],
                        [3,   37,   90,  140,  200,  253,  302,
                         348,  396,  445,  502,  540,  595,  632,
                         693,  743,  808,  846,  892,  947,  995,
                         1049, 1090, 1149, 1198, 1257, 1306, 1350,
                         1419, 1465, 1514, 1553, 1603, 1651, 1714,
                         1769, 1811, 1868, 1940, 1975, 2036, 2088,
                         2143, 2185, 2213, 2269, 2319, 2376, 2436,
                         2505, 2563, 2621, 2687, 2760, 2812, 2858,
                         2901, 2945, 3003])

    half_leaflets_20 = ([4,   90,  186,  294,  404,  518,  614,
                         712,  818,  894,  980, 1072, 1156, 1274,
                         1354, 1472, 1546, 1652, 1780, 1892, 1972,
                         2080, 2192, 2282, 2426, 2580, 2696, 2818,
                         2932, 3064],
                        [0,   74,  148,  212,  280,  338,  414,
                         470,  544,  616,  684,  752,  822,  902,
                         982, 1050, 1132, 1186, 1250, 1336, 1398,
                         1466, 1556, 1612, 1674, 1728, 1798, 1864,
                         1930, 2014, 2070, 2128, 2190, 2268, 2330,
                         2380, 2442, 2490, 2546, 2608, 2674, 2728,
                         2784, 2852, 2912, 2982, 3034])
    fifth_leaflets_20 = ([0,  175,  360,  495,  725,  855, 1045,
                          1180, 1345, 1515, 1700, 1840, 2030, 2220,
                          2360, 2485, 2635, 2780, 2980],
                         [5,  245,  515,  705,  980, 1295, 1535,
                          1805, 2025, 2250, 2610, 2875])

    def test_fifth(self, universe):
        ag = universe.residues[::5]
        lf = LeafletFinder(ag, select=self.LIPID_HEAD_STRING,
                           n_groups=self.n_groups, method="orientation",
                           cutoff=30, min_group=20)
        assert len(lf.leaflets) == self.n_groups
        assert_equal(lf.sizes, self.n_fifth_leaflets)
        for found, known in zip(lf.groups, self.fifth_leaflets_20):
            assert_equal(found.residues.resindices[::20], known)
