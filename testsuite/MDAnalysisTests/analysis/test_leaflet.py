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


def test_optimize_cutoff():
    universe = mda.Universe(Martini_membrane_gro)
    cutoff, N = optimize_cutoff(universe, select="name PO4", pbc=True)
    assert N == 2
    assert_almost_equal(cutoff, 10.5, decimal=4)


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
        return LeafletFinder(universe, select=self.LIPID_HEAD_STRING, pbc=True,
                             method=self.method, **self.kwargs)

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

    def test_write_selection(self, universe, tmpdir, lfls):
        with tmpdir.as_cwd():
            filename = lfls.write_selection('leaflet.vmd')
            expected_output = lines2one([
                """# Leaflets found by LeafletFinder(select='name PO4', cutoff=15.0 Å, pbc=True)
            # MDAnalysis VMD selection
            atomselect macro leaflet_1 {index 1 13 25 37 49 61 73 85 \\
            97 109 121 133 145 157 169 181 \\
            193 205 217 229 241 253 265 277 \\
            289 301 313 325 337 349 361 373 \\
            385 397 409 421 433 445 457 469 \\
            481 493 505 517 529 541 553 565 \\
            577 589 601 613 625 637 649 661 \\
            673 685 697 709 721 733 745 757 \\
            769 781 793 805 817 829 841 853 \\
            865 877 889 901 913 925 937 949 \\
            961 973 985 997 1009 1021 1033 1045 \\
            1057 1069 1081 1093 1105 1117 1129 1141 \\
            1153 1165 1177 1189 1201 1213 1225 1237 \\
            1249 1261 1273 1285 1297 1309 1321 1333 \\
            1345 1357 1369 1381 1393 1405 1417 1429 \\
            1441 1453 1465 1477 1489 1501 1513 1525 \\
            1537 1549 1561 1573 1585 1597 1609 1621 \\
            1633 1645 1657 1669 1681 1693 1705 1717 \\
            1729 1741 1753 1765 1777 1789 1801 1813 \\
            1825 1837 1849 1861 1873 1885 1897 1909 \\
            1921 1933 1945 1957 1969 1981 1993 2005 \\
            2017 2029 2041 2053 2065 2077 2089 2101 \\
            2113 2125 2137 2149 }
            # MDAnalysis VMD selection
            atomselect macro leaflet_2 {index 2521 2533 2545 2557 2569 2581 2593 2605 \\
            2617 2629 2641 2653 2665 2677 2689 2701 \\
            2713 2725 2737 2749 2761 2773 2785 2797 \\
            2809 2821 2833 2845 2857 2869 2881 2893 \\
            2905 2917 2929 2941 2953 2965 2977 2989 \\
            3001 3013 3025 3037 3049 3061 3073 3085 \\
            3097 3109 3121 3133 3145 3157 3169 3181 \\
            3193 3205 3217 3229 3241 3253 3265 3277 \\
            3289 3301 3313 3325 3337 3349 3361 3373 \\
            3385 3397 3409 3421 3433 3445 3457 3469 \\
            3481 3493 3505 3517 3529 3541 3553 3565 \\
            3577 3589 3601 3613 3625 3637 3649 3661 \\
            3673 3685 3697 3709 3721 3733 3745 3757 \\
            3769 3781 3793 3805 3817 3829 3841 3853 \\
            3865 3877 3889 3901 3913 3925 3937 3949 \\
            3961 3973 3985 3997 4009 4021 4033 4045 \\
            4057 4069 4081 4093 4105 4117 4129 4141 \\
            4153 4165 4177 4189 4201 4213 4225 4237 \\
            4249 4261 4273 4285 4297 4309 4321 4333 \\
            4345 4357 4369 4381 4393 4405 4417 4429 \\
            4441 4453 4465 4477 4489 4501 4513 4525 \\
            4537 4549 4561 4573 4585 4597 4609 4621 \\
            4633 4645 4657 4669 }

    """])
            with open('leaflet.vmd', 'r') as f:
                lines = f.readlines()
            assert lines2one(lines) == expected_output


@skip_spectralclustering
class TestLeafletFinderBySCMembrane(BaseTestLeafletFinderMartini):
    method = "spectralclustering"
    kwargs = {'n_groups': 2, 'cutoff': 100}


class TestLeafletFinderByCOG(BaseTestLeafletFinderMartini):
    method = "center_of_geometry"
    kwargs = {'centers': [[55.63316663, 56.79550008, 73.80222244],
                          [56.81394444, 55.90877751, 33.33372219]]}


@skip_spectralclustering
class TestLeafletFinderMemProtAA(BaseTestLeafletFinder):
    files = [GRO_MEMPROT, XTC_MEMPROT]
    select = "resname POPE POPG"
    LIPID_HEAD_STRING = "name P*"
    method = "spectralclustering"
    kwargs = {'n_groups': 2, 'cutoff': 100}
    leaflet_resids = [list(range(297, 410)) + list(range(518, 546)),
                      list(range(410, 518)) + list(range(546, 573))]


@skip_spectralclustering
class TestLipidEnrichmentMembrane:
    @pytest.fixture()
    def lipen(self):
        u = mda.Universe(Martini_membrane_gro)
        return LipidEnrichment(u, select_protein='protein',
                               select_headgroup='name PO4',
                               select_residues='resname DPPC',
                               enrichment_cutoff=6,
                               distribution='gaussian',
                               compute_p_value=False).run()

    def test_empty_results(self, lipen):
        assert len(lipen.leaflets) == 2
        top, bottom = lipen.leaflets
        assert len(top) == 2
        assert len(bottom) == 2
        assert_equal(top['DPPC']['Near protein'], 0)
        assert_equal(top['all']['Near protein'], 0)


@skip_spectralclustering
class BaseTestLipidEnrichmentMemProt:
    files = [GRO_MEMPROT, XTC_MEMPROT]
    lipid_sel = 'resname POPE POPG'
    headgroup_sel = 'name P*'

    protein_sel = 'protein'
    avg_c = 'Average near protein'
    sd_c = 'SD near protein'
    avg_frac = 'Average fraction near protein'
    sd_frac = 'SD fraction near protein'
    avg_en = 'Average enrichment'
    sd_en = 'SD enrichment'
    med_en = 'Median enrichment'
    p_en = 'Enrichment p-value'

    # lipids don't flip
    keys = {'POPE', 'POPG', 'all'}
    n_u_POPE = 113
    n_l_POPE = 108
    n_u_POPG = 28
    n_l_POPG = 27
    n_upper = 113+28
    n_lower = 108+27
    n_lipids = 276

    @pytest.fixture()
    def universe(self):
        return mda.Universe(*self.files)

    @pytest.fixture()
    def lipen(self, universe):
        return LipidEnrichment(universe, select_protein=self.protein_sel,
                               select_headgroup=self.headgroup_sel,
                               select_residues=self.lipid_sel,
                               enrichment_cutoff=self.cutoff,
                               distribution=self.distribution,
                               compute_p_value=True).run()

    def test_results(self, lipen):
        upper, lower = lipen.leaflets
        assert_equal(set(upper.keys()), self.keys)
        assert_equal(set(lower.keys()), self.keys)

        for i, leaflet in enumerate(lipen.leaflets):
            for lipid in self.keys:
                assert_equal(leaflet[lipid]['Near protein'],
                             self.near_prot[i][lipid])
                assert_almost_equal(leaflet[lipid]['Fraction near protein'],
                                    self.frac[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet[lipid]['Enrichment'],
                                    self.dei[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet[lipid]['Enrichment p-value'],
                                    self.dei_p[i][lipid],
                                    decimal=4)

    def test_results_summary(self, lipen):
        upper, lower = lipen.leaflets_summary
        assert_equal(set(upper.keys()), self.keys)
        assert_equal(set(lower.keys()), self.keys)

        for i, leaflet in enumerate(lipen.leaflets_summary):
            for lipid in self.keys:
                assert_almost_equal(leaflet[lipid][self.avg_c],
                                    self.near_prot_mean[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet[lipid][self.sd_c],
                                    self.near_prot_sd[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet[lipid][self.avg_frac],
                                    self.frac_mean[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet[lipid][self.sd_frac],
                                    self.frac_sd[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet[lipid][self.avg_en],
                                    self.dei_mean[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet[lipid][self.med_en],
                                    self.dei_median[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet[lipid][self.sd_en],
                                    self.dei_sd[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet[lipid][self.p_en],
                                    self.dei_all_p[i][lipid],
                                    decimal=4)

    @pytest.mark.skipif(not has_pandas,
                        reason="Need pandas for this function")
    def test_results_summary_df(self, lipen):
        df = lipen.summary_as_dataframe()
        upper = df[df.Leaflet == 1]
        lower = df[df.Leaflet == 2]

        for i, leaflet in enumerate([upper, lower]):
            for lipid in self.keys:
                assert_almost_equal(leaflet.loc[lipid, self.avg_c],
                                    self.near_prot_mean[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet.loc[lipid, self.sd_c],
                                    self.near_prot_sd[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet.loc[lipid, self.avg_frac],
                                    self.frac_mean[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet.loc[lipid, self.sd_frac],
                                    self.frac_sd[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet.loc[lipid, self.avg_en],
                                    self.dei_mean[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet.loc[lipid, self.sd_en],
                                    self.dei_sd[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet.loc[lipid, self.med_en],
                                    self.dei_median[i][lipid],
                                    decimal=4)
                assert_almost_equal(leaflet.loc[lipid, self.p_en],
                                    self.dei_all_p[i][lipid],
                                    decimal=4)


class TestLipidEnrichmentMemProtGaussian(BaseTestLipidEnrichmentMemProt):
    cutoff = 4
    distribution = 'gaussian'

    # counts near protein by selection language ('around 4 protein')
    near_prot = [
        {'POPE': [4, 7, 6, 2, 3],
         'POPG': [0, 0, 0, 0, 0],
         'all': [4, 7, 6, 2, 3]},
        {'POPE': [8, 11, 10, 10, 7],
         'POPG': [5, 5, 3, 6, 7],
         'all': [13, 16, 13, 16, 14]}
    ]
    frac = [
        {'POPE': [1, 1, 1, 1, 1],
         'POPG': [0, 0, 0, 0, 0],
         'all': [1, 1, 1, 1, 1]},
        {'POPE': [0.6154, 0.6875, 0.7692, 0.625, 0.5],
         'POPG': [0.3846, 0.3125, 0.2308, 0.375, 0.5],
         'all': [1, 1, 1, 1, 1]}
    ]

    dei = [
        {'POPE': [1.2478, 1.2478, 1.2478, 1.2478, 1.2478],
         'POPG': [0, 0, 0, 0, 0],
         'all': [1, 1, 1, 1, 1]},
        {'POPE': [0.7692, 0.8594, 0.9615, 0.7813, 0.625],
         'POPG': [1.9231, 1.5625, 1.1538, 1.875, 2.5],
         'all': [1, 1, 1, 1, 1]}
    ]

    dei_p = [
        {'POPE': [0.4081, 0.2044, 0.2578, 0.6411, 0.512],
         'POPG': [0.4081, 0.2044, 0.2578, 0.6411, 0.512],
         'all': [1, 1, 1, 1, 1]},
        {'POPE': [0.065, 0.1207, 0.2589, 0.0497, 0.0065],
         'POPG': [0.065, 0.1207, 0.2589, 0.0497, 0.0065],
         'all': [1, 1, 1, 1, 1]}
    ]

    near_prot_mean = [
        {'POPE': 4.4, 'POPG': 0, 'all': 4.4},
        {'POPE': 9.2, 'POPG': 5.2, 'all': 14.4},
    ]
    near_prot_sd = [
        {'POPE': 1.8547, 'POPG': 0,      'all': 1.8547},
        {'POPE': 1.4697, 'POPG': 1.3266, 'all': 1.3565},
    ]
    frac_mean = [
        {'POPE': 1,      'POPG': 0,      'all': 1},
        {'POPE': 0.6394, 'POPG': 0.3606, 'all': 1},
    ]
    frac_sd = [
        {'POPE': 0,      'POPG': 0,      'all': 0},
        {'POPE': 0.0888, 'POPG': 0.0888, 'all': 0},
    ]
    dei_mean = [
        {'POPE': 1.2478, 'POPG': 0,      'all': 1},
        {'POPE': 0.7993, 'POPG': 1.8029, 'all': 1},
    ]
    dei_median = [
        {'POPE': 1.2478, 'POPG': 0,      'all': 1},
        {'POPE': 0.7813, 'POPG': 1.875, 'all': 1},
    ]
    dei_sd = [
        {'POPE': 0,      'POPG': 0,      'all': 0},
        {'POPE': 0.1109, 'POPG': 0.4438, 'all': 0},
    ]
    dei_all_p = [
        {'POPE': 0.4289, 'POPG': 0.0722, 'all': 1},
        {'POPE': 0.4349, 'POPG': 0.1780, 'all': 1},
    ]


class TestLipidEnrichmentMemProtBinomial(TestLipidEnrichmentMemProtGaussian):
    distribution = 'binomial'

    near_prot_mean = [
        {'POPE': 4.4, 'POPG': 0, 'all': 4.4},
        {'POPE': 9.2, 'POPG': 5.2, 'all': 14.4},
    ]
    near_prot_sd = [
        {'POPE': 1.8547, 'POPG': 0,      'all': 1.8547},
        {'POPE': 1.4697, 'POPG': 1.3266, 'all': 1.3565},
    ]
    frac_mean = [
        {'POPE': 1,      'POPG': 0,      'all': 1},
        {'POPE': 0.6389, 'POPG': 0.3611, 'all': 1},
    ]
    frac_sd = [
        {'POPE': 0,      'POPG': 0,      'all': 0},
        {'POPE': 0.0556, 'POPG': 0.0556, 'all': 0},
    ]
    dei_mean = [
        {'POPE': 1.2548, 'POPG': 0,      'all': 1},
        {'POPE': 0.8031, 'POPG': 1.8794, 'all': 1},
    ]
    dei_median = [
        {'POPE': 1.2478, 'POPG': 0,      'all': 1},
        {'POPE': 0.7986, 'POPG': 1.8056, 'all': 1},
    ]
    dei_sd = [
        {'POPE': 0.1335, 'POPG': 0,      'all': 0},
        {'POPE': 0.0857, 'POPG': 0.5428, 'all': 0},
    ]
    dei_all_p = [
        {'POPE': 0.6749, 'POPG': 0.0077, 'all': 1},
        {'POPE': 0.1873, 'POPG': 0.1062, 'all': 1},
    ]
