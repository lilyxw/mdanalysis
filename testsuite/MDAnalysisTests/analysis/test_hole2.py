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
from __future__ import print_function, absolute_import

from six.moves import range
import pytest
import glob
import os
import textwrap

import MDAnalysis as mda
from MDAnalysis.analysis import hole2

from numpy.testing import (
    assert_almost_equal,
    assert_equal,
)
import numpy as np
import matplotlib
import mpl_toolkits.mplot3d

import errno

from MDAnalysisTests.datafiles import PDB_HOLE, MULTIPDB_HOLE
from MDAnalysisTests import executable_not_found


def rlimits_missing():
    # return True if resources module not accesible (ie setting of rlimits)
    try:
        # on Unix we can manipulate our limits: http://docs.python.org/2/library/resource.html
        import resource

        soft_max_open_files, hard_max_open_files = resource.getrlimit(
            resource.RLIMIT_NOFILE)
    except ImportError:
        return True
    return False

@pytest.mark.skipif(executable_not_found("hole"),
                    reason="Test skipped because HOLE not found")
class TestHole(object):
    @staticmethod
    @pytest.fixture()
    def profiles(tmpdir):
        with tmpdir.as_cwd():
            filename = PDB_HOLE
            return hole2.hole(filename, random_seed=31415)

    def test_correct_input(self, tmpdir):
        with tmpdir.as_cwd():
            filename = PDB_HOLE
            hole2.hole(filename, random_seed=31415, infile='hole.inp')

        infile = str(tmpdir.join('hole.inp'))
        with open(infile, 'r') as f:
            contents = f.read()

        hole_input = textwrap.dedent("""
            RADIUS simple2.rad
            SPHPDB hole.sph
            SAMPLE 0.200000
            ENDRAD 22.000000
            IGNORE SOL WAT TIP HOH K   NA  CL 
            SHORTO 0
            RASEED 31415
            """)

        # don't check filenames
        assert contents.endswith(hole_input)

    def test_input_with_cpoint(self, tmpdir):
        u = mda.Universe(PDB_HOLE)
        cog = u.select_atoms('protein').center_of_geometry()

        with tmpdir.as_cwd():
            filename = PDB_HOLE
            hole2.hole(filename, random_seed=31415,
                       infile='hole.inp', cpoint=cog)

        infile = str(tmpdir.join('hole.inp'))
        with open(infile, 'r') as f:
            contents = f.read()

        hole_input = textwrap.dedent("""
            RADIUS simple2.rad
            SPHPDB hole.sph
            SAMPLE 0.200000
            ENDRAD 22.000000
            IGNORE SOL WAT TIP HOH K   NA  CL 
            SHORTO 0
            RASEED 31415
            CPOINT -0.0180961507 -0.0122730583 4.1497999943
            """)

        # don't check filenames
        assert contents.endswith(hole_input)

    def test_correct_profile_values(self, profiles):
        values = list(profiles.values())
        assert len(values) == 1, 'Should only have 1 HOLE profile'
        profile = values[0]
        assert len(profile) == 425, 'Wrong number of points in HOLE profile'
        assert_almost_equal(profile.rxn_coord.mean(),
                            -1.41225,
                            err_msg='Wrong mean HOLE rxn_coord')
        assert_almost_equal(profile.radius.min(),
                            1.19707,
                            err_msg='Wrong minimum HOLE radius')


@pytest.mark.skipif(executable_not_found("hole"),
                    reason="Test skipped because HOLE not found")
class BaseTestHole(object):
    filename = MULTIPDB_HOLE
    start = 5
    stop = 7

    # HOLE is so slow so we only run it once and keep it in
    # the class; note that you may not change universe.trajectory
    # (eg iteration) because this is not safe in parallel
    @pytest.fixture()
    def universe(self):
        return mda.Universe(self.filename)

    @pytest.fixture()
    def hole(self, universe, tmpdir):
        with tmpdir.as_cwd():
            h = hole2.HoleAnalysis(universe)
            h.run(start=self.start, stop=self.stop, random_seed=31415)
        return h

    @pytest.fixture()
    def frames(self, universe):
        return [ts.frame for ts in universe.trajectory[self.start:self.stop]]


class TestHoleAnalysis(BaseTestHole):

    def test_correct_profile_values(self, hole, frames):
        assert_equal(sorted(hole.profiles.keys()), frames,
                     err_msg="hole.profiles.keys() should contain the frame numbers")
        assert_equal(list(hole.frames), frames,
                     err_msg="hole.frames should contain the frame numbers")
        data = np.transpose([(len(p), p.rxn_coord.mean(), p.radius.min())
                             for p in hole.profiles.values()])
        assert_equal(data[0], [401, 399], err_msg="incorrect profile lengths")
        assert_almost_equal(data[1], [1.98767,  0.0878],
                            err_msg="wrong mean HOLE rxn_coord")
        assert_almost_equal(data[2], [1.19819, 1.29628],
                            err_msg="wrong minimum radius")

    def test_min_radius(self, hole):
        values = np.array([[5., 1.19819],
                           [6., 1.29628]])
        assert_almost_equal(hole.min_radius(), values,
                            err_msg="min_radius() array not correct")

    def test_plot(self, hole):
        ax = hole.plot(label=True)
        err_msg = "HoleAnalysis.plot() did not produce an Axes instance"
        assert isinstance(ax, matplotlib.axes.Axes), err_msg

    def test_plot3D(self, hole):
        ax = hole.plot3D()
        err_msg = "HoleAnalysis.plot3D() did not produce an Axes3D instance"
        assert isinstance(ax, mpl_toolkits.mplot3d.Axes3D), err_msg

    def test_plot3D_rmax(self, hole):
        ax = hole.plot3D(r_max=2.5)
        err_msg = "HoleAnalysis.plot3D(rmax=float) did not produce an Axes3D instance"
        assert isinstance(ax, mpl_toolkits.mplot3d.Axes3D), err_msg

    def test_context_manager(self, universe, tmpdir):
        with tmpdir.as_cwd():
            with hole2.HoleAnalysis(universe) as h:
                h.run()
                h.run()
                h.create_vmd_surface(filename='hole.vmd')

        sphpdbs = tmpdir.join('*.sph')
        assert len(glob.glob(str(sphpdbs))) == 0
        outfiles = tmpdir.join('*.out')
        assert len(glob.glob(str(outfiles))) == 0
        vdwradii = tmpdir.join('simple2.rad')
        assert len(glob.glob(str(vdwradii))) == 0
        pdbfiles = tmpdir.join('*.pdb')
        assert len(glob.glob(str(pdbfiles))) == 0
        oldfiles = tmpdir.join('*.old')
        assert len(glob.glob(str(oldfiles))) == 0
        vmd_file = tmpdir.join('hole.vmd')
        assert len(glob.glob(str(vmd_file))) == 1


class TestHoleAnalysisLong(BaseTestHole):

    start = 0
    stop = 11

    def test_gather(self, hole):
        gd = hole.gather(flat=False)
        for i, p in enumerate(hole.profiles.values()):
            assert_almost_equal(p.rxn_coord, gd['rxn_coord'][i])
            assert_almost_equal(p.radius, gd['radius'][i])
            assert_almost_equal(p.cen_line_D, gd['cen_line_D'][i])

    def test_gather_flat(self, hole):
        gd = hole.gather(flat=True)
        i = 0
        for p in hole.profiles.values():
            j = i+len(p.rxn_coord)
            assert_almost_equal(p.rxn_coord, gd['rxn_coord'][i:j])
            assert_almost_equal(p.radius, gd['radius'][i:j])
            assert_almost_equal(p.cen_line_D, gd['cen_line_D'][i:j])
            i = j
        assert_equal(i, len(gd['rxn_coord']))

    def test_min_radius(self, hole):
        rad = hole.min_radius()
        for (f1, p), (f2, r) in zip(hole.profiles.items(), rad):
            assert_equal(f1, f2)
            assert_almost_equal(min(p.radius), r)

    def test_over_order_parameters(self, hole):
        op = np.array([6.10501252e+00, 4.88398472e+00, 3.66303524e+00, 2.44202454e+00,
                       1.22100521e+00, 1.67285541e-07, 1.22100162e+00, 2.44202456e+00,
                       3.66303410e+00, 4.88398478e+00, 6.10502262e+00])
        profiles = hole.over_order_parameters(op, frames=None)
        assert len(op) == len(profiles)

        for key, rmsd in zip(profiles.keys(), np.sort(op)):
            assert key == rmsd

        idx = np.argsort(op)
        arr = np.array(list(hole.profiles.values()))
        for op_prof, arr_prof in zip(profiles.values(), arr[idx]):
            assert op_prof is arr_prof

    def test_over_order_parameters_frames(self, hole):
        op = np.array([6.10501252e+00, 4.88398472e+00, 3.66303524e+00, 2.44202454e+00,
                       1.22100521e+00, 1.67285541e-07, 1.22100162e+00, 2.44202456e+00,
                       3.66303410e+00, 4.88398478e+00, 6.10502262e+00])
        profiles = hole.over_order_parameters(op, frames=np.arange(7))
        assert len(profiles) == 7
        for key, rmsd in zip(profiles.keys(), np.sort(op[:7])):
            assert key == rmsd

        idx = np.argsort(op[:7])
        values = list(hole.profiles.values())[:7]
        arr = np.array(values)
        for op_prof, arr_prof in zip(profiles.values(), arr[idx]):
            assert op_prof is arr_prof

    def test_bin_radii(self, hole):
        radii, bins = hole.bin_radii(bins=100)
        dct = hole.gather(flat=True)
        coords = dct['rxn_coord']
        assert len(bins) == 101
        assert_almost_equal(bins[0], coords.min())
        assert_almost_equal(bins[-1], coords.max())
        assert len(radii) == (len(bins)-1)
        # check first frame profile
        first = hole.profiles[0]
        for row in first:
            coord = row.rxn_coord
            rad = row.radius
            for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
                if coord > lower and coord <= upper:
                    assert rad in radii[i]
                    break
            else:
                raise AssertionError('Radius not in binned radii')

    def test_histogram_radii(self, hole):
        means, _ = hole.histogram_radii(aggregator=np.mean,
                                        bins=100)
        radii, _ = hole.bin_radii(bins=100)
        assert means.shape == (100,)
        for r, m in zip(radii, means):
            assert_almost_equal(r.mean(), m)

@pytest.mark.skipif(executable_not_found("hole"),
                    reason="Test skipped because HOLE not found")
class TestHoleModule(object):
    try:
        # on Unix we can manipulate our limits: http://docs.python.org/2/library/resource.html
        import resource
        soft_max_open_files, hard_max_open_files = resource.getrlimit(
            resource.RLIMIT_NOFILE)
    except ImportError:
        pass

    @staticmethod
    @pytest.fixture()
    def universe():
        return mda.Universe(MULTIPDB_HOLE)

    @pytest.mark.skipif(rlimits_missing,
                        reason="Test skipped because platform does not allow setting rlimits")
    def test_hole_module_fd_closure(self, universe, tmpdir):
        """test open file descriptors are closed (MDAnalysisTests.analysis.test_hole.TestHoleModule): Issue 129"""
        # If Issue 129 isn't resolved, this function will produce an OSError on
        # the system, and cause many other tests to fail as well.
        #
        # Successful test takes ~10 s, failure ~2 s.

        # Hasten failure by setting "ulimit -n 64" (can't go too low because of open modules etc...)
        import resource

        # ----- temporary hack -----
        # on Mac OS X (on Travis) we run out of open file descriptors
        # before even starting this test (see
        # https://github.com/MDAnalysis/mdanalysis/pull/901#issuecomment-231938093);
        # if this issue is solved by #363 then revert the following
        # hack:
        #
        import platform
        if platform.platform() == "Darwin":
            max_open_files = 512
        else:
            max_open_files = 64
        #
        # --------------------------

        resource.setrlimit(resource.RLIMIT_NOFILE,
                           (max_open_files, self.hard_max_open_files))

        with tmpdir.as_cwd():
            try:
                H = hole2.HoleAnalysis(universe, cvect=[0, 1, 0], sample=20.0)
            finally:
                self._restore_rlimits()

            # pretty unlikely that the code will get through 2 rounds if the MDA
            # issue 129 isn't fixed, although this depends on the file descriptor
            # open limit for the machine in question
            try:
                for i in range(2):
                    # will typically get an OSError for too many files being open after
                    # about 2 seconds if issue 129 isn't resolved
                    H.run()
            except OSError as err:
                if err.errno == errno.EMFILE:
                    raise pytest.fail(
                        "hole2.HoleAnalysis does not close file descriptors (Issue 129)")
                raise
            finally:
                # make sure to restore open file limit !!
                self._restore_rlimits()

    def _restore_rlimits(self):
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (self.soft_max_open_files, self.hard_max_open_files))
        except ImportError:
            pass
