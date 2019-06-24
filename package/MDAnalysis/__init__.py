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
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#

"""
:mod:`MDAnalysis` --- analysis of molecular simulations in python
=================================================================

MDAnalysis (https://www.mdanalysis.org) is a python toolkit to analyze
molecular dynamics trajectories generated by CHARMM, NAMD, Amber,
Gromacs, or LAMMPS.

It allows one to read molecular dynamics trajectories and access the
atomic coordinates through numpy arrays. This provides a flexible and
relatively fast framework for complex analysis tasks. In addition,
CHARMM-style atom selection commands are implemented. Trajectories can
also be manipulated (for instance, fit to a reference structure) and
written out. Time-critical code is written in C for speed.

Help is also available through the mailinglist at
http://groups.google.com/group/mdnalysis-discussion

Please report bugs and feature requests through the issue tracker at
http://issues.mdanalysis.org

Citation
--------

When using MDAnalysis in published work, please cite

    N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and
    O. Beckstein. MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics
    Simulations. J. Comput. Chem. 32 (2011), 2319--2327, doi:`10.1002/jcc.21787`_
    https://www.mdanalysis.org

For citations of included algorithms and sub-modules please see the references_.

.. _`10.1002/jcc.21787`: http://dx.doi.org/10.1002/jcc.21787
.. _references: https://docs.mdanalysis.org/documentation_pages/references.html


Getting started
---------------

Import the package::

  >>> import MDAnalysis

(note that not everything in MDAnalysis is imported right away; for
additional functionality you might have to import sub-modules
separately, e.g. for RMS fitting ``import MDAnalysis.analysis.align``.)

Build a "universe" from a topology (PSF, PDB) and a trajectory (DCD, XTC/TRR);
here we are assuming that PSF, DCD, etc contain file names. If you don't have
trajectories at hand you can play with the ones that come with MDAnalysis for
testing (see below under `Examples`_)::

  >>> u = MDAnalysis.Universe.from_files(PSF, DCD)

Select the C-alpha atoms and store them as a group of atoms::

  >>> ca = u.select_atoms('name CA')
  >>> len(ca)
  214

Calculate the centre of mass of the CA and of all atoms::

  >>> ca.center_of_mass()
  array([ 0.06873595, -0.04605918, -0.24643682])
  >>> u.atoms.center_of_mass()
  array([-0.01094035,  0.05727601, -0.12885778])

Calculate the CA end-to-end distance (in angstroem)::
  >>> import numpy as np
  >>> coord = ca.positions
  >>> v = coord[-1] - coord[0]   # last Ca minus first one
  >>> np.sqrt(np.dot(v, v,))
  10.938133

Define a function eedist():
  >>> def eedist(atoms):
  ...     coord = atoms.positions
  ...     v = coord[-1] - coord[0]
  ...     return sqrt(dot(v, v,))
  ...
  >>> eedist(ca)
  10.938133

and analyze all timesteps *ts* of the trajectory::
  >>> for ts in u.trajectory:
  ...      print eedist(ca)
  10.9381
  10.8459
  10.4141
   9.72062
  ....

See Also
--------
:class:`MDAnalysis.core.universe.Universe` for details


Examples
--------

MDAnalysis comes with a number of real trajectories for testing. You
can also use them to explore the functionality and ensure that
everything is working properly::

  from MDAnalysis import *
  from MDAnalysis.tests.datafiles import PSF,DCD, PDB,XTC
  u_dims_adk = Universe.from_files(PSF,DCD)
  u_eq_adk = Universe.from_files(PDB, XTC)

The PSF and DCD file are a closed-form-to-open-form transition of
Adenylate Kinase (from [Beckstein2009]_) and the PDB+XTC file are ten
frames from a Gromacs simulation of AdK solvated in TIP4P water with
the OPLS/AA force field.

.. [Beckstein2009] O. Beckstein, E.J. Denning, J.R. Perilla and T.B. Woolf,
   Zipping and Unzipping of Adenylate Kinase: Atomistic Insights into the
   Ensemble of Open <--> Closed Transitions. J Mol Biol 394 (2009), 160--176,
   doi:10.1016/j.jmb.2009.09.009

"""
from __future__ import absolute_import

__all__ = ['Universe', 'as_Universe', 'Writer', 'fetch_mmtf',
           'AtomGroup', 'ResidueGroup', 'SegmentGroup']

import logging
import warnings

logger = logging.getLogger("MDAnalysis.__init__")

from .version import __version__
try:
    from .authors import __authors__
except ImportError:
    logger.info('Could not find authors.py, __authors__ will be empty.')
    __authors__ = []

# Registry of Readers, Parsers and Writers known to MDAnalysis
# Metaclass magic fills these as classes are declared.
_READERS = {}
_SINGLEFRAME_WRITERS = {}
_MULTIFRAME_WRITERS = {}
_PARSERS = {}
_SELECTION_WRITERS = {}
# Registry of TopologyAttributes
_TOPOLOGY_ATTRS = {}

# Storing anchor universes for unpickling groups
import weakref
_ANCHOR_UNIVERSES = weakref.WeakValueDictionary()
del weakref

# custom exceptions and warnings
from .exceptions import (
    SelectionError, FinishTimeException, NoDataError, ApplicationError,
    SelectionWarning, MissingDataWarning, ConversionWarning, FileFormatWarning,
    StreamWarning
)

from .lib import log
from .lib.log import start_logging, stop_logging

logging.getLogger("MDAnalysis").addHandler(log.NullHandler())
del logging

# only MDAnalysis DeprecationWarnings are loud by default
warnings.filterwarnings(action='once', category=DeprecationWarning,
                        module='MDAnalysis')


from . import units

# Bring some often used objects into the current namespace
from .core.universe import Universe, as_Universe, Merge
from .core.groups import AtomGroup, ResidueGroup, SegmentGroup
from .coordinates.core import writer as Writer

# After Universe import
from .coordinates.MMTF import fetch_mmtf

from .migration.ten2eleven import ten2eleven

from .due import due, Doi, BibTeX

due.cite(BibTeX((
            "@inproceedings{gowers2016, "
            "title={MDAnalysis: A Python package for the rapid analysis "
            "of molecular dynamics simulations}, "
            "author={R. J. Gowers and M. Linke and "
            "J. Barnoud and T. J. E. Reddy and M. N. Melo "
            "and S. L. Seyler and D. L. Dotson and J. Domanski, and "
            "S. Buchoux and I. M. Kenney and O. Beckstein},"
            "journal={Proceedings of the 15th Python in Science Conference}, "
            "pages={102-109}, "
            "year={2016}, "
            "editor={S. Benthall and S. Rostrup}, "
            "note={Austin, TX, SciPy.} "
            "}"
            )),
         description="Molecular simulation analysis library",
         path="MDAnalysis", cite_module=True)
due.cite(Doi("10.1002/jcc.21787"),
         description="Molecular simulation analysis library",
         path="MDAnalysis", cite_module=True)

del Doi, BibTeX
