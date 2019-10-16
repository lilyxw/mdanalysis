.. Contains the formatted docstrings for the core modules located in 'mdanalysis/MDAnalysis/core'
.. currentmodule:: MDAnalysis

**************************
Core modules
**************************

The :mod:`MDAnalysis.core` modules contain functionality essential for
MDAnalysis, such as the central data structures in
:mod:`MDAnalysis.core.universe` and :mod:`MDAnalysis.core.groups` or
the selection definitions and parsing in
:mod:`MDAnalysis.core.selection`.

Constructing
============

.. autosummary::

  :toctree: generated/

  Universe
  Merge

Attributes 
==========

.. autosummary::

  :toctree: generated/

  AtomGroup
  ResidueGroup

Topology system
===============

The topology system is primarily of interest to developers.

.. toctree::
   :maxdepth: 1

   core/topology
   core/topologyobjects
   core/topologyattrs

.. SeeAlso:: :ref:`Developer notes for Topology
             Parsers <topology-parsers-developer-notes>`

Selection system
================

The selection system is primarily of interest to developers.

.. toctree::
   :maxdepth: 1

   core/selection

Flag system
============

The flag system contains the global behavior of MDAnalysis. It is
normally not necessary to change anything here.

.. toctree::
   :maxdepth: 1

   core/init
