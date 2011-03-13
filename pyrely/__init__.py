#!/usr/bin/python
#-*- coding: utf8 -*-

"""
Uncertainty Quantification in python
====================================

pyrely is a Python module implementing state-of-the-art numerical method for
the treatment of uncertainties in computer models.

Alike other toolboxes such as FERUM it is developped for educational
purposes. It thus aims at providing an easy-to-use command-line API, and
a pure high-level pythonic implementation so that existing techniques can
be overloaded and new ones can be added to the toolkit without much
experience in computer programming.

The probabilistic models are based on OpenTURNS, but the initial API has
been overloaded so that it is now fully compatible with other array-based
toolkits (numpy, scipy and others).

See http://pyrely.sourceforge.net for complete documentation.
"""

# Author(s): Vincent Dubourg <vincent.dubourg@gmail.com>
# License: BSD style

from .distributions import *
