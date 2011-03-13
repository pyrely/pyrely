#!/usr/bin/python
#-*- coding: utf8 -*-

"""
Compatibility functions for OpenTURNS and Numpy.
"""

# Author(s): Vincent Dubourg <vincent.dubourg@gmail.com>
# License: BSD style

import openturns as ot
import numpy as np


def NumericalSample_to_array(NS):
    """This function converts a NumericalSample from OpenTURNS into a two-
    dimensional array from numpy.

    Parameters
    ----------
    NS: ot.NumericalSample
        A NumericalSample.

    Returns
    -------
    a: np.atleast_2d
        Two-dimensional array.
    """
    a = np.zeros((NS.getSize(), NS.getDimension()))
    for i in range(NS.getSize()):
        for j in range(NS.getDimension()):
            a[i, j] = NS[i][j]
    return a


def array_to_NumericalSample(a):
    """This function converts an array from numpy into a NumericalSample from
    OpenTURNS.

    Parameters
    ----------
    a: np.array
        An array.

    Returns
    -------
    NS: ot.NumericalSample
        A NumericalSample.
    """
    a = np.array(a)
    if a.ndim <= 1:
        a = np.atleast_2d(a).T
    NS = ot.NumericalSample(1, ot.NumericalPoint(a[0].tolist()))
    for i in range(a.shape[0] - 1):
        NS.add(ot.NumericalPoint(a[i + 1].tolist()))
    return NS


def NumericalPoint_to_array(NP):
    """This function converts a NumericalPoint from OpenTURNS into a one-
    dimensional array from numpy.

    Parameters
    ----------
    NP: ot.NumericalPoint
        A NumericalPoint.

    Returns
    -------
    a: np.array
        One-dimensional array.
    """
    a = np.zeros(NP.getDimension())
    for i in range(NP.getDimension()):
        a[i] = NP[i]
    return a


def array_to_NumericalPoint(a):
    """This function converts a one-dimensional array from numpy into a
    NumericalPoint from OpenTURNS.

    Parameters
    ----------
    a: np.array
        One-dimensional array (will be flattened anyway).

    Returns
    -------
    NP: ot.NumericalPoint
        A NumericalPoint.
    """
    a = np.ravel(a)
    NP = ot.NumericalPoint(a.size)
    for i in range(NP.getDimension()):
        NP[i] = a[i]
    return NP
