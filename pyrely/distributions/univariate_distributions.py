#!/usr/bin/python
#-*- coding: utf8 -*-

"""
Univariate distributions based on OpenTURNS.
"""

# Author(s): Vincent Dubourg <vincent.dubourg@gmail.com>
# License: BSD style

import openturns as ot
import numpy as np
from .ot_compatibility import NumericalSample_to_array, \
                              array_to_NumericalSample, \
                              NumericalPoint_to_array, \
                              array_to_NumericalPoint

_doc_attributes = """
    Attributes
    ----------

    set_parameters: callable
        (Re-) define the parameters of the distribution.

    get_internals: callable
        Internal parameters of the distribution.

    pdf: callable
        Probability density function.

    cdf: callable
        Cumulative density function.

    quantile: callable
        Quantile function (inverse of cdf).

    loglike: callable
        Log-likelihood function.

    rand: callable
        Pseudo-random numbers.

    mean: callable
        Mean of the distribution.

    stdv: callable
        Standard deviation of the distribution.

    skewness: callable
        Skewness of the distribution.

    kurtosis: callable
        Kurtosis of the distribution.
    """


class UnivariateDistribution():
    """Base class for univariate distributions.
    """
    _doc_parameters = """
    Parameters
    ----------

    setting: string, optional
        The setting used to define the distribution, either 'moments' or
        'internals'. Default is 'internals' as in OpenTURNS.

    mean, stdv: floats, if setting == 'moments'
        The mean and standard deviation of the distribution.

    a, b: floats, if setting == 'internals'
        The lower and upper bounds of the uniform distribution such that
        its pdf reads::

            f(x) = 1 / (b - a) if x in [a, b]
                   0 otherwise

    """
    __doc__ += _doc_parameters + _doc_attributes

    _ot_distribution = ot.Distribution(ot.Uniform())
    _settings = ['moments', 'internals']
    _required_internals = ['a', 'b']

    def _compute_internals(self, moments):
        """This function computes the distribution internal parameters from
        its two first moments.
        """

        [mean, stdv] = moments
        internals = {}
        internals['a'] = mean - np.sqrt(3) * stdv
        internals['b'] = mean + np.sqrt(3) * stdv

        return internals

    def __init__(self, name, **kwargs):

        self.__name__ = name
        if len(dict(**kwargs)) != 0:
            self.set_parameters(**kwargs)

    def _check_parameters(self, **kwargs):
        """Check distribution parameters.
        """

        parameters = dict(**kwargs)

        if 'setting' not in parameters.keys():
            parameters['setting'] = 'internals'

        if parameters['setting'] not in self._settings:
            raise(ValueError("Unexpected setting method: %s." \
                  % parameters['setting'] + "Expecting one amongst: %s" \
                  % self._settings))

        missing_args = []
        if parameters['setting'] == 'moments':
            for arg in ['mean', 'stdv']:
                if arg not in parameters.keys():
                    missing_args += [arg]
        else:
            for arg in self._required_internals:
                if arg not in parameters.keys():
                    missing_args += [arg]
        if missing_args != []:
            raise(ValueError("Missing arguments: %s" % missing_args))

        return parameters

    def set_parameters(self, **kwargs):
        """Set distribution parameters.
        """ + self._doc_parameters

        parameters = self._check_parameters(**kwargs)

        if 'shift' in parameters.keys():
            self.shift == parameters['shift']

        if parameters['setting'] == 'moments':
            parameters.update(self.get_internals([parameters['mean'],
                                                  parameters['stdv']]))

        internals = ot.NumericalPoint(len(self._required_internals))
        for k in range(len(self._required_internals)):
            internals[k] = parameters[self._required_internals[k]]
        self._ot_distribution.setParametersCollection(internals)

    def get_internals(self, moments=None):

        if moments == None:
            ot_par = self._ot_distribution.getParametersCollection()[0]
            internals = {}
            for i in range(ot_par.getDimension()):
                internals[self._internals[i]] = ot_par[i]
        else:
            internals = self._compute_internals(moments)

        return internals

    def pdf(self, x):

        x = np.asanyarray(x)
        f = self._ot_distribution.computePDF(array_to_NumericalSample(x))

        return NumericalSample_to_array(f).reshape(x.shape)

    def cdf(self, x):

        x = np.asanyarray(x)
        F = self._ot_distribution.computeCDF(array_to_NumericalSample(x))

        return NumericalSample_to_array(F).reshape(x.shape)

    def quantile(self, p):

        p = np.asanyarray(p)
        # computeQuantile functions only accept NumericalScalar as args...
        q = np.zeros(p.size)
        for i in range(p.size):
            q[i] = self._ot_distribution.computeQuantile(p[i])[0]

        return q.reshape(p.shape)

    def mean(self):

        return self._ot_distribution.getMean()[0]

    def stdv(self):

        return self._ot_distribution.getStandardDeviation()[0]

    def skewness(self):

        return self._ot_distribution.getSkewness()[0]

    def kurtosis(self):

        return self._ot_distribution.getKurtosis()[0]

    def loglike(self, values):

        return np.log(self.pdf(values)).sum()

    def rand(self, size=1):

        r = self._ot_distribution.getNumericalSample(size)
        r = NumericalSample_to_array(r)
        if r.size == 1:
            r = r.ravel()[0]

        return r


class Uniform(UnivariateDistribution):
    """Uniform distribution.
    """
    _doc_parameters = """
    Parameters
    ----------

    setting: string, optional
        The setting used to define the distribution, either 'moments' or
        'internals'. Default is 'internals' as in OpenTURNS.

    mean, stdv: floats, if setting == 'moments'
        The mean and standard deviation of the distribution.

    a, b: floats, if setting == 'internals'
        The lower and upper bounds of the uniform distribution such that
        its pdf reads::

            f(x) = 1 / (b - a) if x in [a, b]
                   0 otherwise

    """
    __doc__ += _doc_parameters + _doc_attributes

    _ot_distribution = ot.Distribution(ot.Uniform())
    _settings = ['moments', 'internals']
    _required_internals = ['a', 'b']

    def _compute_internals(self, moments):
        """This function computes the distribution internal parameters from
        its two first moments.
        """

        [mean, stdv] = moments
        internals = {}
        internals['a'] = mean - np.sqrt(3) * stdv
        internals['b'] = mean + np.sqrt(3) * stdv

        return internals


class Normal(UnivariateDistribution):
    """Normal distribution.
    """
    _doc_parameters = """
    Parameters
    ----------

    setting: string, optional
        The setting used to define the distribution, either 'moments' or
        'internals'. Default is 'internals' as in OpenTURNS.

    mean, stdv: floats, if setting == 'moments'
        The mean and standard deviation of the distribution.

    mu, sigma: floats, if setting == 'internals'
        The mean and standard deviation of the normal distribution such that
        its pdf reads::

            f(x) = 1 / sigma * phi((x - mu) / sigma)

        where phi denotes the standard normal pdf::

            phi(u) = 1 / sqrt(2 * pi) * exp( - u ** 2 / 2)
    """
    __doc__ += _doc_parameters + _doc_attributes

    _ot_distribution = ot.Distribution(ot.Normal())
    _settings = ['moments', 'internals']
    _required_internals = ['mu', 'sigma']

    def _compute_internals(self, moments):
        """This function computes the distribution internal parameters from
        its two first moments.
        """

        [mean, stdv] = moments
        internals = {}
        internals['mu'] = mean
        internals['sigma'] = stdv

        return internals


class Lognormal(UnivariateDistribution):
    """Lognormal distribution.
    """
    _doc_parameters = """
    Parameters
    ----------

    setting: string, optional
        The setting used to define the distribution, either 'moments' or
        'internals'. Default is 'internals' as in OpenTURNS.

    mean, stdv: floats, if setting == 'moments'
        The mean and standard deviation of the distribution.

    LAMBDA, zeta: floats, if setting == 'internals'
        The mean and standard deviation of the lognormal distribution such that
        its pdf reads::

            f(x) = 1 / zeta * phi((log(x - shift) - LAMBDA) / zeta)

        where phi denotes the standard normal pdf::

            phi(u) = 1 / sqrt(2 * pi) * exp( - u ** 2 / 2)

    shift: float, optional
        An offset for the distribution.
    """
    __doc__ += _doc_parameters + _doc_attributes

    _ot_distribution = ot.Distribution(ot.LogNormal())
    _settings = ['moments', 'internals']
    _required_internals = ['LAMBDA', 'zeta']

    def _compute_internals(self, moments):
        """This function computes the distribution internal parameters from
        its two first moments.
        """

        [mean, stdv] = moments
        cov = stdv / mean
        zeta = np.sqrt(np.log(1. + cov ** 2.))
        LAMBDA = np.log(mean) - 0.5 * zeta ** 2.
        internals = {}
        internals['LAMBDA'] = LAMBDA
        internals['zeta'] = zeta

        return internals


class Gamma(UnivariateDistribution):
    """Gamma distribution.
    """
    _doc_parameters = """
    Parameters
    ----------

    setting: string, optional
        The setting used to define the distribution, either 'moments' or
        'internals'. Default is 'internals' as in OpenTURNS.

    mean, stdv: floats, if setting == 'moments'
        The mean and standard deviation of the distribution.

    k, LAMBDA: floats, if setting == 'internals'
        The mean and standard deviation of the gamma distribution such that
        its pdf reads::

            f(x) = TODO

    shift: float, optional
        An offset for the distribution.
    """
    __doc__ += _doc_parameters + _doc_attributes

    _ot_distribution = ot.Distribution(ot.Gamma())
    _settings = ['moments', 'internals']
    _required_internals = ['k', 'LAMBDA']

    def _compute_internals(self, moments):
        """This function computes the distribution internal parameters from
        its two first moments.
        """

        [mean, stdv] = moments
        internals = {}
        internals['k'] = mean ** 2. / stdv ** 2.
        internals['LAMBDA'] = mean / stdv ** 2.

        return internals
