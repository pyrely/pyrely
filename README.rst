.. -*- mode: rst -*-

About
=====

A python package implementing state-of-the-art numerical methods arising in the
field of uncertainty quantification : from statistical inference to uncertainty
propagation for various purposes such as

   - Reliability analysis: estimation of low probabilities. (TODO)
   - Reliability sensitivity analysis: estimation of probability derivatives.
     (TODO)
   - Sensitivity analysis: quantification and ranking of the different sources
     of uncertainty arising in a model. (TODO)


Download (TODO)
===============

You can download the source code from SourceForge:

http://sourceforge.net/projects/pyrel/files/


Dependencies
============

The required dependencies to build the software are python >= 2.5,
setuptools, NumPy >= 1.1, SciPy >= 0.6, openTURNS >= 0.13,
scikits-learn >= 0.6

To run the tests you will also need nose >= 0.10.


Install (TODO)
==============

This packages uses distutils, which is the default way of installing
python modules. The install command is::

  python setup.py install


Mailing list (TODO)
===================

There's a general and development mailing list, visit
https://lists.sourceforge.net/lists/listinfo/pyrely-general to
subscribe to the mailing list.


Development
===========

Code
----

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com/pyrel/pyrel.git

or if you have write privileges::

    git clone git@github.com:pyrel/pyrel.git


Bugs
----

If you find a bug, you are more than welcome to report it on the mailing list,
and propose a patch in the form of a pull request on Git Hub.


Testing (TODO)
--------------

After installation, or after you added your own contribution please run the
tests from the makefile::

    make test


