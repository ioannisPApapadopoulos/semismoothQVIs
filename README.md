# semismoothQVIs

### Synopsis ###

A firedrake implementation of the semismooth Newton method for obstacle-type quasivariational inequalities as described in

"A globalized inexact semismooth Newton method for nonsmooth fixed point equations involving variational inequalities" (2024), A. Alphonse, C. Christof, M. Hintermüller, I. P. A. Papadopoulos

The implementation trick is to leverage one step of vinewtonrsls in PETSc in order to implement the active-set. This package works in parallel.

### Dependencies and installation ###


The code is written in Python using Firedrake: a finite element solver platform. Firedrake is well documented here: firedrakeproject.org.

First install Firedrake: https://www.firedrakeproject.org/download.html. Make sure to activate the Firedrake venv and run a firedrake-clean!

    source firedrake/firedrake/bin/activate
    firedrake-clean

Then download and pip install the semismoothQVIs library:

    git clone https://github.com/ioannisPApapadopoulos/semismoothQVIs.git
    cd semismoothQVIs/
    pip3 install .
    cd ../

### Contributors ###

Ioannis P. A. Papadopoulos (papadopoulos@wias-berlin.de)


### License ###

GNU LGPL, version 3.
