# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import sys
import os
import re
import glob


version = re.findall('__version__ = "(.*)"',
                     open('semismoothQVIs/__init__.py', 'r').read())[0]

packages = [
    "semismoothQVIs",
    ]

CLASSIFIERS = """
Development Status :: 2 - Pre-Alpha
Environment :: Console
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Programming Language :: Python
Topic :: Scientific/Engineering :: Mathematics
"""
classifiers = CLASSIFIERS.split('\n')[1:-1]

# TODO: This is cumbersome and prone to omit something
demofiles = glob.glob(os.path.join("examples", "*", "*.py"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*.py"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*.xml*"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*", "*.geo"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*", "*.xml*"))

# Don't bother user with test files
[demofiles.remove(f) for f in demofiles if "test_" in f]

setup(name="semismoothQVIs",
      version=version,
      author="Ioannis Papadopoulos",
      author_email="papadopoulos@wias-berlin.de",
      url="https://github.com/ioannisPApapadopoulos/semismoothQVIs",
      description="semismoothQVIs",
      long_description="--",
      classifiers=classifiers,
      license="GNU LGPL v3 or later",
      packages=packages,
      package_dir={"semismoothQVIs": "semismoothQVIs"},
      data_files=[(os.path.join("share", "semismoothQVIs", os.path.dirname(f)), [f])
                  for f in demofiles],
    )