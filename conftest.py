#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Obspy's testing configuration file.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import platform
import sys
import time

import numpy as np
import pytest

from obspy.core.util.version import get_git_version
from obspy.core.util import NETWORK_MODULES
import obspy.core.util.testing as otest
# from obspy.core.util.testing import _create_report, DEFAULT_TEST_SERVER, _send_report


PSTATS_HELP = """
Call "python -m pstats obspy.pstats" for an interactive profiling session.

The following commands will produce the same output as shown above:
  sort cumulative
  stats obspy. 20

Type "help" to see all available options.
"""

# Set legacy printing for numpy so the doctests work regardless of the numpy
# version.
try:
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass

HOSTNAME = platform.node().split('.', 1)[0]


@pytest.fixture(scope='session', autouse=True)
def set_numpy_print_options():
    """
    Make sure the doctests print the same style of output across all numpy
    versions.
    """
    try:
        np.set_printoptions(legacy='1.13')
    except (TypeError, AttributeError):
        pass


def pytest_addoption(parser):
    parser.addoption('--obspy-version', action='store_true', default=False,
                     help='print obspy version and exit')
    parser.addoption('--network', action='store_true', default=False,
                     help='test network modules', )
    # reporting options
    report = parser.getgroup('Reporting Options')
    report.addoption('--report', action='store_true', default=False,
                     help='automatically submit a test report')
    report.addoption('--server', default=otest.DEFAULT_TEST_SERVER,
                     help='report server (default is tests.obspy.org)')
    report.addoption('--node', dest='hostname', default=HOSTNAME,
                     help='nodename visible at the report server')
    report.addoption('--log', default=None,
                     help='append log file to test report')
    report.addoption('--ci-url', default=None, dest="ci_url",
                     help='URL to Continuous Integration job page.')
    report.addoption('--pr-url', default=None,
                     dest="pr_url", help='Github (Pull Request) URL.')
    # other options
    others = parser.getgroup('Additional Options')
    others.addoption('--tutorial', action='store_true',
                     help='add doctests in tutorial')
    others.addoption('--no-formatting', action='store_true',
                     help='skip code formatting test')
    others.addoption('--keep-images', action='store_true',
                     help='store images created during image comparison '
                          'tests in subfolders of baseline images')
    others.addoption('--keep-only-failed-images', action='store_true',
                     help='when storing images created during testing, '
                          'only store failed images and the corresponding '
                          'diff images (but not images that passed the '
                          'corresponding test).')


def pytest_collection_modifyitems(config, items):
    """ Preprocessor for collected tests. """
    network_nodes = set(NETWORK_MODULES)
    for item in items:
        # get the obspy model test originates from (eg clients.arclink)
        obspy_node = '.'.join(item.nodeid.split('/')[1:3])
        # if test is a network test apply network marker
        if obspy_node in network_nodes:
            item.add_marker(pytest.mark.network)


def pytest_configure(config):
    # if obspy version just print it and exit
    if config.getoption('--obspy-version'):
        print(get_git_version())
        sys.exit(0)
    # If the all option is not set skip all network tests
    if not config.getoption('--network'):
        setattr(config.option, 'markexpr', 'not network')


def pytest_sessionstart(session):
    """ Add a results dict  to session object. """
    # add results dict and start time to session object
    session._results = dict()
    session._starttime = time.time()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item):
    """
    Hook to append each test result to pytest session.

    See this stack overflow for the basic idea: https://goo.gl/hRCPSv
    """
    outcome = yield
    result = outcome.get_result()
    if result.when == 'call':
        item.session._results[result.nodeid] = result


def pytest_sessionfinish(session, exitstatus):
    """ Hook called when all tests runs finish. """
    if session.config.getoption('--report'):
        results, params = otest._create_report(session)
        breakpoint()
        otest._send_report(session, params)
