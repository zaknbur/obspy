"""
Testing utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import os
import platform
import shlex
import sys
import time
import warnings
from argparse import ArgumentParser
from collections import defaultdict, Counter

import numpy as np
import pytest
from lxml import etree

import obspy
from obspy.core.util.base import NamedTemporaryFile, MATPLOTLIB_VERSION
from obspy.core.util.misc import MatplotlibBackend
from obspy.core.util.version import get_git_version

# The default url to upload test resports.
DEFAULT_TEST_SERVER = 'tests.obspy.org'

# this dictionary contains the locations of checker routines that determine
# whether the module's tests can be executed or not (e.g. because test server
# is unreachable, necessary ports are blocked, etc.).
# A checker routine should return either an empty string (tests can and will
# be executed) or a message explaining why tests can not be executed (all
# tests of corresponding module will be skipped).
MODULE_TEST_SKIP_CHECKS = {
    'clients.seishub':
        'obspy.clients.seishub.tests.test_client._check_server_availability'}


try:
    OBSPY_PATH = obspy.__path__[0]
except (AttributeError, IndexError):
    OBSPY_PATH = ''


def _create_report(session):
    """
    If `server` is specified without URL scheme, 'https://' will be used as a
    default.
    """
    # import additional libraries here to speed up normal tests
    from future import standard_library
    with standard_library.hooks():
        import urllib.parse

    from obspy.core.util.version import get_git_version
    from xml.etree import ElementTree
    from obspy.core.compatibility import urlparse
    import requests
    import obspy.core.util.base as obase
    # from obspy.core.base import ALL_MODULES, NETWORK_MODULES
    from xml.sax.saxutils import escape

    def _read_log(path):
        """ Try to read the system log, else return None. """
        import codecs
        try:
            data = codecs.open(path, 'r', encoding='UTF-8').read()
        except Exception:
            print("Cannot open log file %s" % log)
        else:
            return escape(data)

    def _get_module_name(nodeid):
        """ Convert a pytest nodeid to an obspy model. """
        return '.'.join(nodeid.split('/')[1:3]).replace('.tests', '')

    def _get_tested_module_dict(results):
        """ Return a dict of {tested_module: [test_results, ]} """
        out = defaultdict(list)
        for name, result in results.items():
            out[_get_module_name(name)].append(result)
        return out

    def _split_node(nodeid):
        """ split a node into test_module, test_class, test_name """
        test_module_name = nodeid.split('::')[0].replace('/', '.')
        nodes = nodeid.split('/')[-1].split('::')[1:]
        if len(nodes) == 1:  # if this is a test outside of class
            test_name = nodes[-1]
            cls_name = ''
        else:
            test_name = nodes[-1]
            cls_name = nodes[0]
        return test_module_name.replace('.py', ''), cls_name, test_name

    def _count_outcomes(ttrs):
        """ Count the outcomes of test results. """
        # map pytest outcomes to names expected in report
        oc_map = dict(failed='failures', errored='errors', skipped='skipped',
                      passed='passes')
        count = Counter(oc_map[x.outcome] for x in ttrs)
        count['tests'] = len(ttrs)
        # init empty counts so counter can be used to update dict later on
        for outcome in oc_map.values():
            if outcome not in count:
                count[outcome] = 0
        count.pop('passes', None)
        return count

    def _dict2xml(result, doc=None):
        """ Generate xml document to send to server. """
        doc = doc or ElementTree.Element("report")
        for key, value in result.items():
            key = key.split('(')[0].strip()
            if isinstance(value, dict):
                child = ElementTree.SubElement(doc, key)
                _dict2xml(value, child)
            elif value is not None:
                if isinstance(value, (str, native_str)):
                    ElementTree.SubElement(doc, key).text = value
                elif isinstance(value, (str, native_str)):
                    ElementTree.SubElement(doc, key).text = str(value, 'utf-8')
                else:
                    ElementTree.SubElement(doc, key).text = str(value)
            else:
                ElementTree.SubElement(doc, key)
        return ElementTree.tostring(doc)

    def _get_module_info(module, ttrs):
        """ Create a dict of results for a given module. """

        out = {'installed': installed}
        # TODO get modules that fail to import...
        # add a failed-to-import test module to report with an error
        # if module in import_failures:
        #     result['obspy'][module]['timetaken'] = 0
        #     result['obspy'][module]['tested'] = True
        #     result['obspy'][module]['tests'] = 1
        #     # can't say how many tests would have been in that suite so just
        #     # leave 0
        #     result['obspy'][module]['skipped'] = 0
        #     result['obspy'][module]['failures'] = {}
        #     result['obspy'][module]['errors'] = {
        #         'f%s' % (errors): import_failures[module]}
        #     tests += 1
        #     errors += 1
        #     continue
        if module not in ttrs:
            return out
        test_results = ttrs[module]  # a list of test results for module
        # count tests errors, runs, failures, skips
        outcomes = _count_outcomes(test_results)
        # test results
        out['timetaken'] = sum(x.duration for x in test_results)
        out['tested'] = True
        out['tests'] = outcomes['tests']
        out['skipped'] = outcomes['skipped']
        # depending on module type either use failure (network related modules)
        # or errors (all others)
        out['errors'] = {}
        out['failures'] = {}
        count = 0
        for result in test_results:
            # TODO figure out how to tell if failure or error (errors have
            # outcome == 'failed)
            if result.outcome == 'failed':
                out['failures']['f%d' % count] = result.longreprtext
                count += 1
        return out

    def _get_dependency_versions():
        """ Return a dict of versions of obspy dependencies. """
        out = {}
        for module in obase.DEPENDENCIES:
            if module == "pep8-naming":
                module_ = "pep8ext_naming"
            else:
                module_ = module
            temp = module_.split('.')
            try:
                mod = __import__(module_,
                                 fromlist=[native_str(temp[1:])])
            except ImportError:
                version_ = '---'
            else:
                try:
                    version_ = mod.__version__
                except AttributeError:
                    version_ = '???'
            out[module] = version_
        return out

    def _get_system_info(hostname):
        """ Get system information. """
        out = {}
        for func in ['system', 'release', 'version', 'machine',
                     'processor', 'python_version', 'python_implementation',
                     'python_compiler', 'architecture']:
            try:
                temp = getattr(platform, func)()
                if isinstance(temp, tuple):
                    temp = temp[0]
                out[func] = temp
            except Exception:
                out[func] = ''
        # set node name to hostname if set
        out['node'] = hostname
        # post only the first part of the node name (only applies to MacOS X)
        try:
            out['node'] = result['platform']['node'].split('.')[0]
        except Exception:
            pass
        return out

    def _get_skipped_test_details(test_results):
        """ Append message for each skipped test. """
        out = []
        for module in test_results:
            for test_result in test_results[module]:
                if test_result.outcome != 'skipped':
                    continue
                module = _get_module_name(test_result.nodeid)
                test_mod, cls, test_name = _split_node(test_result.nodeid)
                reason = test_result.longrepr[-1].replace('Skipped: ', '')
                out.append((module, test_mod, cls, test_name, reason))
        return out

    def _get_slowest_tests(results):
        """ Return a tuple of slowests tests and their runtimes. """
        out = []
        # get a dict of {test_node: duration}
        test_durations = {i: v.duration for i, v in results.items()}
        # get a list of test_nodes sorted by duration (fastest first)
        nodes_by_time = sorted(test_durations, key=lambda x: test_durations[x])
        for node in nodes_by_time[:-20:-1]:
            time_str = "%0.3fs" % test_durations[node]
            nodesplit = node.split('::')
            test_name = nodesplit[-1]
            other = '.'.join(nodesplit[:-1]).replace('/', '.')
            test_id_str = "%s (%s)" % (test_name, other)
            out.append((time_str, test_id_str))
        return out

    # get command line arg info used in the report
    hostname = session.config.getoption('--node')
    ci_url = session.config.getoption('--ci-url')
    pr_url = session.config.getoption('--pr-url')
    log = session.config.getoption('--log')
    # get time taken for entire test suite to run
    timestamp = time.time()
    timetaken = timestamp - session._starttime
    # get dict of {module: [test_result,]}
    ttrs = _get_tested_module_dict(session._results)
    import_failures = []
    # get obspy version
    try:
        installed = get_git_version()
    except Exception:
        installed = ''
    # init result dict and add basics to it
    result = {'timestamp': int(timestamp), 'timetaken': timetaken, }
    if log:
        result['install_log'] = _read_log(log)
    # add slowest tests to result
    result['slowest_tests'] = _get_slowest_tests(session._results)

    # add extra urls
    if ci_url is not None:
        result['ciurl'] = ci_url
    if pr_url is not None:
        result['prurl'] = pr_url
    result['obspy'] = {'installed': installed}
    # init a dict with ints for counting test outcomes
    outcome_count = _count_outcomes(list(session._results.values()))
    # iterate each module and add results to dict, also increment outcomes
    for module in sorted(obase.ALL_MODULES):
        info = _get_module_info(module, ttrs)
        result['obspy'][module] = info
    # add package versions to report
    result['dependencies'] = _get_dependency_versions()
    # add system / environment settings to report
    result['platform'] = _get_system_info(hostname)
    # update outcome counts
    result.update(outcome_count)
    # append info on skipped tests:
    result['skipped_tests_details'] = _get_skipped_test_details(ttrs)
    # generate params to send to sever
    params = urllib.parse.urlencode({
        'timestamp': timestamp,
        'system': result['platform']['system'],
        'python_version': result['platform']['python_version'],
        'architecture': result['platform']['architecture'],
        'tests': outcome_count['tests'],
        'failures': outcome_count['failures'],
        'errors': outcome_count['errors'],
        'modules': len(ttrs) + len(import_failures),
        'xml': _dict2xml(result)
    })
    return result, params


def _send_report(session, params):
    """ Send the test report to the server. """

    from future import standard_library

    from obspy.core.compatibility import urlparse
    import requests

    # get command line arg info used in the report
    server = session.config.getoption('--server')

    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain"}
    url = server
    if not urlparse(url).scheme:
        url = "https://" + url
    response = requests.post(url=url, headers=headers,
                             data=params.encode('UTF-8'))
    # get the response
    if response.status_code == 200:
        report_url = response.json().get('url', server)
        print('Your test results have been reported and are available at: '
              '{}\nThank you!'.format(report_url))
    # handle errors
    else:
        print("Error: Could not send a test report to %s." % (server))
        print(response.reason)


def _configure_parser():
    """
    Configure the command line parser.
    """
    from obspy.core.util.version import get_git_version
    hostname = platform.node().split('.', 1)[0]

    parser = ArgumentParser(prog='obspy-runtests',
                            description='A command-line program that runs all '
                                        'ObsPy tests.')
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + get_git_version())
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose mode')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='quiet mode')
    parser.add_argument('--raise-all-warnings', action='store_true',
                        help='All warnings are raised as exceptions when this '
                             'flag is set. Only for debugging purposes.')

    # filter options
    filter = parser.add_argument_group('Module Filter',
                                       'Providing no modules will test all '
                                       'ObsPy modules which do not require an '
                                       'active network connection.')
    filter.add_argument('-a', '--all', action='store_true',
                        dest='test_all_modules',
                        help='test all modules (including network modules)')
    filter.add_argument('-x', '--exclude', action='append',
                        help='exclude given module from test')
    filter.add_argument('tests', nargs='*',
                        help='test modules to run')

    # timing / profile options
    timing = parser.add_argument_group('Timing/Profile Options')
    timing.add_argument('-t', '--timeit', action='store_true',
                        help='shows accumulated run times of each module')
    timing.add_argument('-s', '--slowest', default=0, type=int, dest='n',
                        help='lists n slowest test cases')
    timing.add_argument('-p', '--profile', action='store_true',
                        help='uses cProfile, saves the results to file ' +
                             'obspy.pstats and prints some profiling numbers')

    # reporting options
    report = parser.add_argument_group('Reporting Options')
    report.add_argument('-r', '--report', action='store_true',
                        help='automatically submit a test report')
    report.add_argument('-d', '--dontask', action='store_true',
                        help="don't explicitly ask for submitting a test "
                             "report")
    report.add_argument('-u', '--server', default=DEFAULT_TEST_SERVER,
                        help='report server (default is tests.obspy.org)')
    report.add_argument('-n', '--node', dest='hostname', default=hostname,
                        help='nodename visible at the report server')
    report.add_argument('-l', '--log', default=None,
                        help='append log file to test report')
    report.add_argument('--ci-url', default=None, dest="ci_url",
                        help='URL to Continuous Integration job page.')
    report.add_argument('--pr-url', default=None,
                        dest="pr_url", help='Github (Pull Request) URL.')

    # other options
    others = parser.add_argument_group('Additional Options')
    others.add_argument('--tutorial', action='store_true',
                        help='add doctests in tutorial')
    others.add_argument('--no-flake8', action='store_true',
                        help='skip code formatting test')
    others.add_argument('--keep-images', action='store_true',
                        help='store images created during image comparison '
                             'tests in subfolders of baseline images')
    others.add_argument('--keep-only-failed-images', action='store_true',
                        help='when storing images created during testing, '
                             'only store failed images and the corresponding '
                             'diff images (but not images that passed the '
                             'corresponding test).')
    return parser


def _obspy_to_pytest_nodeid(obspy_str):
    """
    Convert an obspy test node id to a pytest nodeid.

    Eg; io.mseed -> obspy/io/mseed/tests
    """
    # get base path
    path = []
    # iterate each node in obspy_str, try to determine where file ends
    # and test cases start
    for node in obspy_str.split('.'):
        # if the current node is a directory append to path
        if os.path.isdir(os.path.join(*(path + [node]))):
            path.append(node)
        # if the current node is a python file
        elif os.path.exists(os.path.join(*(path + [node + '.py']))):
            path.append(node + '.py')
        # else it may be a test case, add :: and append to path
        else:
            path[-1] = path[-1] + '::' + node
    return os.path.join(*path)


def _convert_to_pytest_input(args, input_args=None):
    """
    Translate the parsed obspy-runtest arguments into pytest input.
    """
    out = []  # init a list for storing output

    # if version is specified just bring obspy version and exit
    if getattr(args, 'version', None):
        print(get_git_version())
        sys.exit(0)
    # handle setting verbosity and warning filters
    if args.verbose:
        out.append(['--verbose %d' % args.verbose])
    elif args.quiet:
        out.append('--quiet')
        # TODO test if this actually suppresses warnings in pytest runner
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
    else:  # Leave
        # show all NumPy warnings
        np.seterr(all='print')
        # ignore user warnings
        warnings.simplefilter("ignore", UserWarning)
    # handle filtering options
    if args.raise_all_warnings:  # raise all warnings as errors
        np.seterr(all='raise')
        out.append('--Werror')
    if args.test_all_modules:
        out.append('--all')
    for exclude in args.exclude or []:
        out.append('--exclude %s' % _obspy_to_pytest_nodeid(exclude))

    # handle timing options
    # TODO figure out timeit and profile
    if getattr(args, 'slowest', None) is not None:
        out.append('--durations %d' % args.slowest)

    # handle reporting options
    if args.report or 'OBSPY_REPORT' in os.environ.keys():
        out.append('--report')
    if args.server or 'OBSPY_REPORT_SERVER' in os.environ.keys():
        server = os.environ.get('OBSPY_REPORT_SERVER') or args.server
        if server != DEFAULT_TEST_SERVER:
            out.append('--server %s' % server)
    if getattr(args, 'node', None):
        out.append('--node %s' % args.node)
    if getattr(args, 'log', None):
        out.append('--log')
    if getattr(args, 'ci_url', None):
        out.append('--ci-url %s' % args.ci_url)
    if getattr(args, 'pr_url', None):
        out.append('--pr-url %s' % args.pr_url)

    # TODO figure out what to do with these
    # if args.keep_images:
    #     os.environ['OBSPY_KEEP_IMAGES'] = ""
    # if args.keep_only_failed_images:
    #     os.environ['OBSPY_KEEP_ONLY_FAILED_IMAGES'] = ""
    # if args.no_flake8:
    #     os.environ['OBSPY_NO_FLAKE8'] = ""

    return out


def run_tests(argv=None, interactive=True):
    """
    Run the obspy test suite with pytest. Return pytest's exit code.
    """
    MatplotlibBackend.switch_backend("AGG", sloppy=False)
    # get input and make sure it is split
    input_args = sys.argv[1:] if argv is None else shlex.split(argv)
    # All arguments are used by the test runner and should not interfere
    # with any other module that might also parse them, e.g. flake8.
    if sys.argv:
        sys.argv = sys.argv[:1]
    # configure parser and parse input
    parser = _configure_parser()
    args = parser.parse_args(input_args)
    # convert input to a pytest-able string
    pytest_input = _convert_to_pytest_input(args, input_args)
    # run pytest!
    return pytest.main(pytest_input)



    # return run_tests(verbosity, args.tests, report, args.log, args.server,
    #                  args.test_all_modules, args.timeit, interactive, args.n,
    #                  exclude=args.exclude, tutorial=args.tutorial,
    #                  hostname=args.hostname, ci_url=args.ci_url,
    #                  pr_url=args.pr_url)

