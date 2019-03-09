# -*- coding: utf-8 -*-
"""
Tests for ObsPy's testrunner.

testrunner's functions are primarily a compatibility layer from ObsPy's old
test runner to use pytest under the hood.
"""
import shlex

from obspy.testing.testrunner import _convert_to_pytest_input, _configure_parser


class TestPytestStrConverter:
    """
    Tests for converting obspy's test runner output to pytest-able inputs.
    """
    parser = _configure_parser()  # configure an obspy parser for use in tests

    def convert_str(self, input_str):
        """
        Convert the obspy-runtest input to pytest input.
        """
        input_args = shlex.split(input_str)
        args = self.parser.parse_args(input_args)
        return _convert_to_pytest_input(args)

    def test_all(self):
        """
        Test that both --all and -a return --all
        """
        all1 = self.convert_str('--all')
        all2 = self.convert_str('-a')
        assert all1 == all2 == ['--all']

    def test_exclude(self):
        """
        Test for excluding modules.
        """
        # exclude specifies a single directory
        exclude = self.convert_str('-x obspy.core')
        assert exclude == ['--exclude obspy/core']
        # exclude specifies a single python file
        arg = '-x obspy.core.tests.test_resource_identifier'
        expected = 'obspy/core/tests/test_resource_identifier.py'
        exclude = self.convert_str(arg)[0]
        assert exclude.replace('--exclude ', '') == expected
        # exclude specifies a single test case
        arg = '-x obspy.signal.tests.test_filter.FilterTestCase'
        expected = 'obspy/signal/tests/test_filter.py::FilterTestCase'
        exclude = self.convert_str(arg)[0].replace('--exclude ', '')
        assert expected == exclude
        # test multiple excludes
        arg1 = 'obspy.core.tests.test_event'
        arg2 = ('obspy.signal.tests.test_filter.FilterTestCase.'
                'test_lowpass_cheby_2')
        arg = '-x %s --exclude %s' % (arg1, arg2)
        expected1 = 'obspy/core/tests/test_event.py'
        expected2 = ('obspy/signal/tests/test_filter.py::FilterTestCase::'
                     'test_lowpass_cheby_2')
        exclude = self.convert_str(arg)
        assert expected1 in exclude[0]
        assert expected2 in exclude[1]
