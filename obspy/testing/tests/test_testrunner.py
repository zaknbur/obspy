# -*- coding: utf-8 -*-
"""
Tests for ObsPy's testrunner.

testrunner's functions are primarily a compatibility layer from ObsPy's old
test runner to use pytest under the hood.
"""

import pytest


from obspy.testing.testrunner import _convert_to_pytest_str, _configure_parser



class TestPytestStrConverter:
    """
    Tests for converting obspy's test runner output to pytest-able inputs.
    """
    parser = _configure_parser()  # configure an obspy parser for use in tests

    def convert_str(self, input_str):
        """
        Convert the obspy-runtest input to pytest input.
        """
        args = self.parser.parse_args(input_str)
        return _convert_to_pytest_str(args)

    def test_all(self):
        """
        Test that both --all and -a return --all
        """
        all1 = self.convert_str('--all')
        all2 = self.convert_str('-a')
        assert all1 == all2 == '--all'

    def test_exclude(self):
        """
        Test for excluding modules.
        """
        exclude = self.convert_str('-x obspy.core')
        assert exclude == '--exclude obspy/core'
        exclude = self.convert_str('-x obspy.core.tests.test_resource_id')
        assert exclude == '--exclude obspy/core/tests/test_resource_id.py'











