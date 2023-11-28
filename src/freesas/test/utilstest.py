#!usr/bin/env python
# coding: utf-8

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "28/11/2023"
__copyright__ = "2015-2021, ESRF"

import os
import logging
logger = logging.getLogger("utilstest")
from silx.resources import ExternalResources
downloader = ExternalResources("freesas", "http://www.silx.org/pub/freesas/testdata", "FREESAS_TESTDATA")


def get_datafile(name):
    """Provides the full path of a test file,
    downloading it from the internet if needed

    :param name: name of the file to get
    :return: full path of the datafile
    """
    logger.info(f"Download file {name}")
    fullpath = downloader.getfile(name)
    return fullpath

def clean():
    pass

class TestOptions(object):
    """
    Class providing useful stuff for preparing tests.
    """
    def __init__(self):
        self.TEST_LOW_MEM = False
        """Skip tests using too much memory"""

        self.TEST_RANDOM = False
        """Use a random seed to generate random values"""

        self.options = None

    def __repr__(self):
        return f"TEST_LOW_MEM={self.TEST_LOW_MEM} "\
               f"TEST_RANDOM={self.TEST_RANDOM} "
    @property
    def low_mem(self):
        """For compatibility"""
        return self.TEST_LOW_MEM

    def configure(self, parsed_options=None):
        """Configure the TestOptions class from the command line arguments and the
        environment variables
        """

        if parsed_options is not None and parsed_options.low_mem:
            self.TEST_LOW_MEM = True
        elif os.environ.get('FREESAS_LOW_MEM', 'True') == 'False':
            self.TEST_LOW_MEM = True

        if parsed_options is not None and parsed_options.random:
            self.TEST_RANDOM = True
        if os.environ.get('FREESAS_RANDOM', 'False').lower() in ("1", "true", "on"):
            self.TEST_RANDOM = True


    def add_parser_argument(self, parser):
        """Add extrat arguments to the test argument parser

        :param ArgumentParser parser: An argument parser
        """
        parser.add_argument("-l", "--low-mem", dest="low_mem", default=False,
                            action="store_true",
                            help="Disable test with large memory consumption (>100Mbyte")
        parser.add_argument("-r", "--random", dest="random", default=False,
                            action="store_true",
                            help="Enable actual random number to be generated. By default, stable seed ensures reproducibility of tests")


test_options = TestOptions() #singleton
