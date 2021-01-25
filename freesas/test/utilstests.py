#!usr/bin/env python
# coding: utf-8

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "25/01/2021"
__copyright__ = "2015-2021, ESRF"

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
