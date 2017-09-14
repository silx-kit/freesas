#!usr/bin/env python
# coding: utf-8
from __future__ import print_function

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "16/12/2015"
__copyright__ = "2015, ESRF"

import os
import logging
logger = logging.getLogger("utilstest")
from ..directories import testdata


def get_datafile(name):
    """Provides the full path of a test file,
    downloading it from the internet if needed

    :param name: name of the file to get
    :return: full path of the datafile
    """
    fullpath = os.path.join(testdata, name)
    if not os.path.exists(fullpath):
        logger.error("No such file: %s. Please implement the automatic distribution of the testdata")
        # TODO: automatic download of test-data
    return fullpath
