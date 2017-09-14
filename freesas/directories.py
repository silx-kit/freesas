#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: FreeSAS
#             https://github.com/kif/freesas
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#

"""

Contains the directory name where data are:
 * gui directory with graphical user interface files
 * openCL directory with OpenCL kernels
 * calibrants directory with d-spacing files describing calibrants
 * testdata: if does not exist: create it.

This file is very short and simple in such a way to be mangled by installers
It is used by pyFAI.utils._get_data_path

See bug #144 for discussion about implementation
https://github.com/kif/pyFAI/issues/144
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/12/2015"
__status__ = "development"

import os
import getpass
import tempfile
import logging
logger = logging.getLogger("pyFAI.directories")

FREESAS_DATA = "/usr/share/freesas"
FREESAS_TESTDATA = "/usr/share/freesas/testdata"

# testimage contains the directory name where
data_dir = None
if "FREESAS_DATA" in os.environ:
    data_dir = os.environ.get("FREESAS_DATA")
    if not os.path.exists(data_dir):
        logger.warning("data directory %s does not exist" % data_dir)
elif os.path.isdir(FREESAS_DATA):
    data_dir = FREESAS_DATA
else:
    data_dir = ""

# testdata contains the directory name where test images are located
testdata = None
if "FREESAS_TESTDATA" in os.environ:
    testdata = os.environ.get("FREESAS_TESTDATA")
    if not os.path.exists(testdata):
        logger.error("testimage directory %s does not exist" % testdata)
else:
    testdata = os.path.join(data_dir, "testdata")
    if not os.path.isdir(testdata):
        # create a temporary folder
        testdata = os.path.join(tempfile.gettempdir(), "freesas_testdata_%s" % (getpass.getuser()))
        if not os.path.exists(testdata):
            try:
                os.makedirs(testdata)
            except OSError as err:
                logger.warning("Creating test_directory %s ended in error %s, probably a race condition" % (testdata, err))


