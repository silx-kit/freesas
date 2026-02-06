# -*- coding: utf-8 -*-
#
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2024-2024  European Synchrotron Radiation Facility, Grenoble, France
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "06/02/2026"

import unittest
import logging
import numpy as np
from .utilstest import get_datafile
from ..sasio import load_scattering_data
from ..dnn import DNN, DenseLayer, forward_propagation, preprocess, tanh, relu, sigmoid, linear

logger = logging.getLogger(__name__)

class TestDNN(unittest.TestCase):

    def test_activation_functions(self):
        """
        Test for the activation functions
        """
        x = np.array([-1, 0, 1])

        # Test tanh
        expected_tanh = np.tanh(x)
        self.assertTrue(np.allclose(tanh(x), expected_tanh), msg="tanh function failed")

        # Test relu
        expected_relu = np.maximum(0, x)
        self.assertTrue(np.allclose(relu(x), expected_relu), msg="relu function failed")

        # Test sigmoid
        expected_sigmoid = 1 / (1 + np.exp(-x))
        self.assertTrue(np.allclose(sigmoid(x), expected_sigmoid), msg="sigmoid function failed")

        # Test linear
        expected_linear = x
        self.assertTrue(np.allclose(linear(x), expected_linear), msg="linear function failed")

        logger.info("test_activation_functions ran successfully")


    def test_preprocess(self):
        """
        Test for the preprocessing function
        """
        datfile = get_datafile("bsa_005_sub.dat")
        data = load_scattering_data(datfile)
        q, I, sigma = data.T
        Iprep = preprocess(q, I)
        self.assertEqual(Iprep.max(), 1, msg="range 0-1")
        self.assertEqual(Iprep.shape, (1024,), msg="size 1024")



    def test_forward_propagation(self):
        """
        Test for the forward_propagation function
        """
        try:
            X = np.random.rand(1, 10)
            params = [np.random.rand(10, 20), np.random.rand(20), np.random.rand(20, 10), np.random.rand(10)]
            activations = [np.tanh, np.tanh]
            output = forward_propagation(X, params, activations)
            self.assertEqual(output.shape, (1, 10))
            logger.info("test_forward_propogation ran successfully")
        except Exception as e:
            logger.error(f"test_forward_propagation failed: {e}")
            raise


    def test_DenseLayer(self):
        """
        Test for the DenseLayer class
        """
        try :
            weights = np.random.rand(10, 20)
            bias = np.random.rand(20)
            layer = DenseLayer(weights, bias, 'tanh')
            self.assertEqual(layer.input_size, 10)
            self.assertEqual(layer.output_size, 20)
            output = layer.forward(np.random.rand(1, 10))
            self.assertEqual(output.shape, (1, 20))
            logger.info("test_DenseLayer ran successfully")
        except Exception as e:
            logger.error(f"test_DenseLayer failed: {e}")
            raise



    def test_DNN(self):
        """
        Test for the DNN class
        """
        try:
            layers = [DenseLayer(np.random.rand(10, 20), np.random.rand(20), 'tanh'),
                      DenseLayer(np.random.rand(20, 10), np.random.rand(10), 'tanh')]
            dnn = DNN(*layers)
            output = dnn.infer(np.random.rand(1, 10))
            self.assertEqual(output.shape, (1, 10))
            logger.info("test_DNN ran successfully")
        except Exception as e:
            logger.error(f"test_DNN failed: {e}")
            raise


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loader(TestDNN))
    # test_suite.addTest(TestDNN("test_preprocess"))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
