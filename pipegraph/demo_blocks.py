# -*- coding: utf-8 -*-
# The MIT License (MIT)
#
# Copyright (c) 2018 Laura Fernandez Robles,
#                    Hector Alaiz Moreton,
#                    Jaime Cifuentes-Rodriguez,
#                    Javier Alfonso-Cendón,
#                    Camino Fernández-Llamas,
#                    Manuel Castejón-Limas
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from sklearn.base import BaseEstimator


class CustomCombination(BaseEstimator):
    """
    Only For PAELLA purposes
    """

    def fit(self, dominant, other):
        """

        Args:
            dominant:
            other:

        Returns:

        """
        return self

    def predict(self, dominant, other):
        """

        Args:
            dominant:
            other:

        Returns:

        """
        return np.where(dominant < 0, dominant, other)



class CustomPower(BaseEstimator):
    """ Raises X data to power defined such as range as parameter
    Parameters
    ----------
    power : range of integers for the powering operation
    """

    def __init__(self, power=1):
        self.power = power

    def fit(self):
        """"
        Returns
        -------
            self : returns an instance of _CustomPower.
        """
        return self

    def predict(self, X):
        """"
        Parameters
        ----------
        X: iterable
            Data to power.
        Returns
        -------
        result of raising power operation
        """
        return X.values.reshape(-1, ) ** self.power


from pipegraph.adapters import (FitPredictMixin,
                                CustomFitPredictWithDictionaryOutputMixin,
                                )


strategies_for_demo_blocks_adaptees = {
    CustomCombination: FitPredictMixin,
    CustomPower: FitPredictMixin,
}


