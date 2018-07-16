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

import pipegraph.base
import pipegraph.adapters
import pipegraph.demo_blocks

from .base import (PipeGraphRegressor,
                   PipeGraphClassifier,
                   PipeGraph,
                   add_mixins_to_step,
                   )

from .adapters import (FitPredictMixin,
                       FitTransformMixin,
                       AtomicFitPredictMixin,
                       CustomFitPredictWithDictionaryOutputMixin,
                      )


__all__ = ['PipeGraphClassifier',
           'PipeGraphRegressor',
           'PipeGraph',
           'add_mixin_to_step',
           ]
