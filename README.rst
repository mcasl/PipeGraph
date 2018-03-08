Welcome to PipeGraph!
============================================

Introduction
------------

Scikit-Learn's Pipeline is a very useful tool to bind a sequence of transformers and a final estimator
in a single unit capable of working itself as an estimator. This eases managing the fit and predict
of the complete sequence of steps through a simple call to their fit and predict methods,
and in combination with the GridSearchCV function allows to tune the hyperparameters of these steps
while in the search for the best model.

PipeGraph extends the concept of Pipeline by using a graph structure that can handle Scikit-Learn's
objects in imaginative layouts. This allows the user to:

- Express simple and complex sequences of steps, from linear workflows to elaborated graphs
- To have access to all the data generated along the intermediate steps
- To use GridSearchCV on those graphs

Let's see a few examples to understand how we can express these situations through PipeGraph objects.

LICENSE
------------
The MIT License (MIT)

Copyright (c) 2018 Laura Fernandez Robles,
                   Hector Alaiz Moreton,
                   Jaime Cifuentes-Rodriguez,
                   Javier Alfonso-Cendón,
                   Camino Fernández-Llamas,
                   Manuel Castejón-Limas


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.