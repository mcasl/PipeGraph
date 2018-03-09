###############################################################################
# Then, we need to provide the connections amongst steps as a dictionary. Please, refer to :ref:`Figure 2 <fig2>` for a faster understanding.
#
# - The keys of the top level entries of the dictionary must be the same as those of the previously defined steps.
# - The values assocciated to these keys define the variables from other steps that are going to be considered as inputs for the current step. They are dictionaries themselves, where:
#
#   - The keys of the nested dictionary represent the input variables as named at the current step.
#   - The values assocciated to these keys define the steps that hold the desired information and the variables as named at that step. This information can be written as:
#
#     - A tuple with the label of the step in position 0 followed by the name of the output variable in position 1.
#     - A string representing a variable from an external source to the :class:`PipeGraphRegressor` object, such as those provided by the user while invoking the ``fit``, ``predict`` or ``fit_predict`` methods.
#
# For instance, the linear model accepts as input ``X`` the output named ``predict`` at the ``scaler`` step, and as input ``y`` the value of ``y`` passed to ``fit``, ``predict`` or ``fit_predict`` methods.

"""
connections = { 'selector':     {'X':'X'},
                'custom_power': {'X': ('selector', 'sample_weight')},
                'scaler':       {'X': ('selector', 'X')},
                'polynomial_features': {'X': ('scaler', 'predict')},
                'linear_model': {'X': ('polynomial_features', 'predict'),
                                 'y': 'y',
                                 'sample_weight': ('custom_power', 'predict')}  }
"""