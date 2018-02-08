# -*- coding: utf-8 -*-
"""
Created on 2017-11-28

  1.-  Usando Knn como predictor de outliers. Aquí nos basamos en los resultados obtenidos en el paella para delimitar las zonas con outliers y usamos un clasificador para entrenarlo y que sea el responsable de las imputaciones para los datos nuevos.
         TODO: Poder variar el clasificador: Kmeans, Mclust, ...

       2.- Otra opción:
         Para cada iteración hecha en el fit miramos a que modelo pertenece el nuevo dato y ahí calculamos el residuo y vemos si es compliant o no.
         2.1 : ¿a que modelo pertenece? Significando esto el modelo del punto más cercano al nuevo dato
         2.2 : ¿a que modelo pertenece? Significando esto el modelo con mínimo residuo de todos los de esa iteración
         2.3 : ¿a que modelo pertenece? La favorita de Laura: Significando esto el modelo con mínimo residuo relativo al threshold local.
         Para eso en vez de comparar con el threshold lo que hacemos es:
             - para cada dato, en cada modelo, calculamos la probabilidad del residuo dados r_x_j.mean() y r_x_j.std() calculados en el fit. Cogemos como modelo más cercano aquel más ventajoso desde el punto de vista de la probabilidad
         2.4 :  Limitar todo esto a los modelos del cluster de turno




"""
#%%
import pandas as pd

import numpy as np
import scipy

from sklearn.base             import BaseEstimator, TransformerMixin
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.linear_model     import LinearRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.utils            import check_random_state, check_X_y


class Paella(BaseEstimator, TransformerMixin):
    def __init__(self,
                 regressor    = LinearRegression,
                 noise_label  = None,
                 max_it       = 1,
                 regular_size = 100,
                 minimum_size = 30,
                 width_r      = 0.95,
                 n_neighbors  = 1,
                 power        = 10,
                 random_state = 42):

        self.regressor    = regressor
        self.noise_label  = noise_label
        self.max_it       = max_it
        self.regular_size = regular_size
        self.minimum_size = minimum_size
        self.width_r      = width_r
        self.n_neighbors  = n_neighbors
        self.power        = power
        self.random_state = check_random_state(random_state)
        self._location_estimator = np.median
        self._scale_estimator    = np.std

    def fit(self, X, y, classification):
      #  X, y  = check_X_y(X, y) Me da problemas con el nombre de la columna Classification que se pierde
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        data = pd.concat([X, y], axis=1)
        classification = pd.DataFrame({'Classification': classification},  index=X.index, dtype=np.int16)
        self.appointment_    = pd.DataFrame(0, columns=range(self.max_it), index=X.index, dtype=np.int16) # 0 label appointed to outliers
        self.models_         = { iteration_number: dict() for iteration_number in range(self.max_it) }
        self.sample_weight_  = pd.Series(1, index=X.index)
        if self.noise_label is None:
            self.noise_label = classification['Classification'].min()-1

        for iteration_number in range(self.max_it):
            print(iteration_number, " ,")

            indexes_dictionary   = classification.groupby('Classification').groups
            noise_indexes        = indexes_dictionary.pop(self.noise_label, [])
            self.invertedSigmas_ = {cluster: np.linalg.inv(np.cov(data.loc[indexes,:], rowvar=False))
                                   for cluster, indexes in indexes_dictionary.items() }

            model_number = 1  # 0 label assigned to outliers
            for cluster, indexes in indexes_dictionary.items():
                number_of_observations = len(indexes)
                while number_of_observations >= self.minimum_size :
                    size = min(self.regular_size, number_of_observations)
                    seed_point = self.random_state.choice(a=indexes, size=1, replace=False, p=None)
                    D = pd.DataFrame(scipy.spatial.distance.cdist(XA=data.loc[indexes,:],
                                                                  XB=data.loc[seed_point,:],
                                                                  metric="mahalanobis",
                                                                  VI=self.invertedSigmas_[cluster]),
                                     index=indexes,
                                     columns=["Distance"])

                    G_i_indexes = D.sort_values(by="Distance", ascending=True).head(size).index
                    M_i_model   =  self.regressor()
                    M_i_model.fit(X = X.loc[G_i_indexes, :],
                                  y = y.loc[G_i_indexes, :],
                                  sample_weight = self.sample_weight_[G_i_indexes]**self.power)
                    
                    self.models_[iteration_number][model_number] = M_i_model
                    
                    compliant = self._check_adhesion_to_model(iteration_number = iteration_number,
                                                              model_number = model_number,
                                                              X = X,
                                                              y = y,
                                                              indexes = indexes,
                                                              noise_indexes = noise_indexes)
                    if len(compliant) == 0:
                        break
                    
                    self.appointment_.loc[compliant, iteration_number] = model_number
                    indexes = indexes.drop(compliant)
                    number_of_observations = len(indexes)
                    model_number += 1
                    # End of while
            # End of for cluster

# self._penalization['zero_weighted'] ()
            indexes_appointed_zero = X.index[ self.appointment_[iteration_number] == 0 ]
            self.sample_weight_[indexes_appointed_zero] -= 1/self.max_it
            #self.sample_weight_[indexes_appointed_zero] = 0
            #X.loc[indexes_appointed_zero, 'Classification'] = self.noise_label

        # End of for iteration_number
        self.knn = KNeighborsClassifier(self.n_neighbors)
        self.knn.fit(data, self.appointment_)
        return self

    def transform(self, X, y):
        check_is_fitted(self, 'appointment_')
        data = pd.concat([X,y], axis=1)
        appointment =  self.knn.predict(data)
        #TODO: correct the following line to support max_it = 1 (error in dimensions)
        occurrences = (appointment == 0).sum(axis=1)
        sample_weight = 1 - occurrences/self.max_it
        return sample_weight

    def _getResiduals(self, model, X, y, indexes):
        if len(indexes) > 0:
            result = (model.predict(X.loc[indexes, :]) - y.loc[indexes, :]).stack()
        else:
            result = np.array([])
        return result


    def _check_adhesion_to_model(self, iteration_number, model_number, X, y, indexes, noise_indexes):
        model = self.models_[iteration_number][model_number]
        indexes_residuals = self._getResiduals(model = model, X = X, y = y, indexes = indexes)
        core = X.index[ self.appointment_[iteration_number] == model_number ]
        if len(core) > 0:
            reference_residuals = self._getResiduals(model = model, X = X, y = y, indexes = core) 
        else:
            reference_residuals = indexes_residuals
            
        reference_mu    = self._location_estimator(reference_residuals)
        reference_scale = self._scale_estimator(reference_residuals)
        reference_threshold = scipy.stats.norm.ppf(self.width_r, loc = reference_mu, scale = reference_scale)
        compliant = indexes[ (2*reference_mu-reference_threshold <= indexes_residuals) &  (indexes_residuals <= reference_threshold) ]
        if len(noise_indexes) > 0 :
            noise_residuals = self._getResiduals(model = model, X = X, y = y, indexes = noise_indexes)
            compliant.append( noise_indexes[ (2*reference_mu-reference_threshold <= noise_residuals) &
                                                                                   (noise_residuals <= reference_threshold) ] )
        return compliant



    def get_params(self, deep=True):
        return super().get_params(deep=False)
"""

                    self.appointment_.loc[G_i_indexes, iteration_number] = model_number
                    indexes = indexes.drop(G_i_indexes)

"""



