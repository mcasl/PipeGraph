# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 20:00:18 2017

@author: Manolo
"""

class Paella():
    def __init__(self,
                 formula,
                 data, 
                 classification,
                 max_it = 1,
                 bunch_size = 0.3, 
                 min_bunch_size = 0.1, 
                 width = 0.95):
        
        self.formula = formula
        self.data = data
        self.classification = classification
        self.max_it = max_it
        self.bunch_size = bunch_size 
        self.min_bunch_size = min_bunch_size 
        self.width = width
        
        self.occurrences = None
        self.iterations = 0        
        
    def fit(self):
        if self.occurrences is None :
            self.occurrences =  pd.DataFrame(0, index=self.data.index, columns=["Occurrences"])

        rowsDict = {key: rowIndexes for key, rowIndexes in classification.groupby("Classification").groups.items()}
        invertedSigmas = {clusterLabel: np.linalg.inv(np.cov(X, rowvar=False)) for clusterLabel, X in Xdict.items() }

        for dummy in range(self.max_it):
            print(dummy, " ,")
            for clusterLabel, rowIndexes in rowsDict.items() :
                rows = rowIndexes.copy()
                number_of_observations = len(rows)                
                usual_number_within = np.floor(self.bunch_size * number_of_observations).astype(int)
                min_number_within   = np.floor(self.min_bunch_size * number_of_observations).astype(int)
                while number_of_observations > min_number_within :
                    X = self.data.loc[rows,:]
                    number_within = usual_number_within if number_of_observations > usual_number_within else min_number_within
                    seed_point = X.ix[np.random.choice(X.index, 1, replace=False)]
                    D = cdist(XA=X, XB=seed_point, metric="mahalanobis", VI=invertedSigmas[clusterLabel])
                    D = pd.DataFrame(D, index=X.index, columns=["Distance"]).sort_values(by="Distance", ascending=True)
                    within      = D.head(number_within).index
                    model       = sm.ols(formula=formula, data=X.loc[within, :]).fit()
                    residuals   = model.resid
                    rows = rows.drop(within)
                    if len(rows) > 1 :
                        ypred, xpred       = patsy.dmatrices(formula, data=X.loc[rows,:], return_type="dataframe")
                        residuals_out_of_model = model.predict(xpred) - np.asarray(ypred).T[0]
                        threshold          = scipy.stats.norm.ppf(width, loc = residuals_out_of_model.mean(), scale = residuals_out_of_model.std())
                        rows = rows[np.abs(residuals_out_of_model) > threshold]
                    number_of_observations = len(rows)
                
                #Next line goes after End of while; beware not to mess the identation
                self.occurrences.loc[X.index, "Occurrences"] +=  1  
        #Next line goes after End of for loop; beware not to mess the identation
        self.iterations     += self.max_it 

