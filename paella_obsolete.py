# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 20:00:18 2017

@author: Manolo
"""
#%%
"""
\section{The {\em PAELLA}\/ algorithm}
\label{sec:outlier}
It must be noted that the metric of the {\em PAELLA}\/ algorithm is derived from a previous partition $C_k$ $(k=0,\dots,g)$ of the data set, where samples are allocated to $g$ different groups on the basis of the empirical clusters they rest on. The special $k=0$ case gathers the samples which cannot be reliably allocated to any other cluster if the user decides to apply a cluster strategy allowing the presence of noise samples. The reader may find useful \inlinecite{hardy:96:a}, \inlinecite{cuevas+febrero+fraiman:96:a} and \inlinecite{fraley+raftery:98:a} for a description of the methods to determine the number of clusters. The {\em PAELLA}\/ algorithm can be understood as a multi-step procedure structured in three phases:
\subsection{Phase I:} Phase I is aimed at fitting with a local approach the $x_i$ samples from the initial data set $\mathbf{X}$ of interest into different linear models to obtain a collection of hypersurfaces fitting and coating the data set. Phase I also tries to spawn a series of hypersurfaces so that each sample can be subsequently defined as suitable -- or unsuitable -- for the models according to its ``goodness-of-fit''. Of course, in the different trials, the number of  $M_i$ models proposed varies depending on the seed samples chosen at each iteration. For each iteration, Phase I analyzes independently the different clusters one at a time revealing potential outliers and distinguishing between samples in the core of the cloud and those that do not follow the general trend.
\subsection{Phase II:}In Phase II, the outlierness of every sample is assessed against the corresponding collection of models of each trial. As different $G_i$ subsets  and $M_i$ models are available, every sample in the data set can be evaluated for each model $M_i$. Thus, a list of values of the residuals for every $x_i$ sample is obtained for the current trial. Only the smallest residual of $x_i$ is considered, and this residual decides the model that particular sample is associated to. Once the minimum residual for every vector has been obtained, the samples with the biggest residuals are considered as possible outliers in the context of the current trial. This definition of ``prone to outlierness'' is collected in a vector of outlierness identification that will be used as input in Phase III.
\subsection{Phase III:}
The results provided by Phase III allow to draw some conclusions on the pattern of the obtained outliers in a further analysis. This analysis is a key factor to understand the behavior of the system originating the data, since strange behaviors in the actual components of the system might be discovered, and correcting measures to avoid the degeneration of the process may be implemented.
\begin{algorithm}
\caption{Fitting of the hypersurfaces series}
\label{ph1}
\begin{algorithmic}[1]
\STATE One random $x_{k\,i} \in Z_{k\,i}$; $k=1 \ldots g$, $Z_{k\,1}=C_k$ sample is considered as a seed point of the supporting $G_i$ subset.
\STATE The remaining $x_j \in Z_{k\,i}$ points are classified according to their Mahalanobis distance to the seed point $D(x_j,x_{k\,i})$.
\STATE $\beta$ $x_j$ points, those with the smallest $D(x_j,x_{k\,i})$, are added to the $G_i$ subset. If there are less than $\beta$ points, $\beta_{min}$ is used instead.
\STATE A $M_i$ model is inferred from $G_i$ (ideally using a robust and affine equivariant fitting ).
\STATE For all $x_j \in C_k, x_j \notin G_i$ a residual $r_{x_j}$ is evaluated against the $M_i$ model.
\STATE The $x_j$ samples whose residual $r_{x_j}$ reports to have a quantile function value lower than $r$ may be considered as compliant with $M_i$, and be added to $G_i$.
\STATE Steps {\small 1} to {\small 6} are iterated considering $Z_{k\,i+1}=Z_{k\,i}-G_i$, as far as reasonable: the points become exhausted at $Z_{k\,i+1}$ or the density in the subset falls under the r_threshold $q$.
\end{algorithmic}
\end{algorithm}
\begin{algorithm}
\caption{Assessment of outlierness in each trial}
\label{ph2}
\begin{algorithmic}[1]
\STATE The vector $r_{x_j}=\min\{r_{x_j,M_i}=y_j-M_i(x_j),\displaystyle {\forall x_j \in \bigcup_{k=0}^g C_k}, \forall M_i \}$ binds the smallest residuals for each sample to their corresponding $M_i$ ``best" fit.
\STATE $r_{x_j}>\alpha$ identifies the samples prone to outlierness, and thus, a list of outliers in the context of a particular trial can be written to reflect the current results.
\end{algorithmic}
\end{algorithm}
\begin{algorithm}
\caption{Assessment of outlierness in iterated trials and search for the origin of the perturbations.}
\label{ph3}
\begin{algorithmic}[1]
\STATE Phase I and Phase II are iterated according to the time available while a vector containing the frequency of "outlierness" for every sample combines the particular results of each iteration.
\STATE The samples with the biggest outlierness frequency, those above the $\gamma$ quantile, are defined as outliers and separated for a subsequent analysis.
\STATE The process can be repeated with the clean resulting subset for a further detection.
\end{algorithmic}
\end{algorithm}
"""
#%%
import pandas as pd
import statsmodels.formula.api as sm
import numpy as np
import patsy
import scipy 


class Paella():
    def __init__(self,
                 formula,
                 data, 
                 classification,
                 noise_label=None,
                 max_it = 1,
                 beta_rate = 0.3, 
                 betaMin_rate = 0.1, 
                 width_r = 0.95,
                 width_alpha = 0.95,):
        
        self.formula = formula
        self.data = data
        self.classification = classification
        self.noise_label = noise_label
        self.max_it = max_it
        self.beta_rate = beta_rate 
        self.betaMin_rate = betaMin_rate 
        self.width_r = width_r
        self.width_alpha = width_alpha
        
        self.occurrences = None
        self.iterations = 0        
        
    def fit(self):
        def getResiduals(model, X, Y):
            return pd.Series(model.predict(X) - np.asarray(Y).T[0])
        
        if self.occurrences is None :
            self.occurrences =  pd.DataFrame(0, index=self.data.index, columns=["Occurrences"])
            
        rowsDict = self.classification.groupby("Classification").groups
  
        if self.noise_label is not None :
            del rowsDict[self.noise_label]
            
        invertedSigmas = {clusterLabel: np.linalg.inv(np.cov(self.data.loc[rows,:], rowvar=False)) for clusterLabel, rows in rowsDict.items() }

        for dummy in range(self.max_it):
            print(dummy, " ,")
            M_i_model_list = list()
            ## PHASE 1 
            for clusterLabel, rowIndexes in rowsDict.items() :
                rows = rowIndexes.copy()
                number_of_observations = len(rows)                
                beta = np.floor(self.beta_rate * number_of_observations).astype(int)
                betaMin   = np.floor(self.betaMin_rate * number_of_observations).astype(int)
                while number_of_observations > betaMin :
                    beta_within = beta if number_of_observations > beta else betaMin #¿Cambiar betaMin por number_of_observations?
                    # 1.1 One random $x_{k\,i} \in Z_{k\,i}$; $k=1 \ldots g$, $Z_{k\,1}=C_k$ sample is considered as a seed point of the supporting $G_i$ subset.
                    seed_point = np.random.choice(rows, 1, replace=False)
                    # 1.2 The remaining $x_j \in Z_{k\,i}$ points are classified according to their Mahalanobis distance to the seed point $D(x_j,x_{k\,i})$.
                    D = scipy.spatial.distance.cdist(XA=self.data.loc[rows,:],
                                                     XB=self.data.loc[seed_point,:],
                                                     metric="mahalanobis",
                                                     VI=invertedSigmas[clusterLabel])
                    D = pd.DataFrame(D, index=rows, columns=["Distance"]).sort_values(by="Distance", ascending=True)
                    # 1.3 $\beta$ $x_j$ points, those with the smallest $D(x_j,x_{k\,i})$, are added to the $G_i$ subset. If there are less than $\beta$ points, $\beta_{min}$ is used instead.
                    G_i_rows      = D.head(beta_within).index
                    # 1.4 A $M_i$ model is inferred from $G_i$ (ideally using a robust and affine equivariant fitting ).
                    M_i_model       = sm.ols(formula=self.formula, data=self.data.loc[G_i_rows, :]).fit()
                    M_i_model_list.append(M_i_model) # We preserve M_i_model for Phase 2
                    # 1.5 For all $x_j \in C_k, x_j \notin G_i$ a residual $r_{x_j}$ is evaluated against the $M_i$ model.
                    # 1.6 The $x_j$ samples whose residual $r_{x_j}$ reports to have a quantile function value lower than $r$ may be considered as compliant with $M_i$, and be added to $G_i$.
                    rows = rows.drop(G_i_rows) # We discard those samples already in G_i
                    if len(rows) > 1 :
                        ypred, xpred  = patsy.dmatrices(self.formula, data=self.data.loc[rows,:], return_type="dataframe")
                        r_x_j         = getResiduals(model=M_i_model, X=xpred, Y=ypred) # residuals out of model
                        r_threshold   = scipy.stats.norm.ppf(self.width_r, loc = r_x_j.mean(), scale = r_x_j.std())
                        rows = rows[np.abs(r_x_j) > r_threshold] # We discard those samples out of G_i agreeing with M_i_model (discarding is conceptually equivalent to move those samples to the G_i subset)
                        
                    number_of_observations = len(rows)
                    # 1.7 Steps {\small 1} to {\small 6} are iterated considering $Z_{k\,i+1}=Z_{k\,i}-G_i$, as far as reasonable: the points become exhausted at $Z_{k\,i+1}$ or the density in the subset falls under the r_threshold $q$.
            # End of rowsDict.items() loop                 
            
            ## PHASE 2  
            # 2.1 The vector $r_{x_j}=\min\{r_{x_j,M_i}=y_j-M_i(x_j),\displaystyle {\forall x_j \in \bigcup_{k=0}^g C_k}, \forall M_i \}$ binds the smallest residuals for each sample to their corresponding $M_i$ ``best" fit.
            dataYpred, dataXpred = patsy.dmatrices(self.formula, data=self.data, return_type="dataframe")
            r_x_j  = pd.concat( [getResiduals(model=model, X=dataXpred, Y=dataYpred) for model in  M_i_model_list]  , axis=1).abs().min(axis=1)
            # 2.2 $r_{x_j}>\alpha$ identifies the samples prone to outlierness, and thus, a list of outliers in the context of a particular trial can be written to reflect the current results.      
            alpha_threshold   = scipy.stats.norm.ppf(self.width_alpha, loc = r_x_j.mean(), scale = r_x_j.std())
            self.occurrences.loc[ np.abs(r_x_j) > alpha_threshold , "Occurrences"] +=  1 
            
            
            
        #Next line goes after End of for loop; beware not to mess the identation
        self.iterations     += self.max_it 

