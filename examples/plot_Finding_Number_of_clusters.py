"""
.. _example_finding_the_number_of_clusters:

Finding number of clusters
-----------------------------------------

A two step PipeGraph is built by embedding a clustering algorithm such as Kmeans followed by
 a supervised classification algorithm such as Linear Discriminant Analysis. This kind of structure
 is useful in those scenarios in which the dataset comprises different subpopulations
 but the class labels are unknown. A common question that arises in such scenario is related to
 the best number of clusters. For answering that question, we might want to measure the quality
 of the clustering in accordance to the later capability of reproducing the results obtained by
 the clustering algorithm on a model based classification technique such as linear discriminant analysis.
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
from pipegraph import PipeGraph
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns; sns.set()
#%%
###############################################################################
# Generate an artificial dataset
#

X, y = datasets.make_blobs(n_samples=10000, n_features=5, centers=10)


clustering = KMeans(n_clusters=10)
classification = LinearDiscriminantAnalysis()

steps = [('clustering', clustering),
         ('classification', classification)
         ]

pgraph = PipeGraph(steps=steps)
pgraph.inject(sink='clustering',     sink_var='X', source='_External',  source_var='X')
pgraph.inject(sink='classification', sink_var='X', source='_External',  source_var='X')
pgraph.inject(sink='classification', sink_var='y', source='clustering', source_var='predict')

n_folds = 5

number_of_clusters_to_explore = np.arange(2, 50)
gs = GridSearchCV(pgraph, param_grid=dict(clustering__n_clusters=number_of_clusters_to_explore), cv=n_folds)
gs.fit(X)
#%%
###############################################################################
# Plot showing CV scores with standard deviation bands
# Code inspired by http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#sphx-glr-download-auto-examples-exercises-plot-cv-diabetes-py

number_of_clusters = gs.cv_results_['param_clustering__n_clusters'].data.astype(int)
errors = gs.cv_results_['mean_test_score']
deviations = gs.cv_results_['std_test_score']/np.sqrt(n_folds)

plt.figure().set_size_inches(8, 6)
plt.plot(number_of_clusters, errors)
plt.plot(number_of_clusters, errors + deviations, 'b--')
plt.plot(number_of_clusters, errors - deviations, 'b--')
plt.fill_between(number_of_clusters, errors + deviations, errors - deviations, alpha=0.2)
plt.ylabel('CV score +/- std error')
plt.xlabel('Number of clusters')
plt.axhline(np.max(errors), linestyle='--', color='.5')
plt.xlim([number_of_clusters_to_explore[0], number_of_clusters_to_explore[-1]])
plt.show()
#%%
###############################################################################
# The plot shows best results for lower values of the number of cluster parameter.
# It also points out a decline for values bigger than 10.
# Let's see the results for the best estimator.
# Code for plotting the confusion matrix taken from 'Python Data Science Handbook' by Jake VanderPlas

model = gs.best_estimator_
print("Best parameters:", gs.best_params_)

score = model.score(X)
print('Accuracy:',score)

y_based_on_kmeans = model._predict_data[('clustering', 'predict')]
gs_pred = model.predict(X)

mat = confusion_matrix(y_true=y_based_on_kmeans, y_pred=gs_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('kmeans label')
plt.ylabel('predicted label');
plt.show()
#%%



