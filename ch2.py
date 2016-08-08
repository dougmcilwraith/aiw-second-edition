#2.1
import numpy as np
from sklearn import datasets
					
iris = datasets.load_iris() 
np.array(zip(iris.data,iris.target))[0:10]	

#2.2
print(iris.DESCR)
iris.target_names

#2.3
#Psuedocode

#2.4
from sklearn.cluster import KMeans
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
km = KMeans(n_clusters=3)
km.fit(X)

print(km.labels_)

#2.5
from sklearn.cluster import KMeans
from sklearn import datasets
from itertools import cycle, combinations
import matplotlib.pyplot as pl

iris = datasets.load_iris()
km = KMeans(n_clusters=3)
km.fit(iris.data)

predictions = km.predict(iris.data)

colors = cycle('rgb')
markers = cycle('^+o')
labels = ["Cluster 1","Cluster 2","Cluster 3"]
targets = range(len(labels))

feature_index=range(len(iris.feature_names))
feature_names=iris.feature_names
combs=combinations(feature_index,2)

f,axarr=pl.subplots(3,2)
axarr_flat=axarr.flat

for comb, axflat in zip(combs,axarr_flat):
        for target, color, label, marker in zip(targets,colors,labels,markers):
                feature_index_x=comb[0]
                feature_index_y=comb[1]
                axflat.scatter(iris.data[predictions==target,feature_index_x],
                                iris.data[predictions==target,feature_index_y],c=color,label=label,marker=marker)
                axflat.set_xlabel(feature_names[feature_index_x])
                axflat.set_ylabel(feature_names[feature_index_y])

f.tight_layout()
pl.show()

#2.6
from sklearn.mixture import GMM
from sklearn import datasets
from itertools import cycle, combinations
import matplotlib as mpl
import matplotlib.pyplot as pl
import numpy as np

# make_ellipses method taken from: http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_classifier.html#example-mixture-plot-gmm-classifier-py
# Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
# License: BSD 3 clause

def make_ellipses(gmm, ax, x, y):
    for n, color in enumerate('rgb'):
	row_idx = np.array([x,y])
	col_idx = np.array([x,y])
        v, w = np.linalg.eigh(gmm._get_covars()[n][row_idx[:,None],col_idx])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, [x,y]], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.2)
        ax.add_artist(ell)

iris = datasets.load_iris()

gmm = GMM(n_components=3,covariance_type='full', n_iter=20)
gmm.fit(iris.data)

predictions = gmm.predict(iris.data)

colors = cycle('rgb')
markers = cycle('^+o')
labels = ["Cluster 1","Cluster 2","Cluster 3"]
targets = range(len(labels))

feature_index=range(len(iris.feature_names))
feature_names=iris.feature_names
combs=combinations(feature_index,2)

f,axarr=pl.subplots(3,2)
axarr_flat=axarr.flat

for comb, axflat in zip(combs,axarr_flat):
 	for target, color, label,marker in zip(targets,colors,labels,markers):
  		feature_index_x=comb[0]
  		feature_index_y=comb[1]
  		axflat.scatter(iris.data[predictions==target,feature_index_x],
				iris.data[predictions==target,feature_index_y],c=color,label=label,marker=marker)
  		axflat.set_xlabel(feature_names[feature_index_x])
  		axflat.set_ylabel(feature_names[feature_index_y])
		make_ellipses(gmm,axflat,feature_index_x,feature_index_y)

pl.tight_layout()
pl.show()		

#2.7
import numpy as np
import matplotlib.pyplot as pl

from sklearn import decomposition
from sklearn import datasets
from itertools import cycle

iris = datasets.load_iris()
X = iris.data
Y = iris.target

targets = range(len(iris.target_names))
colors = cycle('rgb')
markers = cycle('^+o')

pca = decomposition.PCA(n_components=2)
pca.fit(X)

X = pca.transform(X)

for target,color,marker in zip(targets,colors,markers):
	pl.scatter(X[Y==target,0],X[Y==target,1],label=iris.target_names[target],c=color,marker=marker)

pl.legend()
pl.show()

