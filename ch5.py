import numpy as np
import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc

#Listing 5.6
ground_truth_file_name = './ground_truth.dat'
probability_file_name = './probabilities.out'

ground_truth_file = open(ground_truth_file_name,'r')
probability_file = open(probability_file_name,'r')

ground_truth = np.array(map(int,ground_truth_file))
probabilities = np.array(map(float,probability_file))

ground_truth_file.close()
probability_file.close()

#from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
fpr, tpr, thresholds = roc_curve(ground_truth, probabilities)
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.show()
