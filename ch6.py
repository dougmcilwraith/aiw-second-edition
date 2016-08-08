#6.3
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import perceptron

#Let's set up our data and our target
data = np.array([[0,1],[0,0],[1,0],[1,1]])
target = np.array([0,0,0,1])

#6.4
p = perceptron.Perceptron(n_iter=100)
p_out = p.fit(data,target)
print p_out
msg = ("Coefficients: %s, Intercept: %s")
print msg % (str(p.coef_),str(p.intercept_)) 

#6.5
colors = np.array(['k','r'])
markers = np.array(['*','o'])
for data,target in zip(data,target):
	plt.scatter(data[0],data[1],s=100,c=colors[target],marker=markers[target])

#Need to calculate a hyperplane the straight line as it intersects with z=0
#Recall that our optimisation is solving z=m1x + m2y + c
#If we want to understand the straight line created at the intersection with the viewing plane of x and y (where z=0)
#0=m1x + m2y +c
#m2y -m1x -c
#y = -m1/m2x - c/m2

grad = -p.coef_[0][0]/p.coef_[0][1]
intercept = -p.intercept_/p.coef_[0][1]

x_vals = np.linspace(0,1)
y_vals = grad*x_vals + intercept
plt.plot(x_vals,y_vals)
plt.show()

#6.X A HANDBUILT multilayer perceptron (NOT REPRODUCED IN THE BOOK).
data = np.array([[0,1],[0,0],[1,0],[1,1]])
target = np.array([1,0,1,0])

colors = np.array(['k','r'])
markers = np.array(['*','o'])
for _data,_target in zip(data,target):
	plt.scatter(_data[0],_data[1],s=100,c=colors[_target],marker=markers[_target])

plt.xlabel('x_1')
plt.ylabel('x_2')

#Let's plot the hand built boolean classifier
grad = -1
intercept = 0.5

x_vals = np.linspace(0,1)
y_vals = grad*x_vals + intercept
#plt.scatter(data[:,0],data[:,1],c=colors[target])
plt.plot(x_vals,y_vals,'b')

grad = -1
intercept = 1.5
x_vals = np.linspace(0,1)
y_vals = grad*x_vals + intercept
plt.plot(x_vals,y_vals,'r')

plt.xlabel('x_1')
plt.ylabel('x_2')

plt.text(0.8,-0.7,"x_2 = -x_1 + 0.5")
plt.text(0,1.65,"x_2 = -x_1 + 1.5")
plt.text(0.4,0.5,"y_1 = 1")
plt.text(0.8,1.5,"y_1 = 0")
plt.text(0,-0.5,"y_1 = 0")
plt.show()

#6.6
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure.modules import BiasUnit

import random

#Create network modules
net = FeedForwardNetwork()
inl = LinearLayer(2)
hidl = SigmoidLayer(2)
outl = LinearLayer(1)
b = BiasUnit()

#6.7
#Create connections
in_to_h = FullConnection(inl, hidl)
h_to_out = FullConnection(hidl, outl)
bias_to_h = FullConnection(b,hidl)
bias_to_out = FullConnection(b,outl)

#Add modules to net
net.addInputModule(inl)
net.addModule(hidl);
net.addModule(b)
net.addOutputModule(outl)

#Add connections to net and sort
net.addConnection(in_to_h)
net.addConnection(h_to_out)
net.addConnection(bias_to_h)
net.addConnection(bias_to_out)
net.sortModules()

#6.8
#input data
d = [(0,0),
     (0,1),
     (1,0),
     (1,1)]

#target class
c = [0,1,1,0]

data_set = SupervisedDataSet(2, 1) # 2 inputs, 1 output

random.seed()
for i in xrange(1000):
    r = random.randint(0,3)
    data_set.addSample(d[r],c[r])

backprop_trainer \
	= BackpropTrainer(net, data_set, learningrate=0.1)

for i in xrange(50):
    err = backprop_trainer.train()
    print "Iter. %d, err.: %.5f" % (i, err)

#6.9
print "[w(x_1,j=1),w(x_2,j=1),w(x_1,j=2),w(x_2,j=2)]: " + str(in_to_h.params)
print "[w(j=1,j=3),w(j=2,j=3)]: "+str(h_to_out.params)
print "[w(x_b,j=1),w(x_b,j=2)]: "+str(bias_to_h.params)
print "[w(x_b,j=3)]:" +str(bias_to_out.params)

#6.10
print "Activating 0,0. Output: " + str(net.activate([0,0]))
print "Activating 0,1. Output: " + str(net.activate([0,1]))
print "Activating 1,0. Output: " + str(net.activate([1,0]))
print "Activating 1,1. Output: " + str(net.activate([1,1]))


###########
# From here onwards:RBMS

# Original Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

#6.12
def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [[[0, 1, 0],[0, 0, 0],[0, 0, 0]],
			[[0, 0, 0],[1, 0, 0],[0, 0, 0]],
			[[0, 0, 0],[0, 0, 1],[0, 0, 0]],
			[[0, 0, 0],[0, 0, 0],[0, 1, 0]]]
    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()
    X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

#6.11
digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
X, Y = nudge_dataset(X, digits.target)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2,random_state=0)

#6.13
# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

###############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X_train, Y_train)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)

#6.14
# Evaluation
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))

print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))


#6.15
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    #print(i)
    #print(comp)
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()
