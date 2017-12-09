# CS 425. Project 5.
# Written by: Ksenia Burova
#
# SVC

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from matplotlib.colors import Normalize
from sklearn import svm
import matplotlib.pyplot as plt

############    Took this plotting fucntion from scikit-learn.org    #################
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_heatmap(filename, parameters):
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.plasma,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.82, vmax=1.0))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(parameters['gamma'])), parameters['gamma'], rotation=45)
    plt.yticks(np.arange(len(parameters['C'])), parameters['C'])
    plt.savefig(filename)
    plt.close()
######################################################################################

def sc_plot(x, y, dataset, labels, filename, p):
    colors= ["pink", "red", "grey", "purple", "blue", "green", "orange", "yellow", "teal", "brown", "magenta" ]
    plt.figure(1, figsize=(10,7))
    plt.subplot(p)
    plt.xlabel(x)
    plt.ylabel(y)
    for i, d in enumerate(dataset):
        plt.scatter(d[0], d[1], color=colors[int(labels[i])],
                    marker='o')

class Data:
    def __init__(self, filename, task, testfile=None):

        self.trainFeatures = np.array([])
        self.validFeatures = np.array([])
        self.testFeatures = np.array([])

        self.trainLabels = np.array([])
        self.validLabels = np.array([])
        self.testFeatures = np.array([])

        if task == 1:

            self.data = np.genfromtxt(filename, delimiter=',')
            self.data = np.delete(self.data, 1, axis=1)

            np.random.shuffle(self.data)

            self.features = self.data[:, :-1]
            self.labels = self.data[:, [-1]]
            self.split()
        elif task == 2:
            self.data = np.genfromtxt(filename, delimiter=' ')
            self.data = np.delete(self.data, [0, 1, 2] , axis=1)
            np.random.shuffle(self.data)

            self.features = self.data[:, :-1]
            self.labels = self.data[:, [-1]]
            self.split()
        else:
            self.trainFeatures = np.genfromtxt(filename, delimiter=' ')
            self.trainLabels = self.trainFeatures[:, [-1]]

            self.trainFeatures = np.delete(self.trainFeatures, -1, axis=1)
            self.trainFeatures = StandardScaler().fit_transform(self.trainFeatures)
            self.trainLabels = self.trainLabels.ravel()

            self.validFeatures = np.genfromtxt(testfile, delimiter=' ')
            self.validLabels = self.validFeatures[:, [-1]]

            self.validFeatures = np.delete(self.validFeatures, -1, axis=1)
            self.validFeatures = StandardScaler().fit_transform(self.validFeatures)
            self.validLabels = self.validLabels.ravel()


    def split(self):
        self.features = StandardScaler().fit_transform(self.features)
        self.trainFeatures, self.validFeatures, self.trainLabels, self.validLabels = train_test_split(
            self.features, self.labels, test_size=0.25, random_state=26)
        self.trainLabels = self.trainLabels.ravel()
        self.validLabels = self.validLabels.ravel()

        # self.validFeatures, self.testFeatures, self.validLabels, self.testLabels = train_test_split(
        #     self.testFeatures, self.testLabels, test_size=0, random_state=26)
        # self.testLabels = self.testLabels.ravel()
        # self.validLabels = self.validLabels.ravel()



parameters = {'kernel': ['rbf'],
              'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
              'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

# get to change these
fine_params1 = {'kernel': ['rbf'],
              'C': np.arange(0.25, 5, 0.25),
              'gamma': np.arange(0.025, 0.5, 0.025)}

#Problem 1: Coarse search --------------------------------------------------
print("Solving classification problem # 1")
d = Data("ionosphere.data", 1)
model = svm.SVC()
clf = GridSearchCV(model, parameters)
clf.fit(d.trainFeatures, d.trainLabels)

print(clf.best_params_)

y_true, y_pred = d.validLabels, clf.predict(d.validFeatures)
print(classification_report(y_true, y_pred))

scores = clf.cv_results_['mean_test_score'].reshape(len(parameters['C']),
                                                     len(parameters['gamma']))

plot_heatmap("coarse1.png", parameters=parameters)

# Problem 1: Fine search --------------------------------------------------

clf = GridSearchCV(model, fine_params1)
clf.fit(d.trainFeatures, d.trainLabels)

print(clf.best_params_)

y_true, y_pred = d.validLabels, clf.predict(d.validFeatures)
print(classification_report(y_true, y_pred))

scores = clf.cv_results_['mean_test_score'].reshape(len(fine_params1['C']),
                                                     len(fine_params1['gamma']))
plot_heatmap("Fine1.png", parameters=fine_params1)


# Problem 2: Coarse search --------------------------------------------------
print("Solving classification problem # 2")
d = Data("vowel-context.data", 2)
model = svm.SVC()
clf = GridSearchCV(model, parameters)
clf.fit(d.trainFeatures, d.trainLabels)

print(clf.best_params_)

y_true, y_pred = d.validLabels, clf.predict(d.validFeatures)
print(classification_report(y_true, y_pred))

scores = clf.cv_results_['mean_test_score'].reshape(len(parameters['C']),
                                                     len(parameters['gamma']))

plot_heatmap("coarse2.png", parameters=parameters)

# Problem 2: Fine search --------------------------------------------------

clf = GridSearchCV(model, fine_params1)
clf.fit(d.trainFeatures, d.trainLabels)

print(clf.best_params_)

y_true, y_pred = d.validLabels, clf.predict(d.validFeatures)
print(classification_report(y_true, y_pred))

scores = clf.cv_results_['mean_test_score'].reshape(len(fine_params1['C']),
                                                     len(fine_params1['gamma']))
plot_heatmap("Fine2.png", parameters=fine_params1)


# Problem 3: Coarse search --------------------------------------------------
print("Solving classification problem # 3")
d = Data("sat.trn", 3, "sat.tst")
model = svm.SVC()
clf = GridSearchCV(model, parameters)
clf.fit(d.trainFeatures, d.trainLabels)

print(clf.best_params_)

y_true, y_pred = d.validLabels, clf.predict(d.validFeatures)
print(classification_report(y_true, y_pred))

scores = clf.cv_results_['mean_test_score'].reshape(len(parameters['C']),
                                                     len(parameters['gamma']))

plot_heatmap("coarse3.png", parameters=parameters)

# Problem 2: Fine search --------------------------------------------------

clf = GridSearchCV(model, fine_params1)
clf.fit(d.trainFeatures, d.trainLabels)

print(clf.best_params_)

y_true, y_pred = d.validLabels, clf.predict(d.validFeatures)
print(classification_report(y_true, y_pred))

scores = clf.cv_results_['mean_test_score'].reshape(len(fine_params1['C']),
                                                     len(fine_params1['gamma']))
plot_heatmap("Fine3.png", parameters=fine_params1)

d.features = np.concatenate((d.trainFeatures,d.validFeatures),axis=0)
d.labels = np.concatenate((d.trainLabels,d.validLabels),axis=0)


# sc_plot('Attr1', 'Attr2', d.features[:, 1:3], d.labels, '1_2.png', 221)
# sc_plot('Attr3', 'Attr4', d.features[:, 3:5], d.labels, '3_4.png', 222)
# sc_plot('Attr2', 'Attr3', d.features[:, 2:4], d.labels, '2_3.png', 223)
# sc_plot('Attr0', 'Attr1', d.features[:, 0:2], d.labels, '0_1.png', 224)
#
# plt.savefig('attrs.png')
# plt.close()