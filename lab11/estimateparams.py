#!/usr/bin/env python3

##############
#### Michael Rockhold, UW ID 1573637
##############

import numpy as np
import re

from skimage import io, feature, filters, exposure, color

from sklearn import svm, metrics, preprocessing
from sklearn.svm import SVC

from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class ImageClassifier:

    def __init__(self, score):
        # Create a support vector classifier
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                            {'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                            {'kernel': ['poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]

        self.score = score
        self.classifier = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)


    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)

        #create one large array of image data
        data = io.concatenate_images(ic)

        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return(data,labels)

    def extract_image_features(self, data):
        # extract feature vector from image data
        def extract(image):
            return feature.hog(color.rgb2gray(image), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')
        return [extract(image) for image in data]

    def train_classifier(self, train_data, train_labels):
        # train the classifier
        self.classifier.fit(train_data, train_labels)
        self.fit_report()

    def predict_labels(self, data):
        # Please do not modify the header
        # predict labels of test data using trained model in self.classifier
        return self.classifier.predict(data)

    def fit_report(self):
        print("Best parameters set found on development set:")
        print()
        print(self.classifier.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = self.classifier.cv_results_['mean_test_score']
        stds = self.classifier.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, self.classifier.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

    def predict_report(self, X_test, y_test):
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, self.classifier.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

def main():

    # Set the parameters by cross-validation
    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        img_clf = ImageClassifier(score)

        # load images
        (x_train_raw, y_train) = img_clf.load_data_from_folder('./train/')
        (x_test_raw, y_test) = img_clf.load_data_from_folder('./test/')

        # convert images into features
        X_train = img_clf.extract_image_features(x_train_raw)
        X_test = img_clf.extract_image_features(x_test_raw)

        # train model
        img_clf.train_classifier(X_train, y_train)

        img_clf.predict_report(X_test, y_test)


if __name__ == "__main__":
    main()
