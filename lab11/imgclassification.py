#!/usr/bin/env python3

##############
#### Michael Rockhold, UW ID 1573637
##############

import numpy as np
import re
import cv2 as cv2
from sklearn import svm, metrics, preprocessing
from skimage import io, feature, filters, exposure, color
from sklearn.model_selection import GridSearchCV

class ImageClassifier:

    def __init__(self):
        # Create a support vector classifier
        # {'C': 1, 'gamma': 0.001, 'kernel': 'linear'}
        self.classifier = svm.SVC(C=1, kernel='linear', gamma=0.001)

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
        return [self.features_from_image(color.rgb2gray(image)) for image in data]

    def extract_features_and_labels(self, dir):
        # load images
        (raw_data, labels) = self.load_data_from_folder(dir)
        # convert images into features
        data = self.extract_image_features(raw_data)
        return(data, labels)

    def train_classifier(self, train_data, train_labels):
        # train the classifier
        self.classifier.fit(train_data, train_labels)

    def predict_labels(self, data):
        # predict labels of test data using trained model in self.classifier
        return self.classifier.predict(data)

    def prep(self,dir):
        (train_data, train_labels) = self.extract_features_and_labels('./train/')
        # train model
        self.train_classifier(train_data, train_labels)
        return(train_data, train_labels)

    def predictAndReport(self, title, data, labels):
        predicted_labels = self.predict_labels(data)
        print(title)
        report(data, labels, predicted_labels)

    def features_from_image(self, image):
        return feature.hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')


def report(data, labels, predicted_labels):
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(labels, predicted_labels, average='micro'))

def main():

    img_clf = ImageClassifier()

    # train the classifier
    (train_data, train_labels) = img_clf.prep('./train/')

    # apply the trained classifier to the training data
    img_clf.predictAndReport("\nTraining results", train_data, train_labels)

    # test model
    (test_data, test_labels) = img_clf.extract_features_and_labels('./test/')

    img_clf.predictAndReport("\nTest results", test_data, test_labels)


if __name__ == "__main__":
    main()
