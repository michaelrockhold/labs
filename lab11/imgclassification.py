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
        def extract(image):
            return feature.hog(color.rgb2gray(image), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')
        return [extract(image) for image in data]

    def train_classifier(self, train_data, train_labels):
        # train the classifier
        self.classifier.fit(train_data, train_labels)

    def predict_labels(self, data):
        # Please do not modify the header
        # predict labels of test data using trained model in self.classifier
        return self.classifier.predict(data)

def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')

    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)

    # train model
    img_clf.train_classifier(train_data, train_labels)

    # apply the trained classifier to the training data
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))

    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTest results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))


if __name__ == "__main__":
    main()
