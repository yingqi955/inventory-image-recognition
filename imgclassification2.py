#!/usr/bin/env python

##############
#### Your name: Yingqi Chen
##############

import numpy as np
import re, math
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color, measure, transform
from sklearn.linear_model import RANSACRegressor
import ransac_score

class ImageClassifier:
    
    def __init__(self):
        self.classifier = None

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
        # Please do not modify the header above

        # extract feature vector from image data

        ########################
        ######## YOUR CODE HERE
        ########################
        l = []
        for im in data:
            im_gray = color.rgb2gray(im)
       
            im_gray = filters.gaussian(im_gray, sigma=0.4)
            
            f = feature.hog(im_gray, orientations=10, pixels_per_cell=(48, 48), cells_per_block=(4, 4), feature_vector=True, block_norm='L2-Hys')
            l.append(f)
        

        feature_data = np.array(l)
        return(feature_data)
    
    
    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above

        # train model and save the trained model to self.classifier

        ########################
        ######## YOUR CODE HERE
        ########################
        self.classifer = svm.LinearSVC()
        self.classifer.fit(train_data, train_labels)

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels

        ########################
        ######## YOUR CODE HERE
        ########################
        predicted_labels = self.classifer.predict(data)
        return predicted_labels

    def line_fitting(self, data):
        # Please do not modify the header

        # fit a line the to arena wall using RANSAC
        # return two lists containing slopes and y intercepts of the line

        ########################
        ######## YOUR CODE HERE
        ########################
        slopes = []
        intercepts = []

        for img in data:
            # Convert the image to grayscale
            gray_image = color.rgb2gray(img)

            # Apply Gaussian filter with sigma around 3
            smoothed_image = filters.gaussian(gray_image, sigma=3)

            # Perform edge detection using Canny
            edges = feature.canny(smoothed_image)

            # Extract coordinates of edge pixels
            edge_coords = np.column_stack(np.where(edges))

            # Custom RANSAC-based line fitting function
            slope, intercept = self.ransac_line(edge_coords[:, 1], edge_coords[:, 0], num_samples=2, repeats=100, acceptable_dist=1)

            slopes.append(slope)
            intercepts.append(intercept)

        # Calculate the final estimated slope and intercept as the median of all estimates
        if slopes:
            slope = np.median(slopes)
            intercept = np.median(intercepts)
        else:
            # Return None if no line was detected
            slope = None
            intercept = None

        # Please do not modify the return type below
        return slope, intercept
    
    
    def ransac_line(self, x_coord, y_coord, num_samples, repeats, acceptable_dist):
        best_line = None
        max_inliers = 0

        for _ in range(repeats):
            # Choose a random subset of points
            random_indices = np.random.choice(len(x_coord), size=num_samples, replace=False)
            x_subset = x_coord[random_indices]
            y_subset = y_coord[random_indices]

            # Fit a line to the subset using polyfit
            line = np.polyfit(x_subset, y_subset, 1)
            slope = line[0]
            intercept = line[1]

            # Calculate the perpendicular distance from the line for all points
            distances = np.abs((slope * x_coord - y_coord + intercept) / np.sqrt(slope**2 + 1))

            # Count the inliers (points within acceptable_dist)
            inliers = np.sum(distances <= acceptable_dist)

            # Check if this model has more inliers than the previous best
            if inliers > max_inliers:
                max_inliers = inliers
                best_line = line

        if best_line is not None:
            best_slope = best_line[0]
            best_intercept = best_line[1]
        else:
            best_slope = None
            best_intercept = None

        return best_slope, best_intercept
    
def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    (wall_raw, _) = img_clf.load_data_from_folder('./wall/')
    
    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)
    
    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
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

    # ransac
    print("\nRANSAC results")
    print("=============================")
    s, i = img_clf.line_fitting(wall_raw)
    print(f"Line Fitting Score: {ransac_score.score(s,i)}/10")

if __name__ == "__main__":
    main()