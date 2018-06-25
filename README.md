# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/1.png
[image2]: ./output_images/2.png
[image3]: ./output_images/3.png


---
### Writeup / README

### Color Histogram

The code for color histogram extration is in the second code cell of the IPython notebook. Here is the results of color histogram of an car image:

![alt text][image1]


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the forth code cell of the IPython notebook. 

I started by reading in one image from each of the `vehicle` and `non-vehicle` dirctories. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the gray color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]


#### 2. Explain how you settled on your final choice of HOG.

I mainly focused on trying different orientation values by comparing the accuracies of the classifiers with different of them. The set used for the final results perfoms best: `orientation=11`, `pixels_per_cell=8`, and `cell_per_block=2`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features and color histogram features. The parameters for HOG features extraction is described above, and the parameters for color features are: `bin_spatial=16` and `histbin=32`.

I first loaded all car and non-car images from the `vehicle` and `non-vehicle` and got all of the HOG and color features from them by using `extract_features` function in the sixth code cell of the IPython notebook. I then created array stacks of feature vectors and labels and split them into training and test data sets.
After using `RobustScaler`, I fitted the training data set with `LinearSVC`, and got an accuracy of 0.9876 in the test data set.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I tried different window postions with different scales in order to cover cars of different distances in the image frame. The final window postions for 3 different scales are:

    | scale   |   window position (pixel)  |  cells_per_step  |
    -----------------------------------------------------------
    |   0.8   |    410 - 480               |     1            |
    -----------------------------------------------------------
    |   1.5   |    400 - 550               |     2            |
    -----------------------------------------------------------
    |   2.0   |    400 - 620               |     2            |


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. To optimize the performance of my classifier, I tried to find the balance between good accuracy and computational efficiency. I tried different parameter sets, and the final one is:  `orientation=11`, `pixels_per_cell=8`, `cell_per_block=2`, `bin_spatial=16`, and `histbin=32`.

Here are some example images of cars identification in the coloumn one of the plots below:

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/lingyun-wu/CarND-Project-05/blob/master/output_videos/project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
To filter false positives, I first used function `apply_threshold` to get off heated area with value less than 2. I then added up all heat map of current frame and previous 9 frames and did `apply_threshold` again to filter out noise which just show up accidently in single frame.

The images in third column of the plot above show the results of false positives being filtered and labels of each car.  


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are still some false positives and unstable bounding boxes in the video, which might be caused by false classifications. Becuase I don't have enough time right now, I will submit this version of video. The next step I will do is to try with some other threshold values for the heated map, and then try some other classifiers, like decision tree or deep learning method. Another thing is the computational efficiency of this pipeline is still not very good. It takes several seconds to work through one single frame, which is too much for real world driving. To solve this, one way is to use a better processor, and the other way is to improve the efficiency of the algorithm, which can be dug deeper in the futrue.
