# Traffic Sign Classifier

## Overview   
   
This writeup reflects upon the **Finding Lane Line** project by explaining how the image processing pipeline works, identifying any shortcomings and proposing potential improvements. 


## Project Goals

The main goals of the **Traffic Sign Classifier** project are:

1. Development of a code pipeline that uses *deep learning* for identifying German road signs  
2. A description of the above pipeline along with an analysis of the approach taken as well as the results

## Reflection

### Pipeline Overview
1. Load the traffic dataset (download from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip))
2. Explore, summarize and visualize the dataset
3. Design (using [Tensorflow](https://www.tensorflow.org/)), train and test a CNN deep learning image classification model 
4. Use the model to make predictions on new images acquired from the web
5. Analyze the softmax probabilities (prediction certainty) of the new images

### Project Code
The image classifier has been implemented as a Jupyter notebook & can be viewed [here](https://github.com/wkhattak/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb).

## Rubric Points

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used a combination of Pandas library and Python to calculate summary statistics of the traffic
signs data set as shown below:

```python
...
n_train = X_train_working_copy.shape[0]
n_validation = X_valid_working_copy.shape[0]
image_shape =  str(X_train_working_copy.shape[1]) + ' x ' + str(X_train_working_copy.shape[2]) + ' x ' + str(X_train_working_copy.shape[3])
n_classes = len(np.unique(y_train_working_copy))

print("Number of training images =", n_train)
print("Number of validation images =", n_validation)
print("Image data shape (Height x Width x Color Channels) =", image_shape)
print("Number of classes =", n_classes)
```
![Train validate summary image](writeup-images/data-exploration.png)

```python
...
n_test = X_test_working_copy.shape[0]
print("Number of test images =", n_test)
```
![Test summary image](writeup-images/data-exploration-test.png)

To get a better idea of the training dataset, the following visualization was generated that shows the image classes along with a sample image from each class:

![Test summary image](writeup-images/data-exploration2.png)

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the dataset. It is a bar chart that shows the distribution of the classes in the training dataset. It is quite clear from the class distribution that we don't have a balanced training dataset.

![Test summary image](writeup-images/data-exploration3.png)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

Based on the exploratory analysis of the dataset, I decided to first augment my data by generating additional data. Failure to do so impacts the accuracy of the model because the more represented classes stand a higher chance of being predicted.

However, rather than blindly augmenting images for all the classes, I decided to only do it for the minority classes i.e. classes that were below 1.6% of the whole dataset.

The image augmentation/class balancing techniques include:

*Rotation
*Horizontal flip
*Zooming in/out
*Affine transformation

The below image shows some examples of the application of the aforementioned techniques:
![Augmentation examples](writeup-images/data-augmentation.png)