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

#### Q1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

##### Augmentation

Based on the exploratory analysis of the dataset, I decided to first augment my data by generating additional data. Failure to do so impacts the accuracy of the model because the more represented classes stand a higher chance of being predicted.

However, rather than blindly augmenting images for all the classes, I decided to only do it for the minority classes i.e. classes that were below 1.6% of the whole dataset.

The image augmentation/class balancing techniques (based on [OpenCV](https://opencv.org/)) include:

* **Rotation**: `cv2.getRotationMatrix2D()` & `cv2.warpAffine()`
* **Horizontal flip**: `cv2.getRotationMatrix2D()` & `cv2.warpAffine()`
* **Zooming in/out**: `cv2.warpPerspective()`
* **Affine transformation**: `cv2.findHomography()` & `cv2.warpPerspective()`

One important point is that not all classes could be rotated or horizontally flipped as it either totally renders the road sign useless or makes it part of the opposite class. On the other hand, this behavior was capitalised upon by generating images for the opposite classes e.g. *keep left <--> keep right*. The below image shows some examples of the application of the aforementioned techniques:

![Augmentation examples](writeup-images/data-augmentation.png)

Below is the result after image augmentation/dataset balancing:

![Dataset augmentation 1](writeup-images/data-augmentation3.png)


![Dataset augmentation 2](writeup-images/data-augmentation2.png)


##### Pre-processing

Following image pre-processing techniques were applied (these techniques are a combination of generally known image enhancement techniques and techniques mentioned in [Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) paper):

1. **Grayscaling**: Images were grayscaled as they seem to increase the accuracy of the model by being indifferent to color. Also, it dramatically reduces image size as there's only 1 color channel, which helps with speeding up the training cycle. `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)` function was used for grayscaling.

2. **Histogram Normalization**: Images were made sharper by adjusting their contrast via Histogram Normalization technique. Instead of applying a single contrast value across the image, different values are applied to each different tiles of the image through Contrast Limited Adaptive Histogram Equalization (CLAHE) technique. This results in much sharper contrast. `cv2.createCLAHE()` function was used to achieve CLAHE.

3. **Adaptive Threshold**: To reduce noise, Adaptive Thresholding was applied that further helps in filtering out unnecessary pixels. `cv2.adaptiveThreshold()` function was employed.

4. **Normalization**: Normalization helps the optimizer to reduce the loss as quickly as possilbe as it doesn't need to do too much searching. Hence, the images were normalized. However, using the prescribed `(pixel - 128)/ 128`technique didn't really achieve `0` mean & `unit` variance. Instead,  `sklearn.preprocessing` package's `scale()` method was used that resulted in achieving `0` mean & `1` variance.

As an example, consider the following image that displays the application of the above techniques:

![Preprocessing example](writeup-images/data-preprocessing.png)

The following image displays the application of the aforementioned techniques on *test* and *validation* datasets:

![Preprocessing pipeline](writeup-images/data-preprocessing2.png)

#### Q2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)? 

The architecture of the CNN is based on the famous [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) architecure for [MNIST](http://yann.lecun.com/exdb/mnist/) digit recognition. However, to increase the depth of the network, one more convolution layer was added. Further, to reduce overfitting and make the network more robust, [L2 Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) & [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) were added. 

My final model architecture consisted of 6 layers as descried in the following table:

| Layer No.|Component| Description	        					| 
|:--:|:--------:|:----------------------------------:| 
|1| Convolution 5x5   | Input 32x32x1, 1x1 Stride, SAME padding, Output 32x32x16, Initializer Xavier|
|| RELU| Activation|
||Max Pooling 2x2| Input 32x32x16, 2x2 Stride, Output 16x16x16| 				
|2| Convolution 5x5   | Input 16x16x16, 1x1 Stride, SAME Padding, Output 16x16x32, Initializer Xavier|
|| RELU| Activation|
||Max Pooling 2x2| Input 16x16x32, 2x2 Stride, Output 8x8x32|
|3| Convolution 5x5   | Input 8x8x32, 1x1 Stride, SAME Padding, Output 8x8x64, Initializer Xavier|
|| RELU| Activation|
||Max Pooling 2x2| Input 8x8x64, 2x2 Stride, Output 4x4x64|
|4| Fully Connected| Input 1024, Output 120, Initializer Xavier|
|| Dropout| Keep Probability 0.5|
|5| Fully Connected| Input 120, Output 84, Initializer Xavier|
|| Dropout| Keep Probability 0.5|
|6| Fully Connected| Input 84, Output 43, Initializer Xavier|

#### Q3. Describe how you trained your model?

The model was trained by feeding in the pre-processed `65,518` grayscale images with a batch size of `128` and `150` epochs. Initially a fixed learning rate of `0.001` was used but later on, an exponentially decaying learning rate with a factor of `0.90` and initial rate of `0.001`was used. Also, all the weights were initialized using [Xavier](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.207.2059&rep=rep1&type=pdf) initializer that aims to keep the scale of the gradients roughly the same in all layers. For optimization, [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) optimizer was used as it is known to perform better than the classical [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) optimizer.

#### Q4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93?

To solve the problem at hand, I started off with the LeNet architecture as is because it is renowned for solving a similar problem i.e. recognizing images of hand written numbers from `0-9`. Based on *convolution*, such an architecture helps to decrease the breadth of the network and enables to increase the depth without compromising the accuracy. Different *learning rates*, *batch sizes* and *number of epochs* were tried. However, just by using the LeNet as is, the validation accuracy was not reaching the required minimum of `0.93`.

Consequently, I started incorporating other suggested techniques, namely *dropout*, *L2 Regularization*,*decaying learning rate*, *Xavier initialising of weights* and *direct feed-in* from convolution layers to the fully connected layers (as described in the  Pierre Sermanet and Yann LeCun paper). Upon experimentation, the introduction of *direct feed-in* made the model behave very unreliably and the validation accuracy was stuck at a very low number despite training the model for a number of epochs. It seems that there might be some fundamental implementation issue in the code that is creating this issue. So it was decided to drop the idea of adding *direct feed-in* for now. After much trail & error, values of `0.001`,`0.90`,`128`,`150` were chosen as *initial learning rate*,*decay rate*,*training batch size*,*number of epochs* were chosen respectively as the most appropriate values for the model. These hyperparameter values resulted in achieving a *validation accuracy* of `0.962` and a *test accuracy* of `0.939`.
 

###Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 