# Project: Traffic Sign Classifier 
## Overview   
   
This project is about classifying road signs by first training a Convolutional Neural Network (CNN) classifier on a German traffic signs dataset & then testing the model on new images downloaded from the web. 

## How Does It Work?
The entire solution comprises the following steps:

1. Reading in the dataset that includes images for training, validation and testing [German signs dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip).
2. Augmenting the training dataset to increase the size of the training dataset.
3. Pre-processing the augmented dataset by applying techniques such as grayscaling, histogram normalization, and normalization.
4. Developing an architecture of the CNN classifier using TensorFlow.
5. Training the classifier through hyperparameter tuning.
6. Validating the model on the validation dataset.
7. Testing the model on test dataset.
8. Predicting the classes of 10 new traffic sign images downloaded from the web.


## Directory Structure
* **saved-models:** Directory containing trained model 
* **test-images:** Directory containing 10 new images for making predictions
* **writeup-images:** Directory containing images that appear in the project report
* **Traffic_Sign_Classifier.html:** Html output of the Python notebook
* **Traffic_Sign_Classifier.ipynb:** Python notebook containing the source code
* **README.md:** Project readme file
* **signnames.csv :** A csv file that contains the mapping between numerical class labels and their actual string labels
* **writeup.md:** Project writeup file containing detailed information about the inner workings of this project

## Requirements
* Python-3.5.2
* OpenCV-3.3.0
* Numpy
* Pandas
* TensorFlow
* Scikit-learn
* Pickle
* Matplotlib
* Jupyter Notebook


## Usage/Examples
Follow the instructions/code in the *Traffic_Sign_Classifier.ipynb* notebook.



## Troubleshooting

**tdqm**

To install, execute conda install -c anaconda tqdm

**libgtk**

The command import cv2 may result in the following error. `ImportError: libgtk-x11-2.0.so.0: cannot open shared object file: No such file or directory.` To address make sure you are switched into the correct environment and try `source activate [your env name];conda install opencv`. If that is unsuccessful please try `apt-get update;apt-get install libgtk2.0-0` (may need to call as sudo).

## License
The content of this project is licensed under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/us/deed.en_US).