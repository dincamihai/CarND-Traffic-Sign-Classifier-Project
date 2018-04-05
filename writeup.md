# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./rst/output_9_1.png "Distribution of training/validation/test sets"
[upload1]: ./upload/8.png "Traffic Sign 1"
[upload3]: ./upload/3.png "Traffic Sign 3"
[upload4]: ./upload/10.png "Traffic Sign 4"
[upload6]: ./upload/6.png "Traffic Sign 6"
[upload7]: ./upload/7.png "Traffic Sign 7"
[prediction1]: ./rst/output_29_2.png                                            
[prediction2]: ./rst/output_29_3.png                                            
[prediction3]: ./rst/output_29_4.png                                            
[prediction4]: ./rst/output_29_5.png                                            
[prediction5]: ./rst/output_29_6.png                                            
[prediction6]: ./rst/output_29_7.png                                            
[prediction7]: ./rst/output_29_8.png                                            
[prediction8]: ./rst/output_29_9.png                                            
[prediction9]: ./rst/output_29_10.png

[luminance-formula]: https://wikimedia.org/api/rest_v1/media/math/render/svg/f84d67895d0594a852efb4a5ac421bf45b7ed7a8
[processed-image]: ./rst/output_18_2.png
[augmented-images]: ./images/augmentation.png
[augmented-dataset]: ./rst/output_15_0.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dincamihai/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I extracted the summary statistics from the data provided uning numpy:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
It is a bar chart showing how the data is distributed among the labels for all sets (training, validation and test)
On the left a random image from training set is shown.
On the right we can see a bar chart with labels on the horizontal and the number of examples on the y-axis.

```
blue - training set
orange - test set
green - validation set
```

![alt text][image9]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


After reading the [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) paper by Pierre Sermanet and Yann LeCun, I decided to use their finding that converting the images to grayscale improves the results of the network.

I am converting the images to grayscale using this formula:

![alt_text][luminance-formula]
that I found in [Grayscale - Wikipedia](https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale)

After converting from 3 channels to 1 channel using the formula above, I normalize the image (mu=0, sigma=1) in order to help the optimizer during the training.

Here is an example of a traffic sign image original - luminance - normalized

![alt text][processed-image]

I decided to generate additional data because some labels had less examples than others and this would lead to poorer results for those labels.

To add more data to the the data set, I used the following techniques in combination (with random parameters on each pass):

 - rotation
 - shift (including color)
 - gaussian filter
 
The goal was to obtain a dataset that has an approximately uniform distribution of examples per label.
I've tried to use image processing techniques that would generate images from the same distribution of the original images.
For example, I have avoided operations like flipping or excessive rotation because the goal of the network is to identify road signs in their normal position and not in any unusual position.
The augmentation could be improved by adding more types of distorsions like noise, brightnes and/or contrast.
The distorsions could be applied in random order to increase the variation.

Here is an example of an original image [first left] and augmented images generated from it:

![alt text][augmented-images]

The augmented dataset has approximately equal number of examples per label so the network will treat all labels as equally important.

![alt text][augmented-dataset]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layers                        | Description	      					                       | Layer name |
|:-----------------------------:|:---------------------------------------------:| :----------|
| Input                         | 32x32x1 Grayscale image   							             | x          |
| Convolution 5x5     	         | 1x1 stride, valid padding, outputs 28x28x6 	  | l1         |
| RELU					                     |												                                   | l1a        |
| Avg pooling	      	           | 2x2 stride, outputs 14x14x6 				              | p1         |
| Convolution 5x5	              | 1x1 stride, valid padding, outputs 10x10x16   | l2         |
| RELU                          |                                               | l2a        |
| Avg pooling	                  | 2x2 stride, outputs 5x5x16 				               | p2         |
| Avg pooling	                  | 2x2 stride, outputs 14x14x6 				              | p11        |
| Avg pooling	                  | 2x2 stride, outputs 14x14x6 				              | p12        |
| Fully connected layer         | 400                                           | fc1        |
| RELU                          |                                               | fc1a       |
| Dropout                       | keep_prob=0.5                                 | fc1drop    |
| Fully connected layer         | 120                                           | fc2        |
| RELU                          |                                               | fc2a       |
| Dropout                       | keep_prob=0.5                                 | fc2drop    |
| Logits                        | 43                                            | logits     |
| Softmax                       |                                               |            |

 
The layers are connected like this:

````
x ---- l1 - l1a - p1 ---- l2 - l2a - p2 ---- fc1 - fc1a - fc1drop ---- fc2 - fc2a - fc2drop ---- logits - softmax
                     \----- p11 - p12 -----/
````

The architecture described above is inspired from [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) paper by Pierre Sermanet and Yann LeCun

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an the "Adam" optimizer with the default parameters.
I've trained the network 10 epochs at a time while keeping the parameters from the previous 10 epochs training.
The learning rate, batch size and keep_prob were varied between the trainings.
In total, it was trained 40 epochs starting with higher learning rate, 0.006, and ending with 0.001. The batch size was also varied from 64 to 512.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

By using the architecture from the paper mentioned above I already had good results. I've spent more time in finding a good way of generating the fake data. The parameters for rotation and shift are quite important because I've noticed that is even possible to generate fake data that would lead the network to have better accuracy on the validation set than the training set.
This means that the training set is too "hard" to predict compared to normal data which indicates that the applied filters parameters are too high. (eg: too much rotation)

The learning rate is also important and needs to be reduced when the accuracy oscilates.

Adding the dropout layers helped when the network produced very good accuracy on the training set but much lower accuracy on the validation set.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.970
* test set accuracy of 0.94

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

At first I did not pass the output from the first convolution layer to the classifier. Adding this modification improved the accuracy by, as described in the paper mentioned above, it passes small details together with the more complex shapes and patterns from layer 2 to the classifier.

* What were some problems with the initial architecture?

Big difference between training and validation accuracy. Fixed with the dropout layers as described above.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

My initial model suffered from overfitting.

* Which parameters were tuned? How were they adjusted and why?

I've tuned the learning rate. Higher learning rate speeded up the initial training.
Lowering the learning rate and keeping the pretrained weights and biases helped in the following training sessions to increase the accuracy which was starting to oscilate.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The convolution layers help by being able to learn similar patterns in different positions in the image (by sharing the weights).
The dropout layers help reducing the high variation problem (overfitting) and helps the model generalize better (perform better on images not seen before)

If a well known architecture was chosen:
* What architecture was chosen?

The model architecture was inpired from [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) paper by Pierre Sermanet and Yann LeCun

* Why did you believe it would be relevant to the traffic sign application?

I've chose this architecture because it holds the record in identifying the traffic signs with 99.17% accuracy

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

High training accuracy shows that the model is complex enough to fit the training data.
Getting high accuracy on the validation data shows that the model is also able to generalize and predict new data and helps in the parameter tuning process.
Once we are satisfied with the model's performance, before putting it in a production system, we can use the test set to make sure the model will continue to work well with new data.
Getting high accuracy on the test set gives us the confidence that the model is ready for production.
In my case, with 94% accuracy in the test set shows that there's still place for improvement. We might need more training data since the model was able to perform much better on the training set and the validation set.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found in google street view on the streets in Germany:

![alt text][upload1] ![alt text][upload3] ![alt text][upload4] 
![alt text][upload6] ![alt text][upload7]

I've only used traffic signs from one of the 43 classes that the network was trained for. Of course, using other signs would not be predicted correctly because the network is not aware of them and would try to fit them in one of the knows classes.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt_text][prediction1] ![alt_text][prediction2]
![alt_text][prediction3] ![alt_text][prediction4]
![alt_text][prediction5] ![alt_text][prediction6]
![alt_text][prediction7] ![alt_text][prediction8]
![alt_text][prediction9]


The model was able to correctly identify all the images which means 100% accuracy. This is good but not surprising because the images are of good quality (not blured, taken during day time, good framed)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 76th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a 80km/h speed limit sign (probability of 0.69), and the prediction is correct. All top five predictions for the first image are for signs from the same class (th "speed limit" class) because they actually look similar.

For the rest of the images the network is 100% confident about the prediction.
Even for the 50km/h speed limit. Perhaps this is because this image is not horizontally squashed like the 80km/h image (which brings the 8 closer to the 0 and makes it harder to identify)

The top five softmax probabilities for all the images were:

```
| Probability | Sign name [label]                                  |
|------------------------------------------------------------------|
| 0.688       | Speed limit (80km/h) [5]                           |
| 0.211       | Speed limit (50km/h) [2]                           |
| 0.076       | Speed limit (100km/h) [7]                          |
| 0.010       | Speed limit (70km/h) [4]                           |
| 0.010       | Speed limit (30km/h) [1]                           |
|------------------------------------------------------------------|

| Probability | Sign name [label]                                  |
|------------------------------------------------------------------|
| 1.000       | Priority road [12]                                 |
| 0.000       | Roundabout mandatory [40]                          |
| 0.000       | Speed limit (50km/h) [2]                           |
| 0.000       | Speed limit (30km/h) [1]                           |
| 0.000       | Speed limit (20km/h) [0]                           |
|------------------------------------------------------------------|

| Probability | Sign name [label]                                  |
|------------------------------------------------------------------|
| 1.000       | Speed limit (50km/h) [2]                           |
| 0.000       | Speed limit (30km/h) [1]                           |
| 0.000       | Speed limit (80km/h) [5]                           |
| 0.000       | Speed limit (100km/h) [7]                          |
| 0.000       | Keep right [38]                                    |
|------------------------------------------------------------------|

| Probability | Sign name [label]                                  |
|------------------------------------------------------------------|
| 1.000       | Children crossing [28]                             |
| 0.000       | Bicycles crossing [29]                             |
| 0.000       | Pedestrians [27]                                   |
| 0.000       | Right-of-way at the next intersection [11]         |
| 0.000       | Speed limit (60km/h) [3]                           |
|------------------------------------------------------------------|

| Probability | Sign name [label]                                  |
|------------------------------------------------------------------|
| 1.000       | Keep right [38]                                    |
| 0.000       | Priority road [12]                                 |
| 0.000       | Roundabout mandatory [40]                          |
| 0.000       | Yield [13]                                         |
| 0.000       | Speed limit (50km/h) [2]                           |
|------------------------------------------------------------------|

| Probability | Sign name [label]                                  |
|------------------------------------------------------------------|
| 1.000       | No entry [17]                                      |
| 0.000       | Stop [14]                                          |
| 0.000       | No passing for vehicles over 3.5 metric tons [10]  |
| 0.000       | Speed limit (100km/h) [7]                          |
| 0.000       | Turn right ahead [33]                              |
|------------------------------------------------------------------|

| Probability | Sign name [label]                                  |
|------------------------------------------------------------------|
| 1.000       | Go straight or right [36]                          |
| 0.000       | Ahead only [35]                                    |
| 0.000       | Children crossing [28]                             |
| 0.000       | Turn left ahead [34]                               |
| 0.000       | Speed limit (60km/h) [3]                           |
|------------------------------------------------------------------|

| Probability | Sign name [label]                                  |
|------------------------------------------------------------------|
| 1.000       | Road work [25]                                     |
| 0.000       | Bumpy road [22]                                    |
| 0.000       | Keep left [39]                                     |
| 0.000       | Dangerous curve to the right [20]                  |
| 0.000       | Priority road [12]                                 |
|------------------------------------------------------------------|

| Probability | Sign name [label]                                  |
|------------------------------------------------------------------|
| 1.000       | No entry [17]                                      |
| 0.000       | Speed limit (60km/h) [3]                           |
| 0.000       | Speed limit (50km/h) [2]                           |
| 0.000       | Speed limit (30km/h) [1]                           |
| 0.000       | Speed limit (20km/h) [0]                           |
|------------------------------------------------------------------|
```


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

In the notebook I'm printing out the feature map from the first convolutional layer (l1).
It seems that this layer is interested in changes in contrast (edges)

Convolutional layer 2 did not produce any output that a human would easily understand.
