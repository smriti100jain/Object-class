# Object_class

Object Classification in Images

In this project, we made Image Classification System which has been trained to identify objects in the images namely, Airplane, 
Car and Horse. The system can do Binary Classification and Multi-Class Classification. The Binary Classifi- cation predicts
whether the image has aforementioned objects or not.The Multi- Class Classification predicts the particular type of object in the 
image and if image doesn’t have any of the mentioned objects it is labeled as ”Other” Class.

DATA SET USED:

We have a data-set of 6000 images, which consists of 964 images labeled as Class 1(Airplane), 1162 images labeled as Class 2(Car),
1492 images labeled as Class 3(Horse) and 2382 images labeled as Class 4(Others). Along with these labels we have two different 
pre-computed feature vectors for each image. They are Histogram of Oriented Gradients(HOG) features and Convolutional 
Neural Network(CNN) features. The HOG feature vectors are of size 5408 and CNN are of size 36865.
The HOG features are created by dividing images(231 × 231) into blocks of 17 × 17 pixels. For each block histogram of gradient 
is created with 8 orientation bins. Each block is represented by 32 values after normalization of histogram with adjacent histograms.
Hence the length of our feature vector is 5408(13 × 13 × 32). The CNN features are extracted using OverFeat software [5]. 
We didn’t look into much details on how these features were extracted.

We have also provided the 1). REPORT OF RESULTS
2). Codes for all the machine learning algorithms used for object classification

