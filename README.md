# HandWritten-Digit-Classification

Handwritten digit recognition is a research problem in the domain of Pattern recognition and computer vision. Human handwritten digit recognition is a complex problem as it may carry different flavors of writing a single word. This technique can be experimented in detecting the characters in long documents, visuals and touch screens.
This is basically a sub problem of image recognition and one can also translate a handwritten character into a digital document.
Our model will classify a handwritten digit into 0 to 9 digits and accuracy score will explain how good classification is.

![image](https://user-images.githubusercontent.com/55511307/124384645-beb6ac00-dceb-11eb-91fa-c4362fbe11b2.png)

LITERATURE REVIEW:

1.	Back Propagation for Handwritten Digit Recognition [1] on zip codes was proposed and it resulted in low training time because of redundant data and constraints. Back propagation was implemented to get the weights and the network of connections.
2.	Multilayer perceptron (MLP) model for recognition of handwritten characters was based on neural networks. This method has the limitations like it stuck on local or global minima while separating the margins. [2]

DATA SET:

Data set used in this project is MNIST from kaggle.
Data has 42,000 training examples with 784 features. 
Each row in dataset is a label of digit against its 28x28[height x width] framed pixels(features).
Each digit is translated into 28x28 frame and stored in csv file as one row labeled with features as 1x28…14x25…..28x28. [3]
Data visualizations shows that data is balanced as the distribution of each label(digit) is uniform.

BASELINE:

Our data has large input dimensional space e.g 784 feature so we tried to implement SVM for classification as a baseline. SVMs are good choice for large dimensional data but we ended up getting too slow results as SVM’s training time becomes cubic for large data sets. 

MAIN APPROACH: 

SVMs are decent choice for image recognition problems [4] but its complexity grows with the size of data [5]. Our data set has 42,000 training examples which produced outputs in cubic time as discussed in baseline.
To avoid this problem, we used a technique proposed at University of California [6] which is to train the SVM on Nearest Neighbors (NN).
Initial inputs are fed to the NNs where pruning happened on the data and then SVMs were introduced on more relevant but smaller set of examples where careful boundary was needed to differentiate.
Since our data has too much dimensions therefore, we used a technique called Principal Component Analysis (PCA) for reducing the dimensions of the training examples. Using the PCA module by scikit-learn we extracted the first 60 principal components of the original data.
These reduced initial 60 components are enough for the interpretation of 96% information of the whole data.
I used train_test_split module to split data into train and test sets. Here test_size=0.4 which means to keep 40% data in test set and random_state=42 for reproducing the same results.

ALGORITHM:

For each query we initially check the Euclidian distance of the test point form all of the present training examples and then we pick the K neighbors which are nearest or in other words whose distance is the smallest. If all of the K nearest neighbors carry the same label then we are done and output those results. Else we transform our distance matrix into kernel matrix and then applied multiclass SVM for the solution.
In the SVM phase we are using the Radial base function (RBF) kernel due to the non-linearity of distance matrix calculated for 2 neighbors. It measures the how close two points are.
RBF works as: 
 ![image](https://user-images.githubusercontent.com/55511307/124384762-400e3e80-dcec-11eb-821d-bc0d5b8f787c.png)

It carries values between 0 and 1.
Here σ is the variance and ||X₁ - X₂|| is the Euclidean distance between two points
As similarity decreases then the distance between those two points also decreases. In our case 

![image](https://user-images.githubusercontent.com/55511307/124384773-4a303d00-dcec-11eb-8f2a-8959c5413bce.png)

 
So X1 and X2 are similar and their Kernel values are large then the X1 and X3.
 
![image](https://user-images.githubusercontent.com/55511307/124384780-54ead200-dcec-11eb-84f7-21aff22bd410.png)

In the implementation we kept k=2 which means only 2 neighbors are being tested for the classification. Here few samples are being tested initially for NN and where NN will find tough for boundary making then SVM will join hands for classification.

Evaluation Metric:
To test the performance of our model we kept the accuracy score and time complexity as our evaluation criteria.
Accuracy score means how correctly our model classified the train data.

RESULTS AND ANALYSIS: 

Baseline model was to apply only SVMs on the data set. We ended up getting very slow but quite accurate results. It produced with 93% accurate result with parameters as rbf kernel and C=1.
But after implementing the main approach we got the accuracy of 96% with output faster than that of baseline.
It means our KNN-SVM model classified 96% of the training data correctly.
 
ERROR ANALYSIS: 

Initially while executing the program on a PC with 8-GB RAM and hard disk of 512GB we got memory overflow error saying the memory size for the allocation of  (42000,784) is not possible on this system.
![image](https://user-images.githubusercontent.com/55511307/124384812-73e96400-dcec-11eb-9db7-2d3780e4d3fe.png)
 

When we switched to a PC with RAM 8GB DDR4 and hard disk 128 SSD we got the desired output without any errors.

Reason was that on initial system page sizes were small. Basically, that system was in overcommit handling mode.

Future Work: 
Further modification is possible in the model as we worked for the accuracy and time complexity issues but one can implement a live example in which human writes a digit on some input screen and our model would predict which digit this is.
Also, this can further be propagated in Image recognition and computer vision applications.
References: 
[1]https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf
[2] https://link.springer.com/content/pdf/10.1007/s10032-002-0094-4.pdf
[3] https://colah.github.io/posts/2014-10-Visualizing-MNIST/
[4]http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.441.6897&rep=rep1&type=pdf
[5]https://www.sciencedirect.com/science/article/pii/S0925231207002962#:~:text=Support%20vector%20machine%20(SVM)%20is,the%20size%20of%20data%20set.
[6] https://medium.com/the-andela-way/applying-machine-learning-to-recognize-handwritten-characters-babcd4b8d705





