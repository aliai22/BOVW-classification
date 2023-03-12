# Image Classification Using Bag of Visual Words


## Abstract
This project implements the *Bag of Visual Words* (BOVW) technique to retrieve information (feature descriptors) from the image and then train a classifier (SVM & Random Forest for this project) on image histograms to predict the class of a given data sample. The same technique was applied on two different datasets, *objects dataset* and *flowers dataset*. The object dataset was already splitted into training and testing parts, however, the other one was not. Hence, it was splitted into *80-20 train-test* portions before proceeding forward. Moreover, all images were resized to a single dimension *(250,250)*.

The code relies on:

* *open-cv python* for reading and plotting images
* *SIFT* from open-cv python for local feature extraction
* *k-means* from scikit learn for generating vocabulary via clustering
* *SVM* and *Random Forest* from scikit learn as classification models

A detailed view of working methodology can be seen in the figure below.

![Untitled Diagram drawio](https://user-images.githubusercontent.com/127010479/224502231-e64cc19e-7861-44e2-9f05-5729a527e781.png)

## Prerequisites
* The dataset should have following structure, where all images belonging to one class are in same folder.
```
|-- MainFolder
.
|----|train
|       |-- class1
|       |-- class2
|       |-- class3
...
|       └-- classN
|----|test
|       |-- class1
|       |-- class2
|       |-- class3
...
|       └-- classN
```
* To execute the code, you should have necessary libraries installed in the environment you are using. 
