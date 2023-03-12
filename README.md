# Image Classification Using Bag of Visual Words


## Abstract
This project implements the *Bag of Visual Words* (BOVW) technique to retrieve information (feature descriptors) from the image and then train a classifier (SVM & Random Forest for this project) on image histograms to predict the class of a given data sample. The same technique was applied on two different datasets, *objects dataset* and *flowers dataset*. The object dataset was already split into training and testing parts, however, the other one was not. Hence, it was split into *80-20 train-test* portions before proceeding forward. Moreover, all images were resized to a single dimension *(250,250)*.

The code relies on:

* *open-cv python* for reading and plotting images
* *SIFT* from open-cv python for local feature extraction
* *k-means* from scikit learn for generating vocabulary via clustering
* *SVM* and *Random Forest* from scikit learn as classification models

A detailed view of working methodology can be seen in the figure below.

![Untitled Diagram drawio(1)](https://user-images.githubusercontent.com/127010479/224530592-d4780e89-c8ce-449e-92b9-29570c56e330.png)



## Instructions for setting up the environment
To execute the code, you should have necessary libraries installed in the environment you are using.
* open-cv python
* python
* scikit learn
* matplotlib
* seaborn
* numpy
To install these libraries, you can run the command in your IDE with the name of package at the end.
```
!pip install
```
## Instructions on running the train and test scripts on the train and test data
The dataset should have following structure, where all images belonging to one class are in same folder.
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
If the data is not splitted in training and testing portions, it should be using following script. You should chnage the path of source directory ,*source_dir*, and class labels in *folder_paths* as per your requirement.
```
source_dir = '/content/flower_photos'


labels = []
images = []
total_images = 0
copied_images = 0


for subfolder in os.listdir(source_dir):
    for image in os.listdir(os.path.join(source_dir, subfolder)):
      total_images += 1
      images.append(image)
      labels.append(subfolder)
      copied_images +=1
  else:
    continue



# shuffling the images to remove any patterns
combined = list(zip(images, labels))
np.random.seed(7)
np.random.shuffle(combined)


train_proportion = 0.8
split_index = int(len(combined)*train_proportion)
train_part = combined[:split_index]
test_part = combined[split_index:]

train_dir = os.path.join(source_dir, 'train')
test_dir = os.path.join(source_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_data = 0
test_data = 0
folder_paths = [os.path.join(source_dir, 'daisy'),
                os.path.join(source_dir, 'dandelion'),
                os.path.join(source_dir, 'roses'),
                os.path.join(source_dir, 'sunflowers'),
                os.path.join(source_dir, 'tulips')]

for image, label in train_part:
  target = os.path.join(source_dir, 'train', label)
  image_path = os.path.join(source_dir,label,image)
  if os.path.exists(target):
    shutil.copy(image_path, target)
  else:
    os.makedirs(target, exist_ok=True)
    shutil.copy(image_path, target)
  train_data += 1

for image, label in test_part:
  target = os.path.join(source_dir, 'test', label)
  image_path = os.path.join(source_dir,label,image)
  if os.path.exists(target):
    shutil.copy(image_path, target)
  else:
    os.makedirs(target, exist_ok=True)
    shutil.copy(image_path, target)
  test_data += 1

print('Total Images Found: {}'.format(total_images))
print('Copied Images: {}'.format(copied_images))
print()
print('Training Samples after 80-20 split: {}'.format(train_data)) 
print('Testing Samples after 80-20 split: {}'.format(test_data))  
```


## Quantitative Results

### SVM Classifier

#### Objects Dataset
```
Accuracy of the model:  1.0
F1 Score of the model:  1.0

False Positive Rate of  " Soccer_Ball " is:  0.0
True Positive Rate of  " Soccer_Ball " is:  1.0

False Positive Rate of  " accordian " is:  0.0
True Positive Rate of  " accordian " is:  1.0

False Positive Rate of  " dollar_bill " is:  0.0
True Positive Rate of  " dollar_bill " is:  1.0

False Positive Rate of  " motorbike " is:  0.0
True Positive Rate of  " motorbike " is:  1.0
```
#### Flowers Dataset
```
Accuracy of the model:  0.9768392370572208
F1 Score of the model:  0.9768765226843721

False Positive Rate of  " daisy " is:  0.0
True Positive Rate of  " daisy " is:  0.7183908045977011

False Positive Rate of  " dandelion " is:  0.02142857142857143
True Positive Rate of  " dandelion " is:  1.0

False Positive Rate of  " roses " is:  0.0
True Positive Rate of  " roses " is:  0.7241379310344828

False Positive Rate of  " sunflowers " is:  0.0035714285714285713
True Positive Rate of  " sunflowers " is:  0.7931034482758621

False Positive Rate of  " tulips " is:  0.005357142857142857
True Positive Rate of  " tulips " is:  0.8850574712643678
```

### Random Forest Classifier

#### Objects Dataset
```
Accuracy of the model:  1.0
F1 Score of the model:  1.0

False Positive Rate of  " Soccer_Ball " is:  0.0
True Positive Rate of  " Soccer_Ball " is:  1.0

False Positive Rate of  " accordian " is:  0.0
True Positive Rate of  " accordian " is:  1.0

False Positive Rate of  " dollar_bill " is:  0.0
True Positive Rate of  " dollar_bill " is:  1.0

False Positive Rate of  " motorbike " is:  0.0
True Positive Rate of  " motorbike " is:  1.0
```
#### Flowers Dataset
```
Accuracy of the model:  0.9768392370572208
F1 Score of the model:  0.9768761394936759

False Positive Rate of  " daisy " is:  0.0
True Positive Rate of  " daisy " is:  0.7225433526011561

False Positive Rate of  " dandelion " is:  0.023172905525846704
True Positive Rate of  " dandelion " is:  1.0

False Positive Rate of  " roses " is:  0.0
True Positive Rate of  " roses " is:  0.7283236994219653

False Positive Rate of  " sunflowers " is:  0.0035650623885918
True Positive Rate of  " sunflowers " is:  0.7976878612716763

False Positive Rate of  " tulips " is:  0.0035650623885918
True Positive Rate of  " tulips " is:  0.8959537572254336
```



## Visual Results


### SVM Classifier

#### Objects Dataset
![index](https://user-images.githubusercontent.com/127010479/224532250-362fa244-b950-4eea-9a2a-e631b2edb5e2.png)

#### Flowers Dataset
![index1](https://user-images.githubusercontent.com/127010479/224532285-fddb160e-7676-45d6-aeaf-34d4a4dd6a46.png)


### Random Forest Classifier

#### Objects Dataset
![index2](https://user-images.githubusercontent.com/127010479/224532331-251817b7-ce22-4deb-a218-6739c113f544.png)

#### Flowers Dataset
![index3](https://user-images.githubusercontent.com/127010479/224532355-1d54cb21-3d06-42e0-a441-3af136c7b786.png)
