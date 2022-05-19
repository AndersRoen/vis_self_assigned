# vis_self_assigned

The link to the self assigned project's repository: https://github.com/AndersRoen/vis_self_assigned.git

The link to the necessary data: https://www.kaggle.com/datasets/gpiosenka/sports-classification
Download the data, and put it in the ```in``` folder.

## Description of self assigned project
During this class, when working with image classification tasks, we have worked with datasets with a limited number of classes. ```cifar10``` and ```mnist_784``` both have 10 classes. This project looks at how one can use the concepts and tools we have learned so far, on a dataset with a lot more classes. The ```sports classification``` dataset has 100 classes, depicting different sports. The dataset is of quite high quality being preprocessed in a quite useful way: the shape of the images have already been resized to ```(224, 224, 3)``` and the sports being played are centralized in the image, to help the model learn better.

## Methods
As mentioned earlier, this project relates to image classification. The kaggle-user, who posted the dataset, states that one can get quite high accuracy - even so high as 98% if you build a very good model.
My goal was to achieve as high accuracy as possible.
The ```sports_classification.py``` script begins with defining the plotting function we've made in class. Then it loads in the data from the ```in``` folder. The data was already split up into ```train```, ```test``` and ```valid```, which meant that I had to tackle the data differently than we've seen in class, as I couldn't do the ```train_test_split``` - which will be important later.
While the data was already split up, I still needed to convert the folders to datasets. This was achieved by using the ```tf.keras.utils.image_dataset_from_directory()``` where you define the image height and width (224 x 224). You also define the batch size, which the user can do from the command line in this script.
The script then defines the model, using a pretrained model (```VGG16```). I found that using the ```adam``` optimizing algorithm performed quite a lot better than your "standard" ```sgd``` optimizer.
The model then trains, plots the loss and accuracy across epochs and saves it to the ```out``` folder, with a user-defined filename. The script then generates a classification report and save it to the ```out``` folder with a user-defined filename.

## Usage
To run this script you should point the command line to the ```vis_self_assigned``` folder as well as run the ```setup.sh```.
The script has 5 command line arguments:
```--vis_name``` which defines the filename of the loss + accuracy plot, such as ```loss_accuracy_plot.png```
```rep_name``` which defines the filename of the classification report, such as ```classification_rep.csv```
```epochs``` which defines the amount of epochs you want the model to train for, I set this to 10
```b_size``` which defines the batch size, I set this to 64.
```learn_rate``` which defines the learning rate of the model. I set this to 0.001, which seems to be the recommended learning rate for the ```adam``` optimizer.
Include all these command line arguments when you run the script.

## Discussion of results
I've included a few of my output files in the ```out``` folder. As seen from ```sports_rep_ep20.csv```, the highest accuracy I achieved was 91% with a learning rate of 0.001, batch size of 64 and 20 epochs. However, 90% can be achieved with the same hyperparameters, except that you only need 10 epochs to do it. Further, 10 epochs yields a smoother loss + accuracy plot.
I tried fiddling with the batch size, setting it to 32. This did not work well, yielding "only" 86% accuracy. 
When looking at the classification reports, it seems the model is either 100% accurate on some sports and then significantly worse on others.
Generally, this script works pretty much as intended, as my goal was to get over 90% accuracy. It doesn't seem to be worthwile to train for 20 epochs when you can get almost identical results in half the time.
