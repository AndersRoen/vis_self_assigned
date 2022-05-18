#importing stuff
import pathlib
# operating system
import os 
import argparse

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
import keras
#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt


#loading in our stuff

#premade plotting function made in class
def plot_history(H, epochs, vis_name): 
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, int(epochs)), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, int(epochs)), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, int(epochs)), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, int(epochs)), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()
    outpath_vis = os.path.join("out", vis_name)
    fig = plt.savefig(outpath_vis, dpi = 300, bbox_inches = "tight")
    return fig

def load_data(b_size):
    # get directory paths - we need to do this a few times
    test_path = pathlib.Path("in", "test")
    train_path = pathlib.Path("in", "train")
    val_path = pathlib.Path("in", "valid")
    
    # now we need to define the parameters
    num_classes = 100 # the amount of different sports
    img_height = 224 # all images in the dataset are 224x224
    img_width = 224
    
    # create 3 datasets from the previously defined paths
    # tensorflows image_dataset_from_directory can make a 
    # dataset from the image files in a given directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size = (img_height, img_width),
        batch_size = int(b_size))
    
    # we do this for each path
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size = (img_height, img_width),
        batch_size = int(b_size))
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        image_size = (img_height, img_width),
        batch_size = int(b_size))
    
    # get class names
    class_names = train_ds.class_names
    
    return train_ds, test_ds, val_ds, class_names, num_classes

def mdl(num_classes, learn_rate):
    model = VGG16(include_top = False, # using the pretrained VGG16 model to perform transfer learning
                  pooling = "avg",
                  input_shape = (224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(256, activation = "relu")(flat1)
    output = Dense(num_classes, activation = "softmax")(class1)
    
    model = Model(inputs = model.inputs,
                  outputs = output)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate= float(learn_rate)), # we use adam instead of the sgd we've used in class, as adam performs better on larger datasets
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics = ["accuracy"])
    return model

def train_eval_model(model, train_ds, epochs, val_ds, test_ds, class_names, rep_name):
    # training the model
    H = model.fit(train_ds, epochs=int(epochs), validation_data=val_ds)
    # printing the model's performance on the test_dataset
    print("PERFOMANCE", model.evaluate(test_ds))
    # creating a classification report - I have to do this a bit differently, as i dont have the X_test, X_train, y_train, y_test split
    y_pred = []  # store predicted labels
    y_true = []  # store true labels
    # iterate over the dataset
    for image_batch, label_batch in test_ds:   
        y_true.append(label_batch) # extracting the true labels from the test_ds dataset
        # compute predictions
        preds = model.predict(image_batch) # predicting the images in the test_ds dataset
        # append predicted labels
        y_pred.append(np.argmax(preds, axis = - 1))
    
    # convert the true and predicted labels into tensors
    correct_labels = tf.concat([item for item in y_true], axis = 0)
    predicted_labels = tf.concat([item for item in y_pred], axis = 0)
    # now we make a classification report as usual
    report = classification_report(correct_labels, predicted_labels, target_names = class_names)
    outpath_rep = os.path.join("out", rep_name)
    #write the report to a csv 
    with open(outpath_rep, "w") as file:
        file.write(str(report))

    return H, report

def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    # add command line parameters
    ap.add_argument("-vn", "--vis_name", required=True, help="the name of the figure")
    ap.add_argument("-rp", "--rep_name", required=True, help="the name of the classification report")
    ap.add_argument("-ep", "--epochs", required=True, help="the amount of epochs you want the model to train")
    ap.add_argument("-bs", "--b_size", required=True, help="the batch size of the model")
    ap.add_argument("-lr", "--learn_rate", required=True, help="the learning rate of the model") # tensorflow recommends 0.001
    args = vars(ap.parse_args())
    return args

def main():
    args = parse_args()
    train_ds, test_ds, val_ds, class_names, num_classes = load_data(args["b_size"])
    model = mdl(num_classes, args["learn_rate"])
    H, report = train_eval_model(model, train_ds, args["epochs"], val_ds, test_ds, class_names, args["rep_name"])
    fig = plot_history(H, args["epochs"], args["vis_name"])
                                    
if __name__ == "__main__":
    main()