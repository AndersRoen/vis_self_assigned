#!/usr/bin/env bash

pip install --upgrade pip
pip install opencv-python scikit-learn tensorflow tensorboard tensorflow-hub pydot scikeras[tensorflow-cpu]
sudo apt-get update
sudo apt-get -y install graphviz