# info523-final-project-NN
 Using Neural Networks to predict mortality from heart disease
 
 ### Built With
 
 R and RStudio
 
 ### Prerequisites
 
 You will need the neuralnet, caret, tidyverse, and ggplot2 packages to run this code.
 
  ```sh
  packages.install("neuralnet")
  packages.install("caret")
  packages.install("tidyverse")
  packages.install("ggplot2")
  ```
  
  ## Usage
  
  This repository contains two R scripts and a [dataset from Kaggle](https://www.kaggle.com/datasets/asgharalikhan/mortality-rate-heart-patient-pakistan-hospital) containing patient data for people with heart disease.
  
  neural_network_predictions.R fits cleans the dataset and fits it to a Neural Network. The Neural Network parameters may be updated by editing the following line in the code:
  ```sh
  nn <- neuralnet(Mortality ~., data = train.data, hidden = c(3, 2), linear.output = F)
  ```
Currently it is using 2 layers where the first layer has 3 neurons and the second layer has 2 neurons. This may be adjusted as needed to find the best fit for the dataset.

 nn_neurons_layers.R may take a few minutes to run. This script performs an analysis on how the different number of neurons in either one or two layers affects the accuracy of the model. It goes through each possible single layer configuration and retrains the data 10 times for each configuration to obtain the average accuracy for each configuration. In the two layer case there are too many combinations to go through each configuration, so instead 100 network configurations are sampled and retrained 10 times to determine their accuracy. These are then plotted using ggplot2 to visualize how the varying number of neurons in each layer affects the model.
