# This script is intending to find an optimal number of neurons in a one or two layer neural net

library(neuralnet)
library(caret)
library(tidyverse)

# read in the dataset
data <- read.csv(".\\data\\FIC.Full CSV.csv")

# Remove variables that are difficult to quantify or overlap with other variables
data <- data %>% select(-Age.Group, -Others, -CO, -Diagnosis)

# Turn string columns into factors then to numeric.
data <- as.data.frame(unclass(data),
                      stringsAsFactors = TRUE)
indx <- sapply(data, is.factor)
data[,indx] <- lapply(data[indx], function(x) as.numeric(x))

# Normalizing the data between 0 and 1
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
data_scaled <- as.data.frame(scale(data, center = mins, 
                                   scale = maxs - mins))
# Loop through ten times per network configuration and average the accuracy of the model configuration
# Total number of neurons should not exceed the number of input nodes

# 1 layer results
oneLayerAvgs <- c()
neurons <- c(1:length(train.data))
for (n in neurons) {
  #initialize the average accuracy to 0
  set.seed(5) # Random seed set for repeatability
  avgAcc <- 0
  for (i in c(1:10)) {
    # Randomly split the data into 70% training and 30% testing
    index <- sample(1:nrow(data_scaled), round(0.7 * nrow(data_scaled)))
    train.data <- data_scaled[index,]
    test.data <- data_scaled[-index,]
    
    # Neural Network
    nn <- neuralnet(Mortality ~., data = train.data, hidden = c(n), linear.output = F)
    
    # predict the mortality on the test data
    y.prediction <- neuralnet::compute(nn, test.data)
    
    # Converts numeric results to "lived" or "died"
    lived.prediction <- round(y.prediction$net.result)
    lived.prediction[which(round(y.prediction$net.result) == 1)] <- "Lived"
    lived.prediction[which(round(y.prediction$net.result) == 0)] <- "Died"
    
    lived.actual <- round(test.data$Mortality)
    lived.actual[which(test.data$Mortality == 1)] <- "Lived"
    lived.actual[which(test.data$Mortality == 0)] <- "Died"
    
    comparison <- data.frame(list(predicted=lived.prediction,actual=lived.actual))
    
    # Calculates statistics for the model performance
    cm = confusionMatrix(as.factor(comparison$predicted), as.factor(comparison$actual))
    avgAcc <- avgAcc + cm$overall[[1]]
  }
  avgAcc <- avgAcc / 10
  oneLayerAvgs <- append(oneLayerAvgs, avgAcc)
}

oneLayerDF <- data.frame(list(Neurons=neurons, Average_Accuracy=oneLayerAvgs))


# 2 layer results - 1540 combinations where the neurons < the number of variables.
# to save computing time, need to randomly sample 100 different configurations
layer1 <- c()
layer2 <- c()
for (i in neurons) {
  for(j in neurons) {
    if ((i + j) <= length(neurons)) {
      layer1 <- append(layer1,i)
      layer2 <- append(layer2,j)
    }
  }
}

set.seed(6)
index <- sample(c(1:length(layer1)), size=100, replace = FALSE)
layer1_rand <- layer1[index]
layer2_rand <- layer2[index]
twoLayerAvgs <- c()

for (n in c(1:length(layer1_rand))) {
  #initialize the average accuracy to 0
  set.seed(5) # Random seed set for repeatability
  avgAcc <- 0
  for (i in c(1:10)) {
    # Randomly split the data into 70% training and 30% testing
    index <- sample(1:nrow(data_scaled), round(0.7 * nrow(data_scaled)))
    train.data <- data_scaled[index,]
    test.data <- data_scaled[-index,]
    
    # Neural Network
    nn <- neuralnet(Mortality ~., data = train.data, hidden = c(layer1_rand[n],layer2_rand[n]), linear.output = F, 
                    stepmax = 50000)
    
    # predict the mortality on the test data
    y.prediction <- neuralnet::compute(nn, test.data)
    
    # Converts numeric results to "lived" or "died"
    lived.prediction <- round(y.prediction$net.result)
    lived.prediction[which(round(y.prediction$net.result) == 1)] <- "Lived"
    lived.prediction[which(round(y.prediction$net.result) == 0)] <- "Died"
    
    lived.actual <- round(test.data$Mortality)
    lived.actual[which(test.data$Mortality == 1)] <- "Lived"
    lived.actual[which(test.data$Mortality == 0)] <- "Died"
    
    comparison <- data.frame(list(predicted=lived.prediction,actual=lived.actual))
    
    # Calculates statistics for the model performance
    cm = confusionMatrix(as.factor(comparison$predicted), as.factor(comparison$actual))
    avgAcc <- avgAcc + cm$overall[[1]]
  }
  avgAcc <- avgAcc / 10
  twoLayerAvgs <- append(twoLayerAvgs, avgAcc)
}

twoLayerDF <- data.frame(list(first_layer=layer1_rand, second_layer=layer2_rand, Average_Accuracy=twoLayerAvgs))

library(ggplot2)

plt1 <- ggplot(oneLayerDF, aes(x = Neurons, y = Average_Accuracy*100,color=Average_Accuracy*100)) + geom_point() +
  xlab("# of Neurons in a Single Layer Neural Network") + ylab("Average Prediction Accuracy (%)") +
  labs(title="Finding the Optimal Network Configuration for a Single Layer Neural Network",
       subtitle = "Models are retrained 10 times with randomly split training (70%) and test (30%) data.\n
       The optimal number of neurons for a single layer network appears to be between 20 and 30.") +
  scale_color_continuous(name="Accuracy (%)")

plt2 <- ggplot(twoLayerDF, aes(x = first_layer, y = second_layer,fill=Average_Accuracy*100,size=Average_Accuracy*100)) + 
  geom_point(shape=21) + scale_size(name = "Prediction Accuracy (%)", range = c(1,9)) +
  xlab("First Layer Neurons") + ylab("Second Layer Neurons") +
  labs(title="Finding the Optimal Network Configuration for a Two Layer Neural Network",
       subtitle = "Randomly selected 100 different Two Layer Configurations and retrained them 10 times
       with randomly split training (70%) and test (30%) data.\n
       For a two layer model, in general, best performance seems to be achieved with the first layer containing 
       more neurons than the second layer.") + scale_fill_continuous(name="Prediction Accuracy (%)")

show(plt1)
show(plt2)

# saving the plots as pngs in the "figures" folder 
ggsave("one_layer_neurons.png", plt1, path = ".//figures")
ggsave("two_layer_neurons.png", plt2, path = ".//figures")
