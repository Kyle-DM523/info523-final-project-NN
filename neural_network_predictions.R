

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

set.seed(5) # Random seed set for repeatability

# Randomly split the data into 70% training and 30% testing
index <- sample(1:nrow(data_scaled), round(0.7 * nrow(data_scaled)))
train.data <- data_scaled[index,]
test.data <- data_scaled[-index,]

# Neural Network
nn <- neuralnet(Mortality ~., data = train.data, hidden = c(3, 2), linear.output = F)

# Plot the neural network
plot(nn)

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
print(cm)
