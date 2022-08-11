

library(neuralnet)
library(tidyverse)

# read in the dataset
data <- read.csv(".\\data\\FIC.Full CSV.csv")

# clean up the data, only going to use a handful of variables to save computation on the neural network model.
data <- data %>% select(-Age.Group, -Others, -CO, -Diagnosis)

# Turn string columns into factors then to numeric.
data <- as.data.frame(unclass(data),
                      stringsAsFactors = TRUE)
indx <- sapply(data, is.factor)
data[,indx] <- lapply(data[indx], function(x) as.numeric(x))

# Normalizing the data between 0 and 1
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, 
                              scale = maxs - mins))

# Split the data into 70% training and 30% testing
index <- sample(1:nrow(data), round(0.7 * nrow(data)))
train.data <- scaled[index,]
test.data <- scaled[-index,]

# Neural Network
nn <- neuralnet(Mortality ~., data = train.data, hidden = c(5, 2), linear.output = F)

# Plot the neural network
plot(nn)