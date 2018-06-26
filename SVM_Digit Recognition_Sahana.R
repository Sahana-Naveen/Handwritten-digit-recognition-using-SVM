#######################################################################################################################
####################################### Handwritten digit recognition using SVM model##################################
#######################################################################################################################

#!!!! SAHANA K !!!!!!

#---------------------------------------------------------------------------------------------------------------------
#-------------------------------------- 1. Business Understanding ----------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

#A classic problem in the field of pattern recognition is that of handwritten digit recognition. 
#Suppose that you have an image of a digit submitted by a user via a scanner, a tablet, or other digital devices. 
#The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 

#--- Objective ---

#Develop a model using Support Vector Machine which should correctly classify the handwritten digits 
#based on the pixel values given as features.


####################################################################################################################

# a) Remove the previous data (if any)

rm(list=ls())

# b) Install necessary packages

install.packages("caret")
install.packages("kernlab")
install.packages("dplyr")
install.packages("readr")
install.packages("ggplot2")
install.packages("caTools")
install.packages("gridExtra")

# c) Load the Packages

library(kernlab)
library(readr)
library(caret)
library(dplyr)
library(ggplot2) 
library(caTools)
library(gridExtra)

# d) Set working directory

setwd("C:/Data_Science/## SVM assesment")

#----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- 2. Data Understanding --------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

#The data files mnist_train.csv and mnist_test.csv contain gray-scale images of hand-drawn digits from 0 to 9.

#Loading the given files 

mnist_train <- read.csv("mnist_train.csv",stringsAsFactors = FALSE,header = FALSE)
mnist_test <- read.csv("mnist_test.csv",stringsAsFactors = FALSE,header = FALSE)

#Understanding Dimensions

dim(mnist_train)    #60000   785
dim(mnist_test)     #10000   785

#Structure of the dataset

str(mnist_train)    #60000 obs. of  785 variables
str(mnist_test)     #10000 obs. of  785 variables

#printing first few rows of train and test dataset

head(train)
head(test)

#Exploring the data

summary(mnist_train)
summary(mnist_test)

# check for duplicated values

sum(duplicated(mnist_train)) #0
sum(duplicated(mnist_test))  #0

#checking missing value

sapply(mnist_train, function(X) sum(is.na(X)))    #No NA's found
sapply(mnist_test, function(X) sum(is.na(X)))     #No NA's found

#Checking for blanks

sapply(mnist_train, function(x) length(which(x == ""))) #There are no blanks
sapply(mnist_test, function(x) length(which(x == "")))  #There are no blanks

#Checking for outliers- Pixel range beyond 255

sapply(mnist_train, function(x) length(which(x <0 & x>255)))
sapply(mnist_test, function(x) length(which(x <0 & x>255)))

#----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- 3. Data Preparation/Cleansing and EDA ----------------------------------------
#----------------------------------------------------------------------------------------------------------------------

#By observation, first column contains all digits from 0-9, so taking this as the target variable.
#Rename dependent variable to digit

names(mnist_train)[1] <- "digit"
names(mnist_test)[1] <- "digit"

#Making target class as factor 

mnist_train$digit <- factor(mnist_train$digit)
mnist_test$digit <- factor(mnist_test$digit)

#stratified sampling to consider 5,000 training observations for modelling

set.seed(20)
trainindices <- sample.split(mnist_train$digit, SplitRatio = 0.08)
sampleTrain <- mnist_train[trainindices,]

#To check if class balance is maintained

Plot1 <- ggplot(mnist_train, aes(digit)) +
  geom_bar(width=.6, fill="tomato2") +
  labs(title = "Frequency of Digits - Train dataset", x = "Digits", y = "Frequency") +
  theme(axis.text = element_text(face="bold"))

Plot2 <- ggplot(sampleTrain, aes(digit)) +
  geom_bar(width=.6, fill="tomato2") +
  labs(title = "Frequency of Digits - Train sample", x = "Digits", y = "Frequency") +
  theme(axis.text = element_text(face="bold"))

Plot3 <- ggplot(mnist_test, aes(digit)) +
  geom_bar(width=.6, fill="tomato2") +
  labs(title = "Frequency of Digits -  Test dataset", x = "Digits", y = "Frequency") +
  theme(axis.text = element_text(face="bold"))

grid.arrange(Plot1, Plot2, Plot3, nrow = 3)

# According to the graphs, balance is maintained in the sample.
# Relative frequencies of the digits has been retained while sampling to create the reduced train data set
# Similar frequency in test dataset also observed


##Scaling data
# we should scale all the variables except the target variable
# Since all other values are in pixels and maximum pixel value is 255, dividing by 255 scales all variables.

sampleTrain[,-c(1)]<-as.data.frame(sampleTrain[,-c(1)]/255)
mnist_test[,-c(1)]<-as.data.frame(mnist_test[,-c(1)]/255)

#----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- 4. Model Building and Validation ---------------------------------------------
# -------------------------------------- 5. HYPER-PARAMETER AND CROSS VALIDATION --------------------------------------
#----------------------------------------------------------------------------------------------------------------------

#________________________________________________________________________________________________
#**************************************  a)  LINEAR KERNEL **************************************
#________________________________________________________________________________________________

#### a.1 Model Building

linear_model <- ksvm(digit~ ., data = sampleTrain, scale = FALSE, kernel = "vanilladot")
linear_eval <- predict(linear_model, mnist_test)
#__________________________________________________________________________________________

#### a.2 confusion matrix
linear_conf <- confusionMatrix(linear_eval,mnist_test$digit)
acc_linear <- linear_conf$overall[1]  #0.9113
sens_linear <- linear_conf$byClass[1] #0.9775
spec_linear <- linear_conf$byClass[2] #0.9867
acc_linear
sens_linear
spec_linear
#__________________________________________________________________________________________

#### a.3 Hyperparameter tuning and cross validation

trainControl <- trainControl(method="cv", number=5)
metric <- "Accuracy"

set.seed(80)
linear_model

# Making a grid of C value 
grid <- expand.grid(C=seq(1, 5, by=1))

# Performing 5 fold cross validation
fit_svm <- train(digit~., data=sampleTrain, method="svmLinear", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit_svm)

# Best tune at C=1, 
# Accuracy = 0.9055927
# Kappa = 0.8950594

plot(fit_svm)

# Plot clearly shows that C=1 is the best.
# Accuracy is good at cost C = 1.
#__________________________________________________________________________________________

#### a.4 Valdiating the linear model after cross validation on test data

evaluate_linear <- predict(fit_svm, mnist_test)

confusionMatrix(evaluate_linear, mnist_test$digit)
# Accuracy after cross validation on test data    = 0.9113
# Kappa : 0.9014 
# Best tunning at C=1

#________________________________________________________________________________________________
#**************************************  b)  RBF KERNEL *****************************************
#________________________________________________________________________________________________

#### b.1 Model Building

RBF_model <- ksvm(digit~ ., data = sampleTrain, scale = FALSE, kernel = "rbfdot")
RBF_eval <- predict(RBF_model, mnist_test)
#__________________________________________________________________________________________

#### b.2 confusion matrix

RBF_conf <- confusionMatrix(RBF_eval,mnist_test$digit)
acc_RBF <- RBF_conf$overall[1]  #0.9519
sens_RBF <- RBF_conf$byClass[1] #0.9846
spec_RBF <- RBF_conf$byClass[2] #0.9894
acc_RBF
sens_RBF
spec_RBF

# The sensitivy, specificity and Accuracy values are much higher in RBF analysis compared to linear model.
#__________________________________________________________________________________________

#### b.3 Hyperparameter tuning and cross validation

trainControl <- trainControl(method="cv", number=5)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.
metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
set.seed(7)
RBF_model

# cost C = 1
# Hyperparameter : sigma =  0.0105061106480373  
# Training error :  0.021671
# Number of Support Vectors : 2285

grid <- expand.grid(.sigma=c(0.01,0.025,0.05), .C=c(0.5,1,2,3))

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = traincontrol method.
fit.svm_rbf <- train(digit~., data=sampleTrain, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm_rbf)
plot(fit.svm_rbf)

##The final values used for the model were sigma = 0.025 and C = 3.  Accuracy = 0.9591
# Lower Sigma value is perfoming good on CV.
#__________________________________________________________________________________________

#### a.4 Valdiating the model after cross validation on test data

evaluate_rbf <- predict(fit.svm_rbf, mnist_test)

confusionMatrix(evaluate_rbf, mnist_test$digit)
# Accuracy : 0.9637 
# Kappa : 0.9596

# RBF is performing fairly good with the above Accuracy and Kappa
# Optimal value for tunning is at sigma = 0.025 & C=3

#*******************************************************************
#******************* OBSERVATION and CONCLUSION  *******************

#                       ACCURACY    KAPPA     COST(C)    SIGMA 

# LINEAR MODEL      :   0.9113      0.9014     1          -

# RBF MODEL         :   0.9519      0.9465     3         0.025


# After Comparing above parameters, RBF Model is performing better than Linear model in
# Accuracy, kappa, sensitivity and specificity. 
#  - RBF model performed better in accurately predicting the digit of digital image.
#  - Non linearity parameter i.e. Sigma is very low i.e. 0.025

##################################################### End ########################################################