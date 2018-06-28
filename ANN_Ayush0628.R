# Code for ANN based models on DJ Basin dataset
# install.packages("ggcorrplot")
# install.packages("devtools")
# install.packages("caret")
# install.packages("mlbench")
# install.packages("RRF")
# install.packages("NeuralNetTools")
# install.packages("plotly")
# install.packages("nnet")

cat("\014") # clear the console window
rm(list = ls()) # removes existing objects from current workspace

library(ggcorrplot)
library(caret)
library(mlbench)
library(purrr)
library(tidyr)
library(ggplot2)
library(nnet)
library(NeuralNetTools)

# Import data set
Dataset_raw = read.csv(file.choose(), header = TRUE) # Import DJ_Dataset.csv
dim(Dataset_raw) #dimensions - 339 x 8

# Dataset = Dataset_raw[, -c(3,6)] #omitting some non-significant
Dataset = Dataset_raw[,] # no columns removed
Dataset = na.omit(Dataset) # Out of 339 rows, only 159 observations left

dim(Dataset) #dimensions
names(Dataset) # feature names

# Rename the columns
# **Note: if any columns are removed from here, make sure the index number is correct
# Better accuracy if BTUGas and AvgPPG are removed as predictors
names(Dataset)[1] = 'Norm365' 
names(Dataset)[2] = 'Bblsft' 
names(Dataset)[3] = 'AvgPPG' 
names(Dataset)[4] = 'AdjWC365' 
names(Dataset)[5] = 'TVD' 
names(Dataset)[6] = 'BTUGas' 
names(Dataset)[7] = 'NeutPor'
names(Dataset)[8] = 'AvgRtNetPay' 

names(Dataset)

dim(Dataset) #dimensions
str(Dataset) #structure
head(Dataset) # top5 values

# Histogram of all predictor variables
# operator passes the df output that results from the function right before the pipe to input 
# it as the first argument of the function right after the pipe.

Dataset %>% 
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

# Combining the 'scale' and 'center' transforms will standardize your data. 
# Attributes will have a mean value of 0 and a standard deviation of 1.
# (https://www.rdocumentation.org/packages/caret/versions/6.0-80/topics/preProcess)
# preprocessParam <- preProcess(Dataset[,-1], method=c("center","scale")) # Dataset[,-1] if not scaling the response
# scaled.data <- predict(preprocessParam, Dataset)
# head(scaled.data)

range.scale = function(x)
{
  (x - min(x)) / (max(x) - min(x))
}

scaled.data	= Dataset
scaled.data[,-1] = apply(Dataset[,-1], MARGIN = 2, FUN = range.scale)
head(scaled.data)

set.seed(1234) # starting point in generation of sequence of random numbers
#expand.grid: create a dataframe from all combinations of supplied vectors
#nnet: size = number of units in hidden layer.Here we assume just 1 hidden layer (unlike deep neural network)
#nnet: decay = regularization parameter to avoid overfitting

nnet.grid = expand.grid(size = 1:15, decay = c(0, 0.01, 0.1, 1, 1.5, 2, 5, 10))
len.nn.grid = dim(nnet.grid)[1] # Calculated as 15(size)*8(decay) = 120
no.of.folds = 10

test.rmse.nn.model = rep(NA, len.nn.grid) # rep() replicates the value 'NA' 120 times here
test.aae.nn.model = rep(NA, len.nn.grid) # Repeated here for AAE

# sample function takes a sample if specified size from the elements of x in sample(x)
# Here any values from 1 to 10
index.values = sample(1:no.of.folds, size = dim(scaled.data)[1], replace = TRUE)

# In ANN, running the loop for 2 parameters - best size and best decay
system.time({

for (i in 1:len.nn.grid)
{

  tmp.mse = rep(NA, no.of.folds)
  tmp.aae = rep(NA, no.of.folds)
  
  for (j in 1:no.of.folds)
  {
    index.out     = which(index.values == j) # gives the true indices of an object                            
    left.out.data = scaled.data[  index.out, ]  #training data                      
    left.in.data  = scaled.data[ -index.out, ]  #testing data
                          
    tmp.nn.model  = nnet(Norm365 ~ ., data = left.in.data,  linout=T, size = nnet.grid$size[i], decay = nnet.grid$decay[i], maxit = 100)   
       
    tmp.pred.nn  	= predict(tmp.nn.model, newdata = left.out.data)  

    tmp.mse[j]   = mean((tmp.pred.nn - left.out.data$Norm365)^2)
    tmp.aae[j]   = abs(tmp.pred.nn - left.out.data$Norm365)
  }

  test.rmse.nn.model[i] = sqrt(mean(tmp.mse))
  test.aae.nn.model[i] = mean(tmp.aae)
}

})

test.rmse.nn	= min(sqrt(mean(tmp.mse)))
test.aae.nn	= min(test.aae.nn.model)
best.decay = nnet.grid[which.min(test.rmse.nn.model), ]$decay
best.size  = nnet.grid[which.min(test.rmse.nn.model), ]$size

best.decay
best.size

# ----------Neural Network Methods -------------

# Method 1 - neuralnet package, https://cran.r-project.org/web/packages/neuralnet/neuralnet.pdf

# library(neuralnet)
# final.model.neunet = neuralnet(Dataset$Norm365~LatLength+PropMassPerFt+HPV+GOR+AvgRtNetPay+AvgPPG, data = scaled.data, hidden = 1, act.fct = "tanh", linear.output = TRUE)
# names(final.model.neunet)
# plot(final.model.neunet, rep ="best")
# # https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/
# predicted = (final.model.neunet$net.result)*(max(scaled.data$Norm365)-min(scaled.data$Norm365))+min(scaled.data$Norm365) #rescaling the results
# rmse.neunet = sum((predicted - scaled.data$Norm365)^2)/nrow(scaled.data$Norm365)
# rmse.neunet
# final.model.neunet$call

# Method 2 - Caret package - Classification and Regression Training

# NNmodel2 <- train(Norm365 ~ LatLength+PropMassPerFt+HPV+GOR+AvgRtNetPay+AvgPPG, Dataset, method='nnet', linout=TRUE, trace = TRUE)
# names(NNmodel2)
# NNmodel2$results
# NNmodel2$metric

# Method 3 - nnet package, https://cran.r-project.org/web/packages/nnet/nnet.pdf

final.model.nn =  nnet(Norm365 ~ ., data = scaled.data,  linout=T, size = best.size, decay = best.decay, maxit = 1000, trace = TRUE)   
names(final.model.nn)

r.squared.nn = (cor(final.model.nn$fitted.values, scaled.data$Norm365))^2
r.squared.nn

train.rmse.nn = sqrt(mean((final.model.nn$fitted.values - scaled.data$Norm365)^2))
train.rmse.nn

train.aae.nn = mean(abs(final.model.nn$fitted.values - scaled.data$Norm365))
train.aae.nn

test.rmse.nn
test.aae.nn


# --------------Plot-----------------------------------------


# windows();plotnet(final.model.nn)
# summary(final.model.nn)
# names(final.model.nn)
# final.model.nn # multiple parameters calculated for the Neural Net. Options can be seen from 'names(final.model.nn)'

names(final.model.nn)

X1 = final.model.nn$fitted.values
Y =  scaled.data$Norm365

windows();plot(X1 ~ Y, pch = 1, xlab = "Observed Norm 365 Oil (bbl/ft)", ylab = "ANN Predicted Norm 365 Oil (bbl/ft)", xaxs ="i", yaxs ="i",
     main = "ANN Model", xlim = c(0,40), ylim = c(0,40))
legend("topleft", bty = "n", legend=paste("R2 = ", format(r.squared.nn,digits=4)))
#h = lm(X1 ~ Y) # Linear Model
#abline(h, col="blue", lwd = 2)
abline(a=0, b = 1, lty = 2, col="red") # Here a = intercept, b = slope


# -------------Prediction and Comparison with Linear Model----

# MVA Model - Linear Regression
lin_model = lm(Norm365~Bblsft+AvgPPG+AdjWC365+TVD+BTUGas+NeutPor+AvgRtNetPay, data=Dataset)
summary(lin_model) # model with R2 = 0.70
names(lin_model)

colMeans(Dataset[sapply(Dataset, is.numeric)])
summary(Dataset) # Basic statistics of the input variables
#Histogram of Parameter of Interest -> Bbls/ft
windows();qplot(Dataset$Bblsft,
      geom="histogram",
      binwidth = 0.5,  
      main = "Histogram: Bbls/ft", 
      xlab = "Bbls/ft",  
      fill=I("blue"), 
      col=I("red"), 
      alpha=I(.2))
# Based on Histogram: Range of values for bbls/ft are 0-50

# MVA Prediction
# the input prediction file has each variable scaled: value-min/(max-min)
# bblsft.data = read.csv(file.choose(), header = TRUE) # Import Input_PredictionDJ_EL.csv
# bblsft.data
# names(bblsft.data)[1] = 'Bblsft' # Renaming the column
 

# scaled.data2 <- predict(preprocessParam, bblsft.data)
# head(scaled.data2)




# # Make sure column names match before using the predict method
# lin_predict=predict(lin_model, newdata = bblsft.data)
# bblsft.data$lin_predict <- lin_predict

# ann_predict = predict(final.model.nn, newdata = scaled.data2)
# bblsft.data$ann_predict <- ann_predict
# bblsft.data

# ann_predict = write.csv(ann.values, file="Output_PredictionDJ_ANN.csv", row.names = FALSE)

source("OPAAT_sensitivity.R")
# the datafile here should have Norm365(Target variable) as the first column
df <- OPAAT_sensitivity(Dataset, lin_model) # sensitivity for Linear Model
df2 <- OPAAT_sensitivity(scaled.data, final.model.nn) # sensitivity for Linear Model

# Re-scale the odd columns before plotting in the next step

# Prediction Comparison
# Need a  loop to plot each variable along with its subsequent prediction value
windows();
ggplot()+
  geom_line(data = df, aes(x = df$AdjWC365, y = df$`AdjWC365  Response`), color = "blue", size = 1)+
  geom_line(data = df2, aes(x = df$AdjWC365, y = df2$`AdjWC365  Response`), color = "red", size = 1)+
  xlab('AdjWC365')+
  ylab('Prediction')

# par(mfrow = c(2,4))
# 
# for (i in 1:nrow(df)){
#   x = df[,i]
#   y = df[,i+1]
#   plot(x,y)
# }

par(mfrow = c(2,4))
for (i in 1:length(df2)){
  x = df2[,i]
  y = df2[,i+1]
  windows();
  ggplot(data=df2, aes(x=x, y=y, group=1)) +
  geom_line(color="red")+
  geom_point()
  }
#_---------------------------
