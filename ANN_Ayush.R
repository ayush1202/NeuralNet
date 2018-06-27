# Code for ANN based models on DJ Basin dataset
# install.packages("ggcorrplot")
# install.packages("devtools")
# install.packages("caret")
# install.packages("mlbench")
# install.packages("RRF")
# install.packages("NeuralNetTools")

cat("\014") # clear the console window
rm(list = ls()) # removes existing objects from current workspace

# Import data set
Dataset_raw = read.csv(file.choose(), header = TRUE)
dim(Dataset_raw) #dimensions - 339 x 8

# Dataset = Dataset_raw[, -c(3,6)] #omitting some non-significant
Dataset = Dataset_raw[,] # no columns removed
Dataset = na.omit(Dataset) # Out of 339 rows, only 159 observations left

dim(Dataset) #dimensions
names(Dataset) # feature names

# Rename the columns

# **Note: if any columns are removed from here, make sure the index number is correct
names(Dataset)[1] = 'Norm365' 
names(Dataset)[2] = 'Bblsft' 
names(Dataset)[3] = 'AvgPPG' 
names(Dataset)[4] = 'AdjWC365' 
names(Dataset)[5] = 'TVD' 
names(Dataset)[6] = 'BTUGas' 
names(Dataset)[7] = 'NeutPor'
names(Dataset)[8] = 'AvgRtNetPay' 

names(Dataset)

library(ggcorrplot)
library(caret)
library(mlbench)

dim(Dataset) #dimensions
str(Dataset) #structure
head(Dataset) # top5 values
names(Dataset) # feature names

# Range Scaling - using caret library 
# (https://www.rdocumentation.org/packages/caret/versions/6.0-80/topics/preProcess)
preprocessParam <- preProcess(Dataset[,-1], method=c("range")) # Dataset[,-1] if not scaling the response
scaled.data <- predict(preprocessParam, Dataset)
head(scaled.data)

# install.packages("nnet")
library(nnet)

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


# --------------Plot----------------------------

library(NeuralNetTools)
# windows();plotnet(final.model.nn)
# summary(final.model.nn)
# names(final.model.nn)
# final.model.nn # multiple parameters calculated for the Nueral Net. Options can be seen from 'names(final.model.nn)'

X1 = final.model.nn$fitted.values
Y =  scaled.data$Norm365

windows();plot(X1 ~ Y, pch = 1, xlab = "Observed Norm 365 Oil (bbl/ft)", ylab = "ANN Predicted Norm 365 Oil (bbl/ft)", xaxs ="i", yaxs ="i",
     main = "ANN Model")
legend("topleft", bty = "n", legend=paste("R2 = ", format(r.squared.nn,digits=4)))
h = lm(X1 ~ Y) # Linear Model
abline(h, col="blue", lwd = 2)
abline(a=0, b = 1, lty = 2, col="red") # Here a = intercept, b = slope


# -------------Prediction and Comparison with Linear Model----

# MVA Model - Linear Regression
lin_model = lm(Norm365~Bblsft+AvgPPG+AdjWC365+TVD+BTUGas+NeutPor+AvgRtNetPay, data=Dataset)
summary(lin_model) # model with R2 = 0.70
names(lin_model)

colMeans(Dataset[sapply(Dataset, is.numeric)])
summary(Dataset) # Basic statistics of the input variables
#Histogram of Parameter of Interest -> Bbls/ft
qplot(Dataset$Bblsft,
      geom="histogram",
      binwidth = 0.5,  
      main = "Histogram: Bbls/ft", 
      xlab = "Bbls/ft",  
      fill=I("blue"), 
      col=I("red"), 
      alpha=I(.2))
# Based on Histogram: Range of values for bbls/ft are 0-50

# MVA Prediction
bblsft.data = read.csv(file.choose(), header = TRUE)
bblsft.data

names(bblsft.data)[1] = 'Bblsft' # Renaming the first column
# Make sure column names match before using the predict method
bblsft.values=predict(lin_model, newdata = bblsft.data)
bblsft.values
lin_predict = write.csv(bblsft.values, file="Output_PredictionDJ_Lin.csv", row.names = FALSE)

# NeuralNetwork Prediction Model  - http://stat.ethz.ch/R-manual/R-patched/library/nnet/html/predict.nnet.html
bblsft.data = read.csv(file.choose(), header = TRUE)
names(bblsft.data)[1] = 'Bblsft'
ann.values = predict(final.model.nn, newdata=bblsft.data,  linout=T) # Getting same Values!!
ann_predict = write.csv(ann.values, file="Output_PredictionDJ_ANN.csv", row.names = FALSE)


# Change to the right axis and change ggplot style
compare = read.csv(file.choose(), header = TRUE)
names(compare)
ggplot(data=compare, aes(x=compare$�..LinModel, y=compare$ANN)) +
  geom_line(linetype = "dashed", color = "red") +  
  geom_point(shape=18, color="red")

