# Code for ANN based models on Midland - Wolfcamp B dataset
# install.packages("ggcorrplot")
# install.packages("devtools")
# install.packages("caret")
# install.packages("mlbench")
# install.packages("RRF")
# install.packages("NeuralNetTools")
# install.packages("plotly")
# install.packages("nnet")
# install.packages("ggplot2")
# install.packages("tidyverse")
# install.packages("DataExplorer") # https://cran.r-project.org/web/packages/DataExplorer/DataExplorer.pdf
# install.packages("Amelia")

# install.packages("acepack")
# install.packages("splines")
# install.packages("Hmisc")
# install.packages("gam")

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
library(tidyverse)
library(DataExplorer)
library(FactoMineR)
library(pastecs)
library(Amelia)
library(Boruta)
library(PerformanceAnalytics)
library(randomForest)
library(dplyr)
library(factoextra)

# Import Midland_Basin_WolfcampB2.csv
Dataset_raw = read.csv(file.choose(), header = TRUE) 

colnames(Dataset_raw) 
Dataset <- Dataset_raw 

dim(Dataset) #dimensions 2293 x 16

# removing norm365boeft, latlength, ftstage, transftstage, translbft, bbls.ft, maxrate
Dataset = Dataset_raw[, -c(1,5,6,7,10,11,12)]

Dataset = na.omit(Dataset) 
dim(Dataset) #dimensions - after removing the outliers

#recheck if the variables are correct - left with 133 rows at this point
dim(Dataset)
names(Dataset) # should have 8 predictor variables

corr<-cor(Dataset)
ggcorrplot(corr, lab = TRUE)

#ANN
range.scale = function(x)
{
  (x - min(x)) / (max(x) - min(x))
}

scaled.data	= Dataset
scaled.data[,-1] = apply(Dataset[,-1], MARGIN = 2, FUN = range.scale) # NOT Scaling Response
#scaled.data[,] = apply(Dataset[,], MARGIN = 2, FUN = range.scale) # Scaling response 
head(scaled.data)

set.seed(1234) # starting point in generation of sequence of random numbers
#expand.grid: create a dataframe from all combinations of supplied vectors
#nnet: size = number of units in hidden layer.Here we assume just 1 hidden layer (unlike deep neural network)
#nnet: decay = regularization parameter to avoid overfitting

nnet.grid = expand.grid(size = 1:10, decay = c(0, 0.01, 0.1, 0.5, 1, 1.5, 2))
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
                          
    tmp.nn.model  = nnet(norm365boft ~ ., data = left.in.data,  linout=T, size = nnet.grid$size[i], decay = nnet.grid$decay[i], maxit = 1000)   
       
    tmp.pred.nn  	= predict(tmp.nn.model, newdata = left.out.data)  

    tmp.mse[j]   = mean((tmp.pred.nn - left.out.data$norm365boft)^2)
    tmp.aae[j]   = abs(tmp.pred.nn - left.out.data$norm365boft)
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
# Method - nnet package, https://cran.r-project.org/web/packages/nnet/nnet.pdf
final.model.nn =  nnet(norm365boft ~ ., data = scaled.data, linout = T, size = 4, decay = 0.01, maxit = 1000, trace = TRUE)   
names(final.model.nn)

r.squared.nn = (cor(final.model.nn$fitted.values, scaled.data$norm365boft))^2
r.squared.nn

train.rmse.nn = sqrt(mean((final.model.nn$fitted.values - scaled.data$norm365boft)^2))
train.rmse.nn
test.rmse.nn

train.aae.nn = mean(abs(final.model.nn$fitted.values - scaled.data$norm365boft))
train.aae.nn
test.aae.nn


# --------------Plot-----------------------------------------

names(final.model.nn)
X1 = final.model.nn$fitted.values
Y =  scaled.data$norm365boft

windows();plot(X1 ~ Y, pch = 1, xlab = "Observed Norm 365 Oil (bbl/ft)", ylab = "ANN Predicted Norm 365 Oil (bbl/ft)", xaxs ="i", yaxs ="i",
     main = "ANN Model", xlim = c(0,30), ylim = c(0,30))
legend("topleft", bty = "n", legend=paste("R2 = ", format(r.squared.nn,digits=4)))
#h = lm(X1 ~ Y) # Linear Model
#abline(h, col="blue", lwd = 2)
abline(a=0, b = 1, lty = 1, lwd=2, col="red") # Here a = intercept, b = slope


#dev.off()
source("C:/Users/AyushRastogi/OneDrive/LOS Files/Midland Basin - ANN ACE Comparison/OPAAT_sensitivity.R")# the datafile here should have Norm365(Target variable) as the first column
source("C:/Users/AyushRastogi/OneDrive/LOS Files/Midland Basin - ANN ACE Comparison/ACE_Wolfcamp.R")

df <- Predictions_ace
df2 <- Predictions_ace_PB
df3 <- Predictions_ols
df4 <- Predictions_PB
df_nn <- OPAAT_sensitivity(scaled.data, final.model.nn) # sensitivity for neural network Model - 8 predictors

dim(df_nn)

# Prediction Comparison - GOR
windows();
ggplot()+
  geom_line(data = df, aes(x = df$GOR365, y = df$`GOR365  Response`), color = "brown", size = 1)+
  geom_line(data = df2, aes(x = df2$GOR365, y = df2$`GOR365  Response`), color = "blue", size = 1)+
  geom_line(data = df3, aes(x = df3$GOR365, y = df3$`GOR365  Response`), color = "green", size = 1)+
  #geom_line(data = df4, aes(x = df4$GOR365, y = df4$`GOR365  Response`), color = "yellow", size = 1)+
  geom_line(data = df_nn, aes(x = df$GOR365, y = df_nn$`GOR365  Response`), color = "red", size = 1)+
  geom_point(data = Dataset, aes(x=Dataset$GOR365, y = Dataset$norm365boft), colour = "black", fill =NA, alpha = 0.1)+
  xlab('GOR')+
  ylab('Prediction')


par(mfrow=c(1,1))
# Prediction Comparison - TVD
windows();
ggplot()+
  geom_line(data = df, aes(x = df$tvd, y = df$`tvd  Response`), color = "brown", size = 1 )+
  geom_line(data = df2, aes(x = df2$tvd, y = df2$`tvd  Response`), color = "blue", size = 1 )+
  geom_line(data = df3, aes(x = df3$tvd, y = df3$`tvd  Response`), color = "green", size = 1)+
  #geom_line(data = df4, aes(x = df4$tvd, y = df4$`tvd  Response`), color = "yellow", size = 1)+
  geom_line(data = df_nn, aes(x = df$tvd, y = df_nn$`tvd  Response`), color = "red", size = 1)+
  geom_point(data = Dataset, aes(x=Dataset$tvd, y = Dataset$norm365boft), colour = "black", fill =NA, alpha = 0.1)+
  xlab('TVD')+
  ylab('Prediction')


#---------------------------


