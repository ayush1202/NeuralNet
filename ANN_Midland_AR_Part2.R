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


# Import data set
Dataset_raw = read.csv(file.choose(), header = TRUE) # Import Midland_Basin_WolfcampB2.csv

colnames(Dataset_raw) #column names - Not represented in best way, so better to rename them 
Dataset <- Dataset_raw # copying it to another df so original remains intact

Dataset = Dataset_raw[, -c(1,5,6,7,10,11,12)]
dim(Dataset)

# removing norm365boeft, latlength, ftstage, transftstage, translbft, bbls.ft, maxrate


# --------------------EDA-------------------

introduce(Dataset)
plot_str(Dataset) # plot - network type structure
plot_missing(Dataset) # plot of missing values for each variable as a percentage
# plot_bar(Dataset) # bar chart for discrete variables only
plot_histogram(Dataset) # plot of histogram with all variables
plot_boxplot(Dataset, by = "norm365boft")
#plot_correlation(Dataset)
plot_density(Dataset)
plot_scatterplot(Dataset, "norm365boft", ggtheme = theme_grey())
# plot_prcomp(data = Dataset) # PCA for %variance explained -  No discrete features
# Another Option to use plots from  https://rpubs.com/plisk/DataExplorer - R Shiny Tool
# create_report(Dataset)

# missing values
missmap(Dataset, col=c("black", "grey"), legend=FALSE)


# Histogram of all predictor variables
# operator passes the dataframe output that results from the function right before the pipe to input 
# it as the first argument of the function right after the pipe.

Dataset %>% 
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()


# Data Cleanup
sum(is.na(Dataset)) # total number of NA values in the dataset
desc_stat = stat.desc(Dataset) 
desc_stat


names(Dataset) # feature names
dim(Dataset) #dimensions
str(Dataset) #structure
head(Dataset) # top 6 values of the dataset


# -----outlier detection-----

# removing the outliers in the dataset with 7 predictors
# Run this one at a time
source('C:/Users/AyushRastogi/OneDrive/LOS Files/Project 8 - ANN SVM DJ Basin/Ayush/NeuralNet/outlier.R')
outlierKD(Dataset, norm365boft)
outlierKD(Dataset, stab365WC)
outlierKD(Dataset, GOR365)
outlierKD(Dataset, avgppg)
outlierKD(Dataset, lbsft)
outlierKD(Dataset, oilgrav)
outlierKD(Dataset, tvd)
outlierKD(Dataset, pctvclay)
outlierKD(Dataset, avgperm)



dim(Dataset) #dimensions
Dataset = na.omit(Dataset) # # Out of 339 rows, only 133 observations left - All outliers removed
dim(Dataset) #dimensions - after removing the outliers

#drawing the histograms again

Dataset %>% 
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()


# ---- Performance Analytics

chart.Correlation(Dataset, histogram=TRUE, pch=19)

# -------- Feature Selection

dim(Dataset) # 1443 x 9 
# Removing the points with NA

# 1. Correlation Plot
# windows();pairs(Dataset) # matrix of scatterplots
corr <- round(cor(Dataset), 2) # correlation matrix, use 70% as cutoff
#windows(); #ggcorrplot(corr, hc.order = TRUE, lab = TRUE, type ='lower') # hc.order = hierarchical clustering
windows();ggcorrplot(corr, hc.order = TRUE, lab = TRUE, type = "lower",
           outline.col = "white",
           ggtheme = ggplot2::theme_gray,
           colors = c("#6D9EC1", "white", "#E46726"))

# 2. GLM
fit_glm = glm(norm365boft~., data = Dataset)
summary(fit_glm) # Remove TVD, WaterSaturation, BtuGas, NeutronPor, Adj_WC
varImp(fit_glm)

# 3. Feature Selection - Boruta Package

boruta.train <- Boruta(norm365boft~., data = Dataset, doTrace = 2, ntree = 1000)
names(boruta.train)
print(boruta.train)
plot(boruta.train)
boruta.train$finalDecision

boruta_signif <- getSelectedAttributes(boruta.train, withTentative = TRUE)
print(boruta_signif)

plot(boruta.train, xlab="", xaxt = "n", main="Feature Importance", ylab="")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels), at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.8)

boruta.df <- attStats(boruta.train)
boruta.df

# 4. Caret 
set.seed(100)

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
results <- rfe(Dataset[,2:9], Dataset[,1], sizes=c(2:9), rfeControl=control)
print(results)
predictors(results)
plot(results, type=c("g", "o"))

rPartMod <- train(norm365boft ~ ., data=Dataset, method="rpart")
rpartImp <- varImp(rPartMod)
print(rpartImp)
plot(rpartImp)

# 5. Random Forest


set.seed(42)
rf_out <- randomForest(norm365boft ~ ., data=Dataset)
# Extracts variable importance (Mean Decrease in Gini Index)
# Sorts by variable importance and relevels factors to match ordering
var_importance <- data_frame(variable=setdiff(colnames(Dataset[,-1]), "norm365boft"),
                             importance=as.vector(importance(rf_out)))
var_importance <- arrange(var_importance, desc(importance))
var_importance$variable <- factor(var_importance$variable, levels=var_importance$variable)

p <- ggplot(var_importance, aes(x=variable, weight=importance, fill=variable))
p <- p + geom_bar() + ggtitle("Variable Importance from Random Forest Fit")
p <- p + xlab("Variables") + ylab("Variable Importance (Mean Decrease in Gini Index)")
p <- p + scale_fill_discrete(name="Variable Name")
p + theme(axis.text.x=element_blank(),
          axis.text.y=element_text(size=12),
          axis.title=element_text(size=16),
          plot.title=element_text(size=18),
          legend.title=element_text(size=16),
          legend.text=element_text(size=12))

# 6. PCA 
# library(FactoMineR)
PCA_results = PCA(Dataset, scale.unit = TRUE, ncp = 5, graph = TRUE)
print(PCA_results) #also prints individual and variables factor map

# Eigenvalues correspond to the amount of the variation explained by each principal component (PC)
eigenvalues <- PCA_results$eig
head(eigenvalues[, 1:2]) # taking the eigenvalue and percent of variance into account

# Printing the bar plot
barplot(eigenvalues[, 2], names.arg=1:nrow(eigenvalues), 
        main = "Variances",
        xlab = "Principal Components",
        ylab = "Percentage of variances",
        col ="steelblue")
# Add connected line segments to the plot
lines(x = 1:nrow(eigenvalues), eigenvalues[, 2], 
      type="b", pch=19, col = "red")

# Scree Plot

fviz_screeplot(PCA_results, ncp=10)
# --------------------------

# Based on Feature Selection results - remove oilgrav - 6th
Dataset <- Dataset[,-c(6)]

colnames(Dataset)
dim(Dataset)

getwd()
write.csv(Dataset, file = "Data_filtered_Midland.csv")


names(Dataset) # should have 7 predictor variables

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

# Method - neuralnet package
# library(neuralnet)
# final.model.neunet1 = neuralnet(Dataset$norm365boft~Dataset$stab365WC+Dataset$GOR365+Dataset$avgppg+
#                                   Dataset$lbsft+Dataset$tvd+Dataset$pctvclay+Dataset$avgperm, rep = 100,
#                                 data = scaled.data, hidden = 4, err.fct = "sse", linear.output = FALSE)
# plot(final.model.neunet1, rep ="best")


# names(final.model.neunet1)
# final.model.neunet1$err.fct
# 
# # train.score <- sapply(final.model.neunet1,function (x) {min(x$result.matrix[c("error"),])})
# 
# cat(paste(c("Training Scores (Logarithmic Loss)\n1 Hidden Layer, 5 Hidden Units:", "1 Hidden Layer, 16 Hidden Units:",
#             "2 Hidden Layer, 5 Hidden Units:", "2 Hidden Layers, 16 Hidden Units Each:"), train.score, collapse = "\n"))
# 
# final.model.neunet1$response
# 
# plotnet(final.model.neunet1)
# 
# dim(scaled.data)




# Method - nnet package, https://cran.r-project.org/web/packages/nnet/nnet.pdf
final.model.nn =  nnet(norm365boft ~ ., data = scaled.data, linout = T, size = 4, decay = 0.005, maxit = 1000, trace = TRUE)   
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
abline(a=0, b = 1, lty = 1, lwd = 2, col="red") # Here a = intercept, b = slope


#dev.off()
source("C:/Users/AyushRastogi/OneDrive/LOS Files/Midland Basin - ANN ACE Comparison/OPAAT_sensitivity.R")# the datafile here should have Norm365(Target variable) as the first column
source("C:/Users/AyushRastogi/OneDrive/LOS Files/Midland Basin - ANN ACE Comparison/ACE_Wolfcamp.R")

# df <- Predictions_ace
# df2 <- Predictions_ace_PB
df3 <- Predictions_ols
# df4 <- Predictions_PB
df_nn <- OPAAT_sensitivity(scaled.data, final.model.nn) # sensitivity for neural network Model - 8 predictors

dim(df_nn)

par(mfrow=c(1,1))
# Prediction Comparison - GOR
windows();
ggplot()+
  # geom_line(data = df, aes(x = df$GOR365, y = df$`GOR365  Response`), color = "yellow", size = 1)+
  # geom_line(data = df2, aes(x = df2$GOR365, y = df2$`GOR365  Response`), color = "blue", size = 1)+
  geom_line(data = df3, aes(x = df3$GOR365, y = df3$`GOR365  Response`), color = "gray", size = 1)+
  # geom_line(data = df4, aes(x = df4$GOR365, y = df4$`GOR365  Response`), color = "pink", size = 1)+
  geom_line(data = df_nn, aes(x = df3$GOR365, y = df_nn$`GOR365  Response`), color = "red", size = 1)+
  geom_point(data = Dataset, aes(x=Dataset$GOR365, y = Dataset$norm365boft), colour = "black", fill =NA, alpha = 0.1)+
  xlab('GOR')+
  ylab('Prediction')



# Prediction Comparison - TVD
windows();
ggplot()+
  #geom_line(data = df, aes(x = df$tvd, y = df$`tvd  Response`), color = "yellow", size = 1 )+
  #geom_line(data = df2, aes(x = df2$tvd, y = df2$`tvd  Response`), color = "blue", size = 1 )+
  geom_line(data = df3, aes(x = df3$tvd, y = df3$`tvd  Response`), color = "gray", size = 1)+
  #geom_line(data = df4, aes(x = df4$tvd, y = df4$`tvd  Response`), color = "pink", size = 1)+
  geom_line(data = df_nn, aes(x = df3$tvd, y = df_nn$`tvd  Response`), color = "red", size = 1)+
  geom_point(data = Dataset, aes(x=Dataset$tvd, y = Dataset$norm365boft), colour = "black", fill =NA, alpha = 0.1)+
  xlab('TVD')+
  ylab('Prediction')


#---------------------------


