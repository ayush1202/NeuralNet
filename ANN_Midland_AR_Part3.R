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

# install.packages('gridExtra')
# install.packages('pastecs')

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

library(corrplot)

# Import the libraries required for the SOM model
library(kohonen)
library(ggplot2)
library(pastecs)
library(gridExtra)
library(grid)

# Colour palette definition
pretty_palette <- c("#1f77b4", '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2')

# Import data set
Dataset_orig = read.csv(file.choose(), header = TRUE) # Import Midland_Basin_WolfcampB2.csv
Dataset_raw = read.csv("Data_filtered_Midland.csv", header = TRUE)


names(Dataset_orig)
names(Dataset_raw)

# 8 selected predictors
# Dataset <- Dataset_raw[,-1] # drop X column
# All Predictors
Dataset <- Dataset_orig

dim(Dataset)
names(Dataset)

# -----outlier detection-----

# removing the outliers in the dataset with 7 predictors
# Run this one at a time
source('C:/Users/AyushRastogi/OneDrive/LOS Files/Project 8 - ANN SVM DJ Basin/Ayush/NeuralNet/outlier.R')

outlierKD(Dataset, norm365boft)
outlierKD(Dataset, stab365WC)
outlierKD(Dataset, GOR365)
outlierKD(Dataset, avgppg)
outlierKD(Dataset, lbsft)
outlierKD(Dataset, tvd)
outlierKD(Dataset, pctvclay)
outlierKD(Dataset, avgperm)

outlierKD(Dataset, norm365boeft)
outlierKD(Dataset, latlength)
outlierKD(Dataset, ftstage)
outlierKD(Dataset, transftstage)
outlierKD(Dataset, translbsft)
outlierKD(Dataset, bbls.ft)
outlierKD(Dataset, maxrate)
outlierKD(Dataset, oilgrav)

sum(is.na(Dataset))
Dataset = na.omit(Dataset)

dim(Dataset) # 2293 x 16 
names(Dataset)


corr<-cor(Dataset)
ggcorrplot(corr, hc.order = TRUE, lab = TRUE)

# remove variables with high correlation
Dataset = Dataset[,-c(1,7,10,12)]

# -------------------SOM Model Training------------------- 

data_train <- Dataset
# Normalizing the variables, using the scale function
# converted to matrix form because SOM function only accepts matrix 
# https://cran.r-project.org/web/packages/kohonen/kohonen.pdf - Pg 18

desc_stat = stat.desc(data_train) 
desc_stat
# if original numbers required, remove the scale function in the line below - Try with both scale and no-scaling 
# Or use the aggregate function  (unscaled function)

scaled_val = scale(data_train) # mean centers and normalize all columns
# dim(scaled_val)
# unscaled_val = attr(scaled_val, 'scaled:scale') + attr(scaled_val, 'scaled:center')

# data_train_matrix <- as.matrix((scaled_val)) # SCALED DATA
data_train_matrix <- as.matrix(scaled_val) # kohonen accepts numeric matrices

names(data_train_matrix) <- names(data_train)
dim(data_train_matrix) # 287 x 12

# SOM Grid Initialization - 15*15 grid = 225 nodes
som_grid <- somgrid(xdim = 15, ydim=15, topo = c("hexagonal"), 
                    neighbourhood.fct = c("gaussian")) 

# options for somgrid(): topo = c("rectangular", "hexagonal"), neighbourhood.fct = c("bubble", "gaussian")

# Train the SOM model
system.time(som_model <- som(data_train_matrix, 
                             grid=som_grid, 
                             rlen=1000, #iterations or times to present data
                             alpha=c(0.01,0.01), # learning rate
                             keep.data = TRUE
))

source("C:/Users/AyushRastogi/OneDrive/LOS Files/Project 8 - ANN SVM DJ Basin/Ayush/NeuralNet/coolBlueHotRed.R")

lapply(data_train, as.numeric)
# Rescaling to plot original scales

#---------------Scaling and Rescaling ---------  
# var <-3 #define the variable to plot
# var_unscaled <- aggregate(as.numeric(data_train[,var]),
#                           by=list(som_model$unit.classif),
#                           FUN=mean, simplify=TRUE)[,2]
# # other option
# # scaled_back = scaled_val * attr(scaled_val, 'scaled:scale') + attr(scaled_val, 'scaled:center')
# plot(som_model, type = "property", property=dvar_unscaled,
#      main=names(data_train)[var], palette.name=coolBlueHotRed)

#--------------------------------------------

# plotHeatMap(som_model, data_train, variable=0) #interactive window for selection

# provides some important information about the model 
#including the mean distance to the closest unit in the map
summary(som_model) 

par(mfrow=c(1,1))
# Can be used to select the optimum model, the curve should flat out at certain point on x axis
plot(som_model, type = "changes")


# ----------------Plotting HeatMaps from SOM Model------------

# Plot the heatmap for a variable at scaled / normalised values

# Defining the variables to plot, the numbers in the dataframe below here denote the column number to be plotted 
# ColumnNumber <- c(10,11,12,13,16,17,18,19,20,26,27,29,33,34,35,37,51,52,54) # 19 variables to be looked in detail

# df <- data.frame(ColumnNumber)
df <- data.frame(Dataset)
df

# creating a simple multi-paneled plot
par(mfrow=c(2,2)) # to reset, enter the command: 
# par(mfrow=c(1,1))

# loop to plot SOM for each selected column
selected = c(1:12) # only selected columns to be plotted
# for (i in 1:25){
for (i in selected){
  plot(som_model, type = "property", property = getCodes(som_model)[,i] , main=colnames(getCodes(som_model))[i],
       palette.name=coolBlueHotRed)
}

# -------------------- SOM VISUALISATION ---------------------

#counts within nodes
plot(som_model, type = "counts", main="Node Counts", palette.name=coolBlueHotRed)
#map quality
plot(som_model, type = "quality", main="Node Quality/Distance", palette.name=coolBlueHotRed)
#neighbour distances
plot(som_model, type="dist.neighbours", main = "SOM neighbour distances", palette.name=grey.colors)
#code spread
plot(som_model, type = "codes")


# ------------------ Clustering SOM results -------------------

# show the WCSS metric for kmeans for different clustering sizes.
# minimizing the WCSS (within-cluster sums of squares) will maximize the distance between clusters
# Can be used as a "rough" indicator of the ideal number of clusters - Based on the elbow criterion
# GetCodes: Extract codebook vectors from a kohonen object

mydata <- getCodes(som_model)
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var)) # here the integer denotes the column, var = variance
for (i in 2:6) wss[i] <- sum(kmeans(mydata,
                                    centers=i)$withinss)

par(mar=c(5.1,4.1,4.1,2.1)) #plot characteristics, mar = 'margin'
plot(1:6, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares", main="Within cluster sum of squares (WCSS)")

# Form clusters on grid
## use Hierarchical clustering to cluster the codebook vectors
som_cluster <- cutree(hclust(dist(getCodes(som_model))), 6) # the integer is the number of clusters 

# Show the map with different colours for every cluster						  
plot(som_model, type="mapping", bgcol = pretty_palette[som_cluster], main = "Clusters")
add.cluster.boundaries(som_model, som_cluster)

#show the same plot with the codes instead of just colours
plot(som_model, type="codes", bgcol = pretty_palette[som_cluster], main = "Clusters")
add.cluster.boundaries(som_model, som_cluster)

# --------------------------------------------------------------


