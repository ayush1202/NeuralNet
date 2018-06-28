# Code for ANN based models on DJ Basin dataset
# install.packages("ggcorrplot")
# install.packages("devtools")
# install.packages("caret")
# install.packages("mlbench")
# install.packages("RRF")
# install.packages("NeuralNetTools")

library(ggcorrplot)
library(caret)
library(mlbench)

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
names(Dataset) # Check the names after renaming

# ------------------Variable Importance/Feature Selection-----------------

# windows();pairs(Dataset) # matrix of scatterplots
corr <- round(cor(Dataset), 2) # correlation matrix, use 70% as cutoff
windows(); ggcorrplot(corr, hc.order = TRUE, lab = TRUE) # hc.order = hierarchical clustering

# GLM
fit_glm = glm(Norm365~., data = Dataset)
summary(fit_glm) # Remove TVD, WaterSaturation, BtuGas, NeutronPor, Adj_WC
varImp(fit_glm)

# Feature Selection - Boruta Package
library(Boruta)
boruta.train <- Boruta(Dataset$Norm365~., data = Dataset, doTrace = 2, ntree = 500)
names(boruta.train)
print(boruta.train)

boruta_signif <- getSelectedAttributes(boruta.train, withTentative = TRUE)
print(boruta_signif)

windows();plot(boruta.train, xlab="", xaxt = "n", main="Feature Importance")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz <- colnames(boruta.train$ImpHistory))
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels), at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.8)

boruta.df <- attStats(boruta.train)
boruta.df

# Caret Package
set.seed(100)
rPartMod <- train(Norm365 ~ ., data=Dataset, method="rpart")
rpartImp <- varImp(rPartMod)
print(rpartImp)

# Random Forest 

library(randomForest)
library(dplyr)
library(ggplot2)

set.seed(42)
rf_out <- randomForest(Dataset$Norm365 ~ ., data=Dataset)
# Extracts variable importance (Mean Decrease in Gini Index)
# Sorts by variable importance and relevels factors to match ordering
var_importance <- data_frame(variable=setdiff(colnames(Dataset[,-1]), "Norm365"),
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

# --------------------------------------------------------------

# PCA and Analysis 

# install.packages("FactoMineR")
library(FactoMineR)
library("devtools")
# install_github("kassambara/factoextra")
library(factoextra)
summary(Dataset)

# install.packages("corrplot")
library(corrplot)
cor.mat <- round(cor(Dataset),2)
corrplot(cor.mat, type="upper", order="hclust", 
         tl.col="black", tl.srt=45)

# install.packages("PerformanceAnalytics")
library("PerformanceAnalytics")
chart.Correlation(Dataset, histogram=TRUE, pch=19)
# Distribution shown on the diagonal, bivariate scatter plots on bottom
# Value of correlation and significance levels
# Each significance level is associated to a symbol : 
# p-values(0, 0.001, 0.01, 0.05, 0.1, 1) <=> symbols("***", "**", "*", ".", " ")

# PCA 
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

# ---------------------------------------------------------------------
