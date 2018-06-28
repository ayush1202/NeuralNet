## Run predictions w.r.t each predictor while the other predictors are kept at their mean values
## Return a data frame containing the predicted response values as columns named by the predictor which was varied
## Inputs:
## train_data = training dataset for the model (Response variable should be column 1)
## model = linear model or ace model
## ll,ul = lower and upper limits of predictors
OPAAT_sensitivity <- function(train_data, model, ll=NULL, ul=NULL, steps=100, ace = FALSE){
  if (length(ll) != length(ul)) {
    stop("ll and ul must be the same length")
  }
  ### Build a single row dataframe containing the mean values of all predictors
  mean_values <- train_data[1,-1]
  for (i in 1:dim(mean_values)[2])
  {
    if (!is.factor(train_data[,i+1])) {
      mean_values[,i] <- mean(train_data[,i+1])
    } else {
      mean_values[,i] <- 0
      mean_values[,i] <- factor(mean_values[,i])
    }
  }
  
  if (is.null(ll) || is.null(ul)) {
    ll <- vector(mode = "numeric", length = dim(mean_values)[2])
    ul <- vector(mode = "numeric", length = dim(mean_values)[2])
    for (i in 1:length(ll)) {
      ll[i] = min(train_data[,i+1])
      ul[i] = max(train_data[,i+1])
    }
  }
  
  for (i in 1:length(ll)) {
    predictor <- seq(from = ll[i], to = ul[i], length.out = steps)
    response <- vector(mode = "numeric", length = length(predictor))
    new_data <- mean_values
    for (j in 1:length(predictor)) {
      new_data[1,i] <- predictor[j]
      if (ace == FALSE) {
        response[j] <- predict(model, newdata = new_data)
      } else {
        response[j] <- ace_predict(train_data = train_data, new_data = new_data, ace_model = model)
      }
    }
    temp_df = data.frame(predictor, response)
    names(temp_df) = c(names(mean_values)[i], paste(names(mean_values)[i], " Response"))
    if (i>1) {
      sensitivity_df = cbind(sensitivity_df, temp_df)
    } else {
      sensitivity_df = temp_df
    }
  }
  return(sensitivity_df)
}