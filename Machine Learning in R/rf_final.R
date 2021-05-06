##########################################################
#
# Team B - Programming R Workgroup Project
#
##########################################################
#
# Authors: Alain Grullon, Alberto De Roni, Rosamaria Mejia,  
#          Francisco Mansilla, Timo Bachmann, Umut Varol
#
##########################################################

# SET YOUR FOLDER PATH
folder_path <- "~/Desktop/MBD/Term-1/PROGRAMMING-R/project"

# LIBRARIES
# ------------------------
library(data.table)
library(dplyr)
library(outliers)
library(caret)
library(e1071) 
library(foreach)
library(doParallel)
library(randomForest)
library(pROC)
# ------------------------
# The purpose of this script is to tune the hyperparameters of the randomForest algorithm
# without having the explicitly run the variable importance/selection for each target.
# Results may deviate from the obtained algorithm predictions.
# ------------------------

#################### READ DATASETS ############################

solar_data <- readRDS(file.path(folder_path,"solar_dataset.RData"));
stations_location <- read.table(file.path(folder_path, "station_info.csv"), sep = ",",header = T);
solar_data_cropped <- solar_data[1:(nrow(solar_data)-1796),1:99]

#dim(solar_data) -> [1] 6909  456
#dim(stations_location) -> 98 4
#dim(additional_variables) -> 6909 101
#dim(solar_data_cropped) -> [1] 5113  99

additional_variables <- readRDS(file.path(folder_path,"additional_variables.RData"));


### Understand data

class <- as.data.table(lapply(solar_data,class))[1]
summary(solar_data) #for only the turbines, put [,2:99]

#### With the following: We want to see how many variables would be exclude if our tolerance of missing
# values would be between 3% missing value and 10% missing. We see that none vairable has more than approx 7%

not_informed <- function(dat, threshold_nas){
  ratio_nas <- sapply(dat, function(x){100*sum(is.na(x))/length(x)});
  not_informed_var <- names(ratio_nas)[ratio_nas > threshold_nas];
  count <- length(not_informed_var)
  return(list(not_informed_var=not_informed_var, count = count));
}


not_informed_count <- c()
conv <- seq(from = 10, to = 3, by = -0.25)

for (i in conv){
  not_informed_count <- c(not_informed_count,not_informed(additional_variables,i)$count);
}

not_informed_count
plot(conv, not_informed_count, xlab = "Threshold", ylab = "N° removed additional variables")


# IMPUTE WITH MEDIAN NAS FROM ADDITIONAL VARIABLES

fill_missing_with_median <- function(x){
  x <- as.numeric(x);
  x[is.na(x)] <- median(x, na.rm = TRUE);
  return(x);
}

additional_variables_imputed <- additional_variables[,lapply(.SD,fill_missing_with_median), .SDcols= setdiff(colnames(additional_variables),"Date")]
dim(additional_variables_imputed)


## WE CHECK THAT WE IMPUTED THE NAN 

check <- c();
for (i in 1:ncol(additional_variables_imputed)){
  check <- c(check,sum(is.na(additional_variables_imputed[,..i])))
}
check
# All our NAN have been removed

# Our data is until 2008 EoY. 

# count_nas <- function(x){
#   ret <- sum(is.na(x));
#   return(ret);
# }

# sapply(solar_data[,2:ncol(solar_data)],count_nas)

#It can be seen that the NAs belong to the test / to-foreast dates which 
#have the same number for each column: 5114. And considering the number of NAs
#for each column being the same (1796), we can prove that the data is missing for the
#same entries for each column. (Note: The date column is complete anyway)
#Thus, the rows will be omitted for the data analysis for futher forecasting.

NA_matrix_dt <- as.data.table(!is.na(solar_data[,2:99]))
NA_matrix_dt[,apply(.SD,2,which.min)] 

sum(sum(NA_matrix_dt[,apply(.SD,2,which.min)]<5114),sum(NA_matrix_dt[,apply(.SD,2,which.min)]>5114))

#We can conclude that we don't have any outliers from the matrix we have.


### OUTLIER DETECTION ------------

outlier_table <- c()

for(i in 1:98){
  if(length(boxplot.stats(unlist(solar_data_cropped[,i+1]),coef = 1.5,do.conf = TRUE,do.out = TRUE)$out)==0){
    outlier_table <- c(outlier_table,NaN)
  }
  else{  
    outlier_table <- c(outlier_table,boxplot.stats(unlist(solar_data_cropped[,i]),coef = 1.5,do.conf = TRUE,do.out = TRUE)$out)
  }
}

98 - sum(is.na(outlier_table)) #We have no outliers in any of the columns

### OUTLIER DETECTION ------------


#################### CORRELATION OF ADDITIONAL VARIABLES #################### 

# REDUNDANT VARIABLES

cors <- abs(cor(additional_variables_imputed));

remove_redundant <- function(correlations,redundant_threshold){
  l <- list()
  for(i in 1:ncol(correlations)){
    mat <- sort(abs(correlations)[,i],decreasing = TRUE)
    l[[i]] <- labels(mat[mat > redundant_threshold][-1])
  }
  return(l)
}

threshold <- 0.7
redundant_vars <- remove_redundant(cors, threshold);


#################### SELECT MOST IMPORTANT VARIABLES FOR EACH STATION ############################

PCA_data <- cbind(solar_data,additional_variables)
class(PCA_data)
head(PCA_data)
#dim(PCA_data) -> 6909 557

select_important<-function(dat, n_vars, y){
  varimp <- filterVarImp(x = dat[1:5113,100:ncol(dat)], y=y, nonpara=TRUE);
  varimp <- data.table(variable=rownames(varimp),imp=varimp[, 1]);
  varimp <- varimp[order(-imp)];
  selected <- varimp$variable[1:n_vars];
  return(selected);
}

# Dimension explations
dim(solar_data)
dim(solar_data_cropped)
dim(additional_variables_imputed)

# We deine a subset called "complete_set" with our training set data and additional variables.
complete_set <- cbind(solar_data[1:5113,],additional_variables_imputed[1:5113,-1])
dim(complete_set)

#################### MODELING ############################

set.seed(140);

### Selection of most significant PCA variables
# Below "a_list" gives an optimized list of features to include in the machine learning algorithm.
# As it performs a stepwise variable importance selection, the loop may take up to several hours
# and is a prerequisite of the following computations.

a_list <- c()
for (z in 2:99){
  
  d <- list(select_important(dat = solar_data, n_vars = 10, y = unlist(solar_data[1:5113,..z])));
  a_list <- c(a_list,d)
  
}

# Definition of "a_matrix", which is filled with the optimal features for every target (=station). 
# The table will be completed after model fitting and is extracted as a RMarkdown input.
a_matrix <- matrix(0, nrow = 10, ncol = 98)
a_matrix <- as.data.table(matrix(unlist(a_list),nrow=10,byrow=F))
colnames(a_matrix) <- colnames(complete_set)[2:(ncol(a_matrix)+1)]

# Here, we provide the list of optimal PCA variables for future model optimizations.
# The list will also be used for other machine learning algorithm computations to reduce fitting time.
write.csv(a_matrix, file.path(folder_path, "PCA_list.csv"), row.names = FALSE) 

# Remove i,j to ensure a smooth model fitting loop. Further preparation steps include an empty 
# vector to facilitate information extraction from each iteration. Finally, we also prepare 
# our output table "matrix_final_predictions".
rm(i)
rm(j)
comp<- c()
matrix_final_prediction <- matrix(0, nrow = 1796, ncol = 98)

### RANDOM FOREST optimization loop
# We run a for-loop through each target station as a dependent variable. Every run will again perform
# hyperparameter optimization on an individual level.

for(i in 1:98){
  
  j <- i+1
  
  # Get list of pre-identified optimal explanatory variables and 
  # define the relevant dataset for taget station i.
  a <- a_list[[i]]
  dat<-as.data.table(as.data.frame(complete_set)[,match(a[[1]],colnames(complete_set))])
  dat<- cbind(complete_set[,j,with=FALSE],dat)

  #head(complete_set)
  #dim(dat)
  
  if (i==1){
    # row indices for training data (70%)
    train_index <- sample(1:nrow(dat), 0.7*nrow(dat));  
    # row indices for validation data (15%)
    val_index <- sample(setdiff(1:nrow(dat), train_index), 0.15*nrow(dat));  
    # row indices for test data (15%)
    test_index <- setdiff(1:nrow(dat), c(train_index, val_index));
  }else{
    train_index <- train_index
    val_index <- val_index
    test_index <- test_index
  }
  
  # Split data into train, validation and test sets
  train <- dat[train_index]; 
  val <- dat[val_index]; 
  test  <- dat[test_index];
  
  dim(dat);
  dim(train);
  dim(val);
  dim(test);

  # Define dependent variable y
  y <- as.matrix(train[,1])
  
  #### START HYPERPARAMETER OPTIMIZATION - RANDOM FOREST ####
  stopImplicitCluster();
  registerDoParallel(cores = detectCores());
  
  ### Define grid
  # n_trees was found optimal as 100 over several runs, hence we fix this value 
  # to increase computation speed of the model fit
  n_trees <- 100
  mtry_values <- seq(from = 2, to = 15, length.out = 14);
  min_nodesize_values <- seq(from = 2, to = 8, length.out = 7);
 
  ### Compute grid search in FOREACH loop
  grid_results <-  foreach (ntree = n_trees, .combine = rbind)%:%
    foreach (mtry = mtry_values, .combine = rbind)%:%
    foreach (nodesize = min_nodesize_values, .combine = rbind)%dopar%{
      
      library(randomForest);
      
      print(sprintf("Start of ntree = %s - mtry = %s - min_nodesize = %s", ntree, mtry, nodesize));
      
      model <- randomForest(y ~ ., data = train[,-1], ntree = n_trees, mtry = mtry_values, nodesize = min_nodesize_values);
      
      # Get model predictions
      predictions_train <- predict(model, newdata = train);
      predictions_val <- predict(model, newdata = val);
      
      # Get errors
      errors_train <- predictions_train - train[,1,with=FALSE];
      errors_val <- predictions_val - val[,1,with=FALSE];
      
      # Compute MAE metric
      mae_train <- round(unlist(lapply(abs(errors_train),mean)), 2);
      mae_val <- round(unlist(lapply(abs(errors_val),mean)), 2);
      
      # Build comparison table 
      data.table(y=colnames(y),mae_train = mae_train,
                 mae_val = mae_val,
                 ntree = n_trees, mtry = mtry_values, nodesize = min_nodesize_values);
    }
  
  # Order results by increasing mse and mae
  grid_results <- grid_results[order(mae_val, mae_train)];
  
  # Check results
  best <- grid_results[1];
  
  ### Train final model
  # Train RandomForest model with optimal obtained set of hyperparameters
  model <- randomForest(y ~ ., data = train[,-1], ntree = best$ntree, mtry = best$mtry, nodesize = best$nodesize);
  
  # Get model predictions
  predictions_train <- predict(model, newdata = train);
  predictions_val <- predict(model, newdata = val);
  predictions_test <- predict(model, newdata = test);
  
  #  Get errors
  errors_train <- predictions_train - train[,1,with=FALSE];
  errors_val <- predictions_val - val[,1,with=FALSE];
  errors_test <- predictions_test - test[,1,with=FALSE];
  
  # Compute MAE metric
  mae_train <- round(unlist(lapply(abs(errors_train),mean)), 2);
  mae_val <- round(unlist(lapply(abs(errors_val),mean)), 2);
  mae_test <- round(unlist(lapply(abs(errors_test),mean)), 2);
  
  
  ## Summary
  comp <- c(comp,data.table(model = paste("RF",colnames(y)), mae_train = mae_train,
                            mae_test = mae_test,
                            ntree=best$ntree,
                            mtry=best$mtry,
                            min_nodesize=best$nodesize
                            ));
  
  
  # Build a final prediction table
  datprediction <- as.data.table(as.data.frame(solar_data)[,match(a[[1]],colnames(solar_data))])
  
  # Add optimal prediction for target station i to collective model output table
  matrix_final_prediction[,i] <- predict(model,newdata=datprediction[5114:nrow(datprediction),])
}

################## END OF ITERATION ------------------------
  

# Complete final prediction table after model iteration
matrix_final_prediction <- cbind(solar_data[5114:nrow(solar_data),"Date"],matrix_final_prediction)
colnames(matrix_final_prediction) <- colnames(solar_data[,1:99])

# Save final prediction table ready to upload on Kaggle
write.csv(matrix_final_prediction, file.path(folder_path, "kaggle_RF.csv"), row.names = FALSE)


# In order to complete our RMarkdown tables on iterations and obtained optimal hyperparameter values,
# we have to extract some selected values from the "comp".
full_matrix_RF <- rbind(a_matrix,as.data.table(matrix(0,nrow=4,ncol=2)),use.names=FALSE)
for(i in 1:98){
  full_matrix_RF[11,i] <- comp[[6*(i-1)+3]]
  full_matrix_RF[12,i] <- comp[[6*(i-1)+4]]
  full_matrix_RF[13,i] <- comp[[6*(i-1)+5]]
  full_matrix_RF[14,i] <- comp[[6*(i-1)+6]]
}

# Ultimately, we save a full table including a set of optimal variables, Mean Absolute Error, and 
# iterated optimal hyperparameter values.
write.csv(full_matrix_RF, file.path(folder_path, "full_matrix_RF.csv"), row.names = FALSE)


#
### end of Random Forest script ###
#