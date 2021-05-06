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

# INDICATE FOLDER PATH
######## --------------------

folder_path <- "/Users/franciscomansillanavarro/Google Drive File Stream/My Drive/Programming R/project/"

######## --------------------

# LIBRARIES
# --------------------
library(data.table)
library(dplyr)
library(outliers)
library(caret);
library(foreach);
library(doParallel);
library(e1071);
library(randomForest);
library(pROC)
# --------------------

#################### READ DATASETS #################### 

solar_data <- readRDS(file.path(folder_path,"solar_dataset.RData"));
stations_location <- read.table(file.path(folder_path, "station_info.csv"), sep = ",",header = T);
solar_data_cropped <- solar_data[1:(nrow(solar_data)-1796),1:99]
additional_variables <- readRDS(file.path(folder_path,"additional_variables.RData"));

# head(additional_variables)
# head(solar_data)
# head(stations_location)
# head(solar_data_cropped)

#dim(solar_data) -> [1] 6909  456
#dim(stations_location) -> 98 4
#dim(additional_variables) -> 6909 101
#dim(solar_data_cropped) -> [1] 5113  99

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
rm(i)
not_informed_count <- c()
conv <- seq(from = 10, to = 3, by = -0.25)
for (i in conv){
  not_informed_count <- c(not_informed_count,not_informed(additional_variables,i)$count);
}
not_informed_count;
plot(conv,not_informed_count,xlab = "Threshold",ylab = "Nº removed additional variables")


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

#Our data is until 2008 EoY. 

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


###START: Perform outlier detection

#Our data is until 2008 EoY. 

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

solar_data_cropped <- solar_data[1:(nrow(solar_data)-1796),1:99]

#We can conclude that we don't have any outliers from the matrix we have.

### OUTLIER DETECTION ----------

outlier_table <- c()
for(i in 1:98){
  if(length(boxplot.stats(unlist(solar_data_cropped[,i+1]),coef = 1.5,do.conf = TRUE,do.out = TRUE)$out)==0){
    outlier_table <- c(outlier_table,NaN)
  }
  else{  
    outlier_table <- c(outlier_table,boxplot.stats(unlist(solar_data_cropped[,i]),coef = 1.5,do.conf = TRUE,do.out = TRUE)$out)
  }
}

98- sum(is.na(outlier_table)) #We have no outliers in any of the columns

### OUTLIER DETECTION ----------

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

dim(additional_variables_imputed)

# We define a subset caled "complete_set" with our training set and additional variables cleaned.
complete_set <- cbind(solar_data[1:5113,],additional_variables_imputed[1:5113,-1])
dim(complete_set)
#################### MODELING ############################

set.seed(140);
stopImplicitCluster();
registerDoParallel(cores = detectCores());

rm(i)
comp<- c()
vec_models <- c()
matrix_final_prediction <- matrix(0, nrow = 1796, ncol = 98)

## Selection of most significant PCA variables
# Here we do not need to iterate again to select the most important variables, as we already have them computed from 
# previous iterations. We keep them in a table "a"
a <- read.csv(file.path(folder_path,"PCA_list.csv"),header=T,sep=",")

modelvec <- c()
# We have 98 stations and we will iterate through them
for(i in 1:98){
  j <- i+1
  # We match in our complete_set the most important variables for that station
  dat<-as.data.table(as.data.frame(complete_set)[,match(a[,i],colnames(complete_set))])
  dat<- cbind(complete_set[,j,with=FALSE],dat)
  head(complete_set)
  dim(dat)
  
  if (i==1){
    # row indices for training data (70%)
    train_index <- sample(1:nrow(dat), 0.8*nrow(dat));  
    # row indices for test data (20%)
    test_index <- sample(setdiff(1:nrow(dat), train_index), 0.2*nrow(dat));  
    
  }else{
    train_index <- train_index
    test_index <- test_index
  }
  
  
  # split data
  train <- dat[train_index]; 
  test  <- dat[test_index];
  
  dim(dat);
  dim(train);
  dim(test);
  
  y <- as.matrix(train[,1])
  # train Linear model
  model <- lm(y ~ ., data = train[,-1]);
  
  # Get model predictions
  predictions_train <- predict(model, newdata = train);
  predictions_test <- predict(model,newdata = test);
  
  # Get errors
  errors_train <- predictions_train - train[,1,with=FALSE];
  errors_test <- predictions_test - test[,1,with=FALSE];
  
  # Compute Metrics
  mae_train <- round(unlist(lapply(abs(errors_train),mean)), 2);
  mae_test <- round(unlist(lapply(abs(errors_test),mean)), 2);
  
  ## Summary
  comp <- c(comp,data.table(model = paste("LM",colnames(y)), mae_train = mae_train,
                            mae_test = mae_test));
  modelvec <- c(modelvec,model$coefficients)
  datprediction <- as.data.table(as.data.frame(solar_data)[,match(a[,i],colnames(solar_data))])
  matrix_final_prediction[,i] <- predict(model,newdata=datprediction[5114:nrow(datprediction),])
}

## Our "a_matrix" will contain all the coefficients of the lm for the different stations.
a_matrix <- matrix(0, nrow = 11, ncol = 98)
a_matrix <- as.data.table(a_matrix)
colnames(a_matrix) <- colnames(complete_set)[2:(ncol(a_matrix)+1)]

for(i in 1:98){
  for(j in 1:11){
  a_matrix[j,i] <- modelvec[j+11*(i-1)]
  }
}
matrix_final_prediction <- cbind(solar_data[5114:nrow(solar_data),"Date"],matrix_final_prediction)
colnames(matrix_final_prediction) <- colnames(solar_data[,1:99])

## We save the model ready to be uploaded to Kaggle
write.csv(matrix_final_prediction, file.path(folder_path, "kaggle_lm2.csv"), row.names = FALSE)

## We save the coefficients to be used on the Rmarkdown
write.csv(a_matrix, file.path(folder_path, "coefficientes_lm.csv"), row.names = FALSE)