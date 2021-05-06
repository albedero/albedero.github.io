folder_path <- "/Users/franciscomansillanavarro/Google Drive File Stream/My Drive/Programming R/project/"

# We load the sets with the different algorithms used
svmdatatest<-read.table(file.path(folder_path, "kaggle_svm.csv"), sep = ",",header = T);
lmdatatest<-read.table(file.path(folder_path, "kaggle_lm.csv"), sep = ",",header = T);
rfdatatest<-read.table(file.path(folder_path, "kaggle_RF.csv"), sep = ",",header = T);

setDT(svmdatatest)
setDT(lmdatatest)
setDT(rfdatatest)


## Comparison with Historical Mean. Our approach here was to compare the average value of our historical data for each station
# and select for each station, the model whose average value was closer to this value.

solar_data <- readRDS(file.path(folder_path,"solar_dataset.RData"));
solar_data_cropped2 <- solar_data[1:(nrow(solar_data)-1796),2:99]

comparison <- rbind(svmdatatest[,lapply(.SD,mean)],lmdatatest[,lapply(.SD,mean)],rfdatatest[,lapply(.SD,mean)])
comp <- abs(comparison - as.numeric(solar_data_cropped2[,lapply(.SD,mean)]))

## The indexcomp selects the algorithm to be used for each station: lm, svm or randomForest. 
indexcomp<- comp[,lapply(.SD,which.min)]
matrix_comp <- as.data.table(matrix(0, nrow = 1796, ncol = 98))
rm(j)
for(i in 1:98){
  j <- i+1
  if(indexcomp[1,..j]==1){
    matrix_comp[,i] <- svmdatatest[,..j]
  }else if(indexcomp[1,..j]==2){
    matrix_comp[,i] <- lmdatatest[,..j]
  }else
    matrix_comp[,i] <- rfdatatest[,..j]
}

## The matrix is filled with the precomputed values of the algorithms
matrix_comp <- cbind(svmdatatest$Date,matrix_comp)
colnames(matrix_comp) <- colnames(svmdatatest)

## The file .csv is ready to be uploaded to Kaggle and check the results (This model did not improve the results of SVM)
write.csv(matrix_comp, file.path(folder_path, "kaggle_compare.csv"), row.names = FALSE)

