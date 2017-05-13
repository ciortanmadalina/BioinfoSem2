library(dummies)
library(ggplot2)
require(corrplot) # correlation plot
library(rpart)
library(e1071)
library(lazy)
library(nnet)
library(tree)
library(ridge)

rm(list=ls(all=TRUE))

input <- read.csv("train.csv", stringsAsFactors = FALSE)
output <- read.csv("test.csv", stringsAsFactors = FALSE)


#Impute missing value on a global dataset
combined <- input #or within(input, rm("SalePrice"))
combined$SalePrice <- NULL
combined <- rbind(combined, output)


factor_variables<-which(sapply(combined[1,],class)=="character")

numeric.df<-combined[,-factor_variables]
categoric.df<-combined[,factor_variables]



#By looking at the histograms for categorical data remove all feature where most values fall into 1 cat
one_dominant_feature <- c('Street',"LandContour","Utilities", "LandSlope", 'Condition1', 'Condition2', 'BldgType', 'RoofMatl',
                          'ExterCond', 'BsmtCond', 'BsmtFinType2', 'Heating', 'Electrical', 'Functional', 'GarageQual',
                          "GarageCond","PavedDrive" , "SaleType" , "SaleCondition")

categoric.df <- categoric.df[,setdiff(names(categoric.df), one_dominant_feature)]

#Remove all features for which we don't have enough data
not_enough_data <- c('MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'GarageFinish', 'BsmtQual', 'BsmtExposure', 
                     'BsmtFinType1', 'MasVnrType', 'GarageType', 'CentralAir' , 'Alley')

categoric.df <- categoric.df[,setdiff(names(categoric.df), not_enough_data)]


#Impute values
categoric.df[is.na(categoric.df$MSZoning), 'MSZoning'] <-names(sort(table(categoric.df$MSZoning), decreasing = TRUE)[1])
categoric.df[is.na(categoric.df$Exterior1st), 'Exterior1st'] <-names(sort(table(categoric.df$Exterior1st), decreasing = TRUE)[1])
categoric.df[is.na(categoric.df$Exterior2nd), 'Exterior2nd'] <-names(sort(table(categoric.df$Exterior2nd), decreasing = TRUE)[1])
categoric.df[is.na(categoric.df$KitchenQual), 'KitchenQual'] <-names(sort(table(categoric.df$KitchenQual), decreasing = TRUE)[1])


#Make sure there are no missing values
colSums(is.na(categoric.df))

#one hot encoding phase
data_factor_onehot <- dummy.data.frame(categoric.df, sep="_")
dim(data_factor_onehot)

##########################################################
#Numeric variables imputation
##########################################################

colSums(is.na(numeric.df))
numeric.df$Id <-NULL #remove id

#Let's find relationships between features

#All basement features seem to be related
bsmt <- numeric.df[, c('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF')]
bsmt$SumBsmt <- bsmt$BsmtFinSF1 + bsmt$BsmtFinSF2 + bsmt$BsmtUnfSF
par(mfrow=c(1,1))
corrplot(cor(bsmt,use="complete.obs"),type = 'upper', method='color', addCoef.col = 'green')
#because TotalBsmtSF is totally corellated with the sum of 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF' we
#can simplify our model by keeping just the total
numeric.df$BsmtFinSF1 <- NULL  
numeric.df$BsmtFinSF2 <- NULL 
numeric.df$BsmtUnfSF <- NULL 


#Ground area surfaces also seem to be related
area <- numeric.df[, c('X1stFlrSF', 'X2ndFlrSF', 'LowQualFinSF', 'GrLivArea')]
area$SumArea <- area$X1stFlrSF + area$X2ndFlrSF + area$LowQualFinSF
par(mfrow=c(1,1))
corrplot(cor(area,use="complete.obs"),type = 'upper', method='color', addCoef.col = 'green')
#because GrLivArea is totally corellated with the sum of 'X1stFlrSF', 'X2ndFlrSF', 'LowQualFinSF' we
#can simplify our model by keeping just the total
numeric.df$X1stFlrSF <- NULL  
numeric.df$X2ndFlrSF <- NULL 
numeric.df$LowQualFinSF <- NULL 

colSums(is.na(numeric.df))

#Remove features with a lot of missing data
numeric.df$LotFrontage <- NULL
numeric.df$GarageYrBlt <- NULL

#For remaining features let's impute with mean
replace_na_with_mean_value <- function(vec) {
  mean_vec <- mean(vec, na.rm = TRUE)
  vec[is.na(vec)] <- mean_vec
  vec
}


numeric.df <- data.frame(apply(numeric.df, 2, replace_na_with_mean_value))

colSums(is.na(numeric.df))

data <- cbind(numeric.df, data_factor_onehot)


##############################
####function declarations
##############################



rmse <- function (log_prediction, log_observation){
  sqrt(mean(log_prediction-log_observation)^2)
}



outputCorrelation <- function(modelName, X, Y){
  n <- ncol(X)
  size.CV<-floor(N/10)
  ranking <- numeric(n)
  CV.err<-matrix(0,nrow=n,ncol=10)
  
  for (i in 1:10) {
    i.ts<-(((i-1)*size.CV+1):(i*size.CV))  
    
    
    X.ts<-X[i.ts,]  
    Y.ts<-Y[i.ts]
    
    i.tr<-setdiff(1:N,i.ts)
    X.tr<-X[i.tr,]
    Y.tr<-Y[i.tr]
    
    #sometimes in the sampling process sd is 0 so let's remove those features because
    #calculating corelation on them would be a division by 0
    #X.tr <- X.tr[,which(apply(X.tr, 2, sd)!=0)]
    correlation<-abs(cor(X.tr,Y.tr))
    ranking<-sort(correlation,dec=T,index.return=T)$ix
    
    for (nb_features in 1:length(ranking)) {
      DS<-cbind(X.tr[,ranking[1:nb_features],drop=F],SalePrice=Y.tr)
      
      if(modelName == 'lm') {
        model<- lm(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      }
      if(modelName == 'rpart') {
        model<- rpart(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      }
      if(modelName == 'tree') {
        model<- tree(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      }
      if(modelName == 'svm'){
        # model<- svm(SalePrice~.,DS, scale = T, center=T, kernel = 'radial', 
        #             shrinking = T, cross = 10, cost = 4, epislon = 0.2)
        model<- svm(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      }
      if(modelName == 'nnet'){
        DS <- scale(DS)
        model<- nnet(SalePrice~.,DS, size =3, linout = TRUE, maxit=150, decay=0.3)
        Y.hat.ts <- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      }
      if(modelName == 'lazy'){
        model<- lazy(SalePrice~.,DS)
        Y.hat.ts<- predict(model,X.ts[,ranking[1:nb_features],drop=F])$h
      }
      if(modelName == 'ridge'){
        model<- linearRidge(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      }
      
      CV.err[nb_features,i]<-rmse(Y.hat.ts,Y.ts)
    }
  }  
  
  par(mfrow=c(1,1))
  plot(1:nrow(CV.err),apply(CV.err,1,mean), type = "o", main = paste(modelName ,' filter feature'), xlab = "number of features", ylab = 'cross validaton error' )
  
  writeLines(paste( modelName, " filter features: ",c(1:n)," ; CV error=",round(apply(CV.err,1,mean),digits=4), " ; std dev=",round(apply(CV.err,1,sd),digits=4)))
  ranking
}






mrmr <- function(modelName, X, Y) {
  n <- ncol(X)
  size.CV<-floor(N/10)
  
  CV.err<-matrix(0,nrow=n,ncol=10)
  
  for (i in 1:10) {
    i.ts<-(((i-1)*size.CV+1):(i*size.CV))  
    X.ts<-X[i.ts,]  
    Y.ts<-Y[i.ts]  
    
    i.tr<-setdiff(1:N,i.ts)
    X.tr<-X[i.tr,]
    Y.tr<-Y[i.tr]
    
    
    correlation<-abs(cor(X.tr,Y.tr))
    
    selected<-c()
    candidates<-1:n
    
    #mRMR ranks the variables by taking account not only the correlation with the output, but also by avoiding redudant variables
    for (j in 1:n) {
      redudancy.score<-numeric(length(candidates))
      if (length(selected)>0) {
        cor.selected.candidates<-cor(X.tr[,selected,drop=F],X.tr[,candidates,drop=F])
        redudancy.score<-apply(cor.selected.candidates,2,mean)
      }
      
      mRMR.score<-correlation[candidates]-redudancy.score
      #print(paste('redudancy.score : ', redudancy.score, '  correlation[candidates] : ', correlation[candidates]))
      selected_current<-candidates[which.max(mRMR.score)]
      selected<-c(selected,selected_current)
      candidates<-setdiff(candidates,selected_current)
      #print(paste(' mRMR.score: ', mRMR.score, ' selected_current : ', selected_current, ' selected :' , selected, ' candidates: ', candidates))
    }
    
    ranking<-selected
    
    for (nb_features in 1:n) {
      DS<-cbind(X.tr[,ranking[1:nb_features],drop=F],SalePrice=Y.tr)
      
      if(modelName == 'lm') {
        model<- lm(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      }
      if(modelName == 'rpart') {
        model<- rpart(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      }
      if(modelName == 'tree') {
        model<- tree(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      }
      if(modelName == 'svm'){
        # model<- svm(SalePrice~.,DS, scale = T, center=T, kernel = 'radial', 
        #             shrinking = T, cross = 10, cost = 4, epislon = 0.2)
        model<- svm(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      }
      if(modelName == 'nnet'){
        DS <- scale(DS)
        model<- nnet(SalePrice~.,DS, size =3, linout = TRUE, maxit=150, decay=0.3)
        Y.hat.ts <- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      }
      if(modelName == 'lazy'){
        model<- lazy(SalePrice~.,DS)
        Y.hat.ts<- predict(model,X.ts[,ranking[1:nb_features],drop=F])$h
      }
      if(modelName == 'ridge'){
        model<- linearRidge(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      }
      CV.err[nb_features,i]<-rmse(Y.hat.ts,Y.ts)
    }
  }  
  
  par(mfrow=c(1,1))
  plot(1:nrow(CV.err),apply(CV.err,1,mean), type = "o", main = paste(modelName ,' MRMR'), xlab = "number of features", ylab = 'cross validaton error' )
  
  writeLines(paste(modelName , "Features: ",c(1:n)," ; CV error=",round(apply(CV.err,1,mean),digits=4), " ; std dev=",round(apply(CV.err,1,sd),digits=4)))
  print(selected)
  selected
}




pca <- function(modelName, X, Y) {
  n <- ncol(X)
  size.CV<-floor(N/10)
  
  CV.err<-matrix(0,nrow=n,ncol=10)
  
  X_pca<-data.frame(prcomp(X,retx=T)$x)
  
  for (i in 1:10) {
    i.ts<-(((i-1)*size.CV+1):(i*size.CV))  
    X.ts<-X_pca[i.ts,]  
    Y.ts<-Y[i.ts]  
    
    i.tr<-setdiff(1:N,i.ts)
    X.tr<-X_pca[i.tr,]
    Y.tr<-Y[i.tr]
    
    for (nb_features in 1:n) {
      DS<-cbind(X.tr[,1:nb_features,drop=F],SalePrice=Y.tr)
      
      if(modelName == 'lm') {
        model<- lm(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,1:nb_features,drop=F])
      }
      if(modelName == 'rpart') {
        model<- rpart(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,1:nb_features,drop=F])
      }
      if(modelName == 'tree') {
        model<- tree(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,1:nb_features,drop=F])
      }
      if(modelName == 'svm'){
        # model<- svm(SalePrice~.,DS, scale = T, center=T, kernel = 'radial', 
        #             shrinking = T, cross = 10, cost = 4, epislon = 0.2)
        model<- svm(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,1:nb_features,drop=F])
      }
      if(modelName == 'nnet'){
        DS <- scale(DS)
        model<- nnet(SalePrice~.,DS, size =3, linout = TRUE, maxit=150, decay=0.3)
        Y.hat.ts <- predict(model,X.ts[,1:nb_features,drop=F])
      }
      if(modelName == 'lazy'){
        model<- lazy(SalePrice~.,DS)
        Y.hat.ts<- predict(model,X.ts[,1:nb_features,drop=F])$h
      }
      if(modelName == 'ridge'){
        model<- linearRidge(SalePrice~.,DS)
        Y.hat.ts <- predict(model,X.ts[,1:nb_features,drop=F])
      }
      CV.err[nb_features,i]<-rmse(Y.hat.ts,Y.ts)
    }
  }  
  par(mfrow=c(1,1))
  plot(1:nrow(CV.err),apply(CV.err,1,mean), type = "o", main = paste(modelName ,' PCA '), xlab = "number of features", ylab = 'cross validaton error' )
  
  writeLines(paste(modelName ," Features: ",c(1:n)," ; CV error=",round(apply(CV.err,1,mean),digits=4), " ; std dev=",round(apply(CV.err,1,sd),digits=4)))
  X_pca
}






forwardSelection <- function(modelName, X, Y) {
  n <- ncol(X)
  size.CV<-floor(N/10)
  
  selected<-NULL
  
  for (round in 1:n) { 
    candidates<-setdiff(1:n,selected)
    
    CV.err<-matrix(0,nrow=length(candidates),ncol=10)
    
    for (j in 1:length(candidates)) {
      features_to_include<-c(selected,candidates[j])
      
      for (i in 1:10) {
        i.ts<-(((i-1)*size.CV+1):(i*size.CV))  
        X.ts<-X[i.ts,features_to_include,drop=F]  
        Y.ts<-Y[i.ts]  
        
        i.tr<-setdiff(1:N,i.ts)
        X.tr<-X[i.tr,features_to_include,drop=F]
        Y.tr<-Y[i.tr]
        
        DS<-cbind(X.tr,SalePrice=Y.tr)
        
        if(modelName == 'lm') {
          model<- lm(SalePrice~.,DS)
          Y.hat.ts <- predict(model,X.ts)
        }
        if(modelName == 'rpart') {
          model<- rpart(SalePrice~.,DS)
          Y.hat.ts <- predict(model,X.ts)
        }
        if(modelName == 'tree') {
          model<- tree(SalePrice~.,DS)
          Y.hat.ts <- predict(model,X.ts)
        }
        if(modelName == 'svm'){
          # model<- svm(SalePrice~.,DS, scale = T, center=T, kernel = 'radial', 
          #             shrinking = T, cross = 10, cost = 4, epislon = 0.2)
          model<- svm(SalePrice~.,DS)
          Y.hat.ts <- predict(model,X.ts)
        }
        if(modelName == 'nnet'){
          DS <- scale(DS)
          model<- nnet(SalePrice~.,DS, size =3, linout = TRUE, maxit=150, decay=0.3)
          Y.hat.ts <- predict(model,X.ts)
        }
        if(modelName == 'lazy'){
          model<- lazy(SalePrice~.,DS)
          Y.hat.ts<- predict(model,X.ts)$h
        }
        if(modelName == 'ridge'){
          model<- linearRidge(SalePrice~.,DS)
          Y.hat.ts <- predict(model,X.ts)
        }
        
        CV.err[j,i]<-rmse(Y.hat.ts,Y.ts)
      }
    }
    CV.err.mean<-apply(CV.err,1,mean)
    CV.err.sd<-apply(CV.err,1,sd)
    selected_current<-which.min(CV.err.mean)              
    selected<-c(selected,candidates[selected_current])
    print(paste("Round ",round," ; Selected feature: ",candidates[selected_current]," ; CV error=",round(CV.err.mean[selected_current],digits=4), " ; std dev=",round(CV.err.sd[selected_current],digits=4)))
    
  }
  
  #print(paste('colnames(X)[selected] :', colnames(X)[selected]))
  #print(paste('colnames(X) ', colnames(X)))
  selected
}



runModel <- function(modelName, X, Y){
  
  size.CV<-floor(N/10)
  
  CV.err<-numeric(10)
  
  for (i in 1:10) {
    i.ts<-(((i-1)*size.CV+1):(i*size.CV))  
    X.ts<-X[i.ts,]  
    Y.ts<-Y[i.ts]  
    
    i.tr<-setdiff(1:N,i.ts)  
    X.tr<-X[i.tr,]
    Y.tr<-Y[i.tr]                          
    
    DS<-cbind(X.tr,SalePrice=Y.tr)
    
    
    if(modelName == 'lm') {
      model<- lm(SalePrice~.,DS)
      Y.hat.ts <- predict(model,X.ts)
    }
    if(modelName == 'rpart') {
      model<- rpart(SalePrice~.,DS)
      Y.hat.ts <- predict(model,X.ts)
    }
    if(modelName == 'tree') {
      model<- tree(SalePrice~.,DS)
      Y.hat.ts <- predict(model,X.ts)
    }
    if(modelName == 'svm'){
      # model<- svm(SalePrice~.,DS, scale = T, center=T, kernel = 'radial', 
      #             shrinking = T, cross = 10, cost = 4, epislon = 0.2)
      model<- svm(SalePrice~.,DS)
      Y.hat.ts <- predict(model,X.ts)
    }
    if(modelName == 'nnet'){
      DS <- scale(DS)
      model<- nnet(SalePrice~.,DS, size =3, linout = TRUE, maxit=150, decay=0.3)
      Y.hat.ts <- predict(model,X.ts)
    }
    if(modelName == 'lazy'){
      model<- lazy(SalePrice~.,DS)
      Y.hat.ts<- predict(model,X.ts)$h
    }
    if(modelName == 'ridge'){
      model<- linearRidge(SalePrice~.,DS)
      Y.hat.ts <- predict(model,X.ts)
    }
    CV.err[i]<-rmse(Y.hat.ts,Y.ts) 
  }
  print(paste(modelName, "  CV error=",round(mean(CV.err),digits=4), " ; std dev=",round(sd(CV.err),digits=4)))
}

runEnsemble <- function(modelName, X, Y){
  
  R<-5
  size.CV<-floor(N/10)
  
  CV.err<-numeric(10)
  
  for (i in 1:10) {
    i.ts<-(((i-1)*size.CV+1):(i*size.CV))  ### Complete the code. i.ts should be the indices of the tessefor the i-th fold
    X.ts<-X[i.ts,]  
    Y.ts<-Y[i.ts]  
    
    i.tr<-setdiff(1:N,i.ts)                ### Complete the code. i.tr should be the indices of the training sefor the i-th fold
    Y.hat.ts.R<-matrix(0,nrow=nrow(X.ts),ncol=R)
    
    for (r in 1:R) {
      i.tr.resample<-sample(i.tr,rep=T)
      X.tr<-X[i.tr.resample,]
      Y.tr<-Y[i.tr.resample]       
      
      DS<-cbind(X.tr,SalePrice=Y.tr)
      
      if(modelName == 'lm') {
        model<- lm(SalePrice~.,DS)
        Y.hat.ts.R[,r] <- predict(model,X.ts)
      }
      if(modelName == 'rpart') {
        model<- rpart(SalePrice~.,DS)
        Y.hat.ts.R[,r] <- predict(model,X.ts)
      }
      if(modelName == 'tree') {
        model<- tree(SalePrice~.,DS)
        Y.hat.ts.R[,r] <- predict(model,X.ts)
      }
      if(modelName == 'svm'){
        # model<- svm(SalePrice~.,DS, scale = T, center=T, kernel = 'radial', 
        #             shrinking = T, cross = 10, cost = 4, epislon = 0.2)
        model<- svm(SalePrice~.,DS)
        Y.hat.ts.R[,r] <- predict(model,X.ts)
      }
      if(modelName == 'nnet'){
        DS <- scale(DS)
        model<- nnet(SalePrice~.,DS, size =3, linout = TRUE, maxit=150, decay=0.3)
        Y.hat.ts.R[,r] <- predict(model,X.ts)
      }
      if(modelName == 'lazy'){
        model<- lazy(SalePrice~.,DS)
        Y.hat.ts.R[,r]<- predict(model,X.ts)$h
      }
      if(modelName == 'ridge'){
        model<- linearRidge(SalePrice~.,DS)
        Y.hat.ts.R[,r] <- predict(model,X.ts)
      }
      
    }
    
    Y.hat.ts<-apply(Y.hat.ts.R,1,mean)
    CV.err[i]<-rmse(Y.hat.ts,Y.ts)
  }
  
  
  print(paste('Ensemble ', modelName, "  CV error=",round(mean(CV.err),digits=4), " ; std dev=",round(sd(CV.err),digits=4)))
}


ensembleSimpleAverage <- function(models, X, Y){
  
  R<-5
  size.CV<-floor(N/10)
  
  CV.err<-numeric(10)
  
  for (i in 1:10) {
    i.ts<-(((i-1)*size.CV+1):(i*size.CV))  ### Complete the code. i.ts should be the indices of the tessefor the i-th fold
    X.ts<-X[i.ts,]  
    Y.ts<-Y[i.ts]  
    
    i.tr<-setdiff(1:N,i.ts)                ### Complete the code. i.tr should be the indices of the training sefor the i-th fold
    Y.hat.ts.R<-matrix(0,nrow=nrow(X.ts),ncol= R * length(models))
    
    for( mi in 1: length(models)){
      modelName = models[mi]
      for (r in 1:R) {
        index <- (mi -1 ) * R + r
        
        i.tr.resample<-sample(i.tr,rep=T)
        X.tr<-X[i.tr.resample,]
        Y.tr<-Y[i.tr.resample]       
        
        DS<-cbind(X.tr,SalePrice=Y.tr)
        
        if(modelName == 'lm') {
          model<- lm(SalePrice~.,DS)
          Y.hat.ts.R[, r] <- predict(model,X.ts)
        }
        if(modelName == 'rpart') {
          model<- rpart(SalePrice~.,DS)
          Y.hat.ts.R[,index] <- predict(model,X.ts)
        }
        if(modelName == 'tree') {
          model<- tree(SalePrice~.,DS)
          Y.hat.ts.R[,r] <- predict(model,X.ts)
        }
        if(modelName == 'svm'){
          # model<- svm(SalePrice~.,DS, scale = T, center=T, kernel = 'radial', 
          #             shrinking = T, cross = 10, cost = 4, epislon = 0.2)
          model<- svm(SalePrice~.,DS)
          Y.hat.ts.R[,r] <- predict(model,X.ts)
        }
        if(modelName == 'nnet'){
          DS <- scale(DS)
          model<- nnet(SalePrice~.,DS, size =3, linout = TRUE, maxit=150, decay=0.3)
          Y.hat.ts.R[,r] <- predict(model,X.ts)
        }
        if(modelName == 'lazy'){
          model<- lazy(SalePrice~.,DS)
          Y.hat.ts.R[,r]<- predict(model,X.ts)$h
        }
        
        if(modelName == 'ridge'){
          model<- linearRidge(SalePrice~.,DS)
          Y.hat.ts.R[,r] <- predict(model,X.ts)
        }
      }
    }
    
    Y.hat.ts<-apply(Y.hat.ts.R,1,mean)
    CV.err[i]<-rmse(Y.hat.ts,Y.ts)
  }
  
  
  print(paste('Ensemble ', modelName, "  CV error=",round(mean(CV.err),digits=4), " ; std dev=",round(sd(CV.err),digits=4)))
}

writePredictionToFile <- function (prediction) {
  predictedSalePrice <- exp(prediction) -1
  result <- cbind(Id = output$Id, SalePrice = predictedSalePrice )
  colnames(result) <- c("Id","SalePrice")
  write.csv(result, "submission.csv",row.names=FALSE)
}




corr.df = cbind(X, SalePrice = Y)
correlations <- abs(cor(corr.df))

# only want the columns that show strong correlations with SalePrice, bigger than 0.5
corr.SalePrice = as.matrix(sort(correlations[,'SalePrice'], decreasing = TRUE))
corr.idx = names(which(apply(corr.SalePrice, 1, function(x) (x > 0.2))))

par(mfrow=c(1,1))
corrplot(as.matrix(correlations[corr.idx,corr.idx]), type = 'upper', method='color', addCoef.col = 'green')

length(corr.idx) #we have 14 features with a significative correlation with SalePrice

#Let's remove the features uncorrelated to saleprice
data <- data[, which(apply(corr.SalePrice, 1, function(x) (x > 0.2)))]


Y<-log(input$SalePrice + 1)

X<- numeric.df[1:nrow(input),]
X<- data[1:nrow(input),]



N<-nrow(input)    #Number of examples
n<-ncol(X)    #Number of input variables



#######################################
##### Evaluate models
#######################################

############# RPART ##################
X<- data[1:nrow(input),]
X<- numeric.df[1:nrow(input),]

runModel('rpart', X, Y) #0.0142
outputCorrelation('rpart', X, Y) #rpart  filter features:  4  ; CV error= 0.0133  ; std dev= 0.0101
mrmr('rpart', X, Y) # rpart Features:  5  ; CV error= 0.0137  ; std dev= 0.0075
pca('rpart', X, Y) # rpart  Features:  2  ; CV error= 0.0179  ; std dev= 0.0148
forwardSelection('rpart', X, Y) #[1] "Round  2  ; Selected feature:  14  ; CV error= 0.0119  ; std dev= 0.0093"
runEnsemble('rpart', X, Y)# [1] "Ensemble  rpart   CV error= 0.015  ; std dev= 0.0077"
ensembleSimpleAverage(c('rpart', 'rpart'), X, Y )

#the winner is outputCorrelation
rpartIndexes <- outputCorrelation('rpart', X, Y)
rpartFeatures <- X[, rpartIndexes[1:4]]

runEnsemble('rpart', rpartFeatures, Y)

X<- data[1:nrow(input),]
#try on the whole dataset
runModel('rpart', X, Y) #0.0139
outputCorrelation('rpart', X, Y) #rpart  filter features:  24  ; CV error= 0.0129  ; std dev= 0.0115
mrmr('rpart', X, Y) # rpart Features:  5  ; CV error= 0.0129  ; std dev= 0.0097
pca('rpart', X, Y) # rpart  Features:  2  ; CV error= 0.0179  ; std dev= 0.0148
forwardSelection('rpart', X, Y) # "Round  32  ; Selected feature:  33  ; CV error= 0.0096  ; std dev= 0.0096"
runEnsemble('rpart', X, Y)#  CV error= 0.0162 

#best performance with forward selection
rpartIndexesTotal <- c(3, 14 , 1 , 4 , 7, 10, 11 ,13, 15, 20, 21, 22, 23 ,24 ,25, 26, 27 ,28 ,29 ,30, 31, 35, 36, 38, 39 ,40, 41, 42,  8 ,19, 17, 33 )
rpartFeaturesTotal <- X[, rpartIndexesTotal]

runEnsemble('rpart', rpartFeaturesTotal, Y)#CV error=  0.0106 



############# SVM ##################

X<- data[1:nrow(input),]
X<- numeric.df[1:nrow(input),]

runModel('svm', X, Y) #0.0155
outputCorrelation('svm', X, Y) #14  ; CV error= 0.0136  ; std dev= 0.0076
mrmr('svm', X, Y) # 7  ; CV error= 0.0128  ; std dev= 0.0064
pca('svm', X, Y) #svm  Features:  19  ; CV error= 0.0141  ; std dev= 0.0087
#forwardSelection('svm', X, Y) #Takes a lot of time  "Round  5  ; Selected feature:  15  ; CV error= 0.0066  ; std dev= 0.0055"
runEnsemble('svm', X, Y)# # 0.0148

#the winner is forward selection
svmIndexes <- c(3, 14,4 , 8 ,15)
svmFeatures <- X[, svmIndexes]
runEnsemble('svm', svmFeatures, Y) # 0.007

############# LAZY ##################

X<- numeric.df[1:nrow(input),] #doesn't work well with one hot encoded cat features

runModel('lazy', X, Y) #0.0403
outputCorrelation('lazy', X, Y) #lazy  filter features:  18  ; CV error= 0.0066  ; std dev= 0.0057
mrmr('lazy', X, Y) #lazy Features:  22  ; CV error= 0.0153  ; std dev= 0.0096
pca('lazy', X, Y) #lazy  Features:  2  ; CV error= 0.0181  ; std dev= 0.0175
forwardSelection('lazy', X, Y) # "Round  2  ; Selected feature:  13  ; CV error= 0.0102  ; std dev= 0.0081"
runEnsemble('lazy', X, Y)# CV error= 0.0876

#the winner is
lazyIndexes <- outputCorrelation('lazy', X, Y)
lazyFeatures <- X[, lazyIndexes[1:18]]

runEnsemble('lazy', lazyFeatures, Y)#CV error= 0.0323

############# TREE ##################
X<- numeric.df[1:nrow(input),] #Doesn't work on entire data set, undefined columns selected

runModel('tree', X, Y) #0.0142
outputCorrelation('tree', X, Y) #tree  filter features:  4  ; CV error= 0.0133  ; std dev= 0.0101
mrmr('tree', X, Y) #tree Features:  4  ; CV error= 0.0122  ; std dev= 0.0093
pca('tree', X, Y) #tree  Features:  14  ; CV error= 0.0171  ; std dev= 0.0118
forwardSelection('tree', X, Y) #
runEnsemble('tree', X, Y)# 

#the winner is



############# LM ##################
X<- data[1:nrow(input),]
X<- numeric.df[1:nrow(input),]

runModel('lm', X, Y) #0.0155
outputCorrelation('lm', X, Y) #
mrmr('lm', X, Y) #
pca('lm', X, Y) #
forwardSelection('lm', X, Y) #
runEnsemble('lm', X, Y)

#the winner is


############# NNET ##################
X<- data[1:nrow(input),]
X<- numeric.df[1:nrow(input),]

runModel('svm', X, Y) #0.0155
outputCorrelation('svm', X, Y) #
mrmr('svm', X, Y) # 
pca('svm', X, Y) #
forwardSelection('svm', X, Y) #
runEnsemble('svm', X, Y)# 

#the winner is



############# RIDGE ##################
X<- data[1:nrow(input),]
X<- numeric.df[1:nrow(input),]

runModel('ridge', X, Y) 
outputCorrelation('svm', X, Y) #
mrmr('svm', X, Y) #
pca('svm', X, Y) #
forwardSelection('svm', X, Y) #
runEnsemble('svm', X, Y)# 

#the winner is



#identify where we can't test on the the whole dataset
X<- data[1:nrow(input),]
X<- numeric.df[1:nrow(input),]

runModel('rpart', X, Y)#0.0139
runModel('svm', X, Y)#0.0147
runModel('lazy', X, Y)
runModel('tree', X, Y)
runModel('lm', X, Y) 
runModel('ridge', X, Y) # 0.0108
runModel('nnet', X, Y) #CV error= 9.6651

