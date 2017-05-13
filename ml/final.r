library(dummies)
library(ggplot2)
require(corrplot) # correlation plot
library(rpart)
library(e1071)
library(lazy)
library(nnet)
library(tree)

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



filterFeatures <- function(modelName, X, Y){
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
      
      CV.err[nb_features,i]<-rmse(Y.hat.ts,Y.ts)
    }
  }  
  
  par(mfrow=c(1,1))
  plot(1:nrow(CV.err),apply(CV.err,1,mean), type = "o", main = paste(modelName ,' selection'), xlab = "number of features", ylab = 'cross validaton error' )
  
  writeLines(paste( modelName, " filter features: ",c(1:n)," ; CV error=",round(apply(CV.err,1,mean),digits=4), " ; std dev=",round(apply(CV.err,1,sd),digits=4)))
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




filterFeatures('rpart', X, Y)
filterFeatures('svm', X, Y)
filterFeatures('lazy', X, Y)
filterFeatures('tree', X, Y)
filterFeatures('lm', X, Y)



mrmr('rpart', X, Y)
mrmr('svm', X, Y)
mrmr('lazy', X, Y)
mrmr('tree', X, Y)
mrmr('lm', X, Y)




pca('rpart', X, Y)
pca('svm', X, Y)
pca('lazy', X, Y)
pca('tree', X, Y)
pca('lm', X, Y)




forwardSelection('rpart', X, Y)
forwardSelection('svm', X, Y)
forwardSelection('lazy', X, Y)
forwardSelection('tree', X, Y)
forwardSelection('lm', X, Y)


ensembleSimpleAverage(c('rpart', 'rpart'), X, Y )
runModel('rpart', X, Y)
runEnsemble('rpart', X, Y)
runEnsemble('lm', X, Y)


runModel('lazy', X, Y)
runEnsemble('lazy', X, Y)


runModel('svm', X, Y)
runEnsemble('svm', X, Y)


# runModel('knn', X, Y)
# runEnsemble('knn', X, Y)

runModel('nnet', X, Y)
runEnsemble('nnet', X, Y)
