library(dummies)
library(ggplot2)
require(corrplot) # correlation plot
library(rpart)
library(e1071)
library(lazy)
library(nnet)

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


plotHist <- function(data_in, start, end) {
  for (i in start:end) {
    if(class(data_in[[i]]) == "character") {
      barplot(prop.table(table(data_in[[i]])), xlab = colnames(data_in)[i], main = paste('Barplot ' , i))
    }else{
      hist(data_in[[i]],freq=FALSE, xlab = colnames(data_in)[i], main = paste('histogram ' , i))
      lines(density(data_in[[i]]), col ='blue')
    }
  }
}

par(mfrow=c(2,3))



##########################################################
#Categoric variables imputation
##########################################################
plotHist(categoric.df, 1, ncol(categoric.df))

#By looking at the histograms for categorical data remove all feature where most values fall into 1 cat
one_dominant_feature <- c('Street',"LandContour","Utilities", "LandSlope", 'Condition1', 'Condition2', 'BldgType', 'RoofMatl',
                          'ExterCond', 'BsmtCond', 'BsmtFinType2', 'Heating', 'Electrical', 'Functional', 'GarageQual',
                          "GarageCond","PavedDrive" , "SaleType" , "SaleCondition")

categoric.df <- categoric.df[,setdiff(names(categoric.df), one_dominant_feature)]
plotHist(categoric.df, 1, ncol(categoric.df))

#Remove all features for which we don't have enough data
not_enough_data <- c('MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'GarageFinish', 'BsmtQual', 'BsmtExposure', 
                     'BsmtFinType1', 'MasVnrType', 'GarageType', 'CentralAir' , 'Alley')

categoric.df <- categoric.df[,setdiff(names(categoric.df), not_enough_data)]


colSums(is.na(categoric.df))

variable_to_keep<-c('MSZoning', 'ExterQual')
#Impute values
categoric.df[is.na(categoric.df$MSZoning), 'MSZoning'] <-names(sort(table(categoric.df$MSZoning), decreasing = TRUE)[1])
categoric.df[is.na(categoric.df$Exterior1st), 'Exterior1st'] <-names(sort(table(categoric.df$Exterior1st), decreasing = TRUE)[1])
categoric.df[is.na(categoric.df$Exterior2nd), 'Exterior2nd'] <-names(sort(table(categoric.df$Exterior2nd), decreasing = TRUE)[1])
categoric.df[is.na(categoric.df$KitchenQual), 'KitchenQual'] <-names(sort(table(categoric.df$KitchenQual), decreasing = TRUE)[1])


#Make sure there are no missing values
colSums(is.na(categoric.df))

dim(categoric.df)

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


#All data filled in!

#let's examine the output
plot(density(input$SalePrice), xlab = 'SalePrice', 'Distribution for sale price') #the distribution of sale prices is right-skewed and does not follow a gaussian
plot(density(log(input$SalePrice + 1)), xlab = 'SalePrice', 'Distribution for log(SalePrice +1)')


X<- numeric.df[1:nrow(input),]
Y<-log(input$SalePrice + 1)
N<-nrow(X)    #Number of examples
n<-ncol(X)    #Number of input variables

train<-cbind(X ,SalePrice=Y)
test <- numeric.df[(nrow(input) + 1):nrow(numeric.df),]
test <- cbind(numeric.df[(nrow(input) + 1):nrow(numeric.df),] , data_factor_onehot[(nrow(input) + 1):nrow(numeric.df),]) 
#test$Id <- output$Id #add back Id which we removed during training because it has to be written to file

#Plot dependencies between SalePrice and X
plotOutputDependency <- function(data_in, output, positions) {
  for (i in positions) {
    plot(data_in[[i]], output, xlab = colnames(data_in)[i], ylab = 'SalePrice', main = paste('Dependency ' , i))
  }
}

# plotDependency <- function(data_in, start, end) {
#   for (i in start:end) {
#     ggplot(data_in, aes(x = names(data_in)[i], y = SalePrice)) +
#       geom_point() + geom_smooth()
#   }
# }
# plotDependency(train, 1,29)

par(mfrow=c(2,3))
plotOutputDependency(X, Y, 1:29)



#Explore possible correlations between parameters
corrplot(cor(numeric.df,use="complete.obs"),type = 'upper', method='color')

ggplot(train, aes(x = GarageArea, y = GarageCars, color = SalePrice)) +
  geom_point() + geom_smooth()+ scale_fill_brewer(palette = "Spectral")

################################
#Run models
################################


runModel <- function(modelName, X, Y){
  if(modelName == 'nnet'){
    X <- scale(X)
  }
  
  size.CV<-floor(N/10)
  
  CV.err<-numeric(10)
  
  for (i in 1:10) {
    i.ts<-(((i-1)*size.CV+1):(i*size.CV))  ### Complete the code. i.ts should be the indices of the tessefor the i-th fold
    X.ts<-X[i.ts,]  
    Y.ts<-Y[i.ts]  
    
    i.tr<-setdiff(1:N,i.ts)                ### Complete the code. i.tr should be the indices of the training sefor the i-th fold
    X.tr<-X[i.tr,]
    Y.tr<-Y[i.tr]                          
    
    DS<-cbind(X.tr,sale_price=Y.tr)
    
    if(modelName == 'rpart') {
      model<- rpart(sale_price~.,DS)
      Y.hat.ts<- predict(model,X.ts)
    }
    if(modelName == 'svm'){
      model<- svm(sale_price~.,DS)
      Y.hat.ts<- predict(model,X.ts)
    }
    if(modelName == 'nnet'){
      model<- nnet(sale_price~.,DS, size =5, linout = TRUE)
      Y.hat.ts<- predict(model,X.ts)
    }
    if(modelName == 'lazy'){
      model<- lazy(sale_price~.,DS)
      Y.hat.ts<- predict(model,X.ts)$h
    }
    CV.err[i]<-mean((Y.hat.ts-Y.ts)^2)
  }
  print(paste(modelName, "  CV error=",round(mean(CV.err),digits=4), " ; std dev=",round(sd(CV.err),digits=4)))
}

runEnsemble <- function(modelName, X, Y){
  if(modelName == 'nnet'){
    X <- scale(X)
  }
  R<-20
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
      
      DS<-cbind(X.tr,sale_price=Y.tr)
      
      if(modelName == 'rpart') {
        model<- rpart(sale_price~.,DS)
        Y.hat.ts.R[,r]<- predict(model,X.ts)
      }
      if(modelName == 'svm'){
        model<- svm(sale_price~.,DS)
        Y.hat.ts.R[,r]<- predict(model,X.ts)
      }
      if(modelName == 'nnet'){
        model<- nnet(sale_price~.,DS, size =5, linout = TRUE)
        Y.hat.ts.R[,r]<- predict(model,X.ts)
      }
      if(modelName == 'lazy'){
        model<- lazy(sale_price~.,DS)
        Y.hat.ts.R[,r]<- predict(model,X.ts)$h
      }
      
    }
    
    Y.hat.ts<-apply(Y.hat.ts.R,1,mean)
    CV.err[i]<-mean((Y.hat.ts-Y.ts)^2)
  }
  
  
  print(paste('Ensemble ', modelName, "  CV error=",round(mean(CV.err),digits=4), " ; std dev=",round(sd(CV.err),digits=4)))
}


runModel('rpart', X, Y)
runEnsemble('rpart', X, Y)

runModel('lazy', X, Y)
runEnsemble('lazy', X, Y)


runModel('svm', X, Y)
runEnsemble('svm', X, Y)


runModel('knn', X, Y)
runEnsemble('knn', X, Y)

runModel('nnet', X, Y)
runEnsemble('nnet', X, Y)


####
featureSelection <- function(modelName, X, Y, minCorrelation){
  size.CV<-floor(N/10)
  
  CV.err<-matrix(0,nrow=n,ncol=10)
  print(dim(CV.err))
  numberOfFeatures <- n
  for (i in 1:10) {
    i.ts<-(((i-1)*size.CV+1):(i*size.CV))  
    X.ts<-X[i.ts,]  
    Y.ts<-Y[i.ts]  
    
    i.tr<-setdiff(1:N,i.ts)
    X.tr<-X[i.tr,]
    Y.tr<-Y[i.tr]
    
    correlation<-abs(cor(X.tr,Y.tr))
    correlation <- correlation[which(correlation[,1] > minCorrelation),]
    ranking<-sort(correlation,dec=T,index.return=T)$ix
    
    numberOfFeatures <- length(correlation)
    
    for (nb_features in 1:numberOfFeatures) {
      DS<-cbind(X.tr[,ranking[1:nb_features],drop=F],sale_price=Y.tr)
      #print(paste('Trying features ', names(DS)));
      model<- rpart(sale_price~.,DS)
      
      Y.hat.ts<- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      
      CV.err[nb_features,i]<-mean((Y.hat.ts-Y.ts)^2)
    }
  }  
  
  writeLines(paste("#Features: ",c(1:numberOfFeatures)," ; CV error=",round(apply(CV.err,1,mean),digits=4), " ; std dev=",round(apply(CV.err,1,sd),digits=4)))
}

featureSelection('rpart', X, Y, 0)

correlation1<-abs(cor(X,Y))
correlation1[1:n]

length(correlation1[which(correlation1[,1] > 0.5),])
dim(correlation1)
ranking<-sort(correlation1,dec=T,index.return=T)$ix
ranking
names(X)
######



mrmr <- function(X, Y) {
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
        #print(paste('cor.selected.candidates : ', cor.selected.candidates, ' redudancy.score ', redudancy.score))
      }
      
      mRMR.score<-correlation[candidates]-redudancy.score
      #print(paste('redudancy.score : ', redudancy.score, '  correlation[candidates] : ', correlation[candidates]))
      selected_current<-candidates[which.max(mRMR.score)]
      selected<-c(selected,selected_current)
      candidates<-setdiff(candidates,selected_current)
      #print(paste(' mRMR.score: ', mRMR.score, ' selected_current : ', selected_current, ' selected :' , selected, ' candidates: ', candidates))
    }
    
    ranking<-selected
    #print(ranking)
    
    for (nb_features in 1:n) {
      DS<-cbind(X.tr[,ranking[1:nb_features],drop=F],imdb_score=Y.tr)
      #print(names(DS))
      model<- rpart(imdb_score~.,DS)
      
      Y.hat.ts<- predict(model,X.ts[,ranking[1:nb_features],drop=F])
      
      CV.err[nb_features,i]<-mean((Y.hat.ts-Y.ts)^2)
    }
  }  
  
  writeLines(paste("#Features: ",c(1:n)," ; CV error=",round(apply(CV.err,1,mean),digits=4), " ; std dev=",round(apply(CV.err,1,sd),digits=4)))
}

mrmr(X,Y)

forwardSelection <- function(X, Y) {
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
        
        DS<-cbind(X.tr,imdb_score=Y.tr)
        model<- rpart(imdb_score~.,DS)
        
        Y.hat.ts<- predict(model,X.ts)
        
        CV.err[j,i]<-mean((Y.hat.ts-Y.ts)^2)
      }
    }
    CV.err.mean<-apply(CV.err,1,mean)
    CV.err.sd<-apply(CV.err,1,sd)
    selected_current<-which.min(CV.err.mean)              
    selected<-c(selected,candidates[selected_current])
    print(paste("Round ",round," ; Selected feature: ",candidates[selected_current]," ; CV error=",round(CV.err.mean[selected_current],digits=4), " ; std dev=",round(CV.err.sd[selected_current],digits=4)))
    
  }
  
  print(paste('colnames(X)[selected] :', colnames(X)[selected]))
  print(paste('colnames(X) ', colnames(X)))
}

forwardSelection(X,Y)


#Predict values and write to file
model<- rpart(SalePrice~.,train)
prediction<- predict(model,test)


writePredictionToFile <- function (prediction) {
  predictedSalePrice <- exp(prediction) -1
  result <- cbind(Id = output$Id, SalePrice = predictedSalePrice )
  colnames(result) <- c("Id","SalePrice")
  write.csv(result, "submission.csv",row.names=FALSE)
}

writePredictionToFile(prediction)

