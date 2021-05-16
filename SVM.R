library(ggplot2)
library(caret)
library(dplyr)
library(DMwR)
library(e1071)
library(factoextra)
library(pROC)

#Read the preprocessed file
new_df= read.csv("Imputed_data.csv",header=TRUE)
barplot(table(new_df$y))
dim(new_df)
str(new_df)

new_df = new_df[,-1]

#irrelevant & redundant vars can be handled by SVM , ignoring feature selection 

new_df$y<-as.factor(new_df$y)
table(new_df$y)


#check missing rows
sum(!complete.cases(new_df))


#create dummy vars
df1 <- dummyVars(~job+marital+education+housing+loan+contact+month+day_of_week+pdays+poutcome+age_group, data=new_df, sep="_",fullRank=TRUE)
df2 <- predict(df1,new_df)
# add back to original dataframe
new_df2 = data.frame(new_df[,-c(2:9,12,14,21)],df2)
#check structure, only y var is factor type
str(new_df2)

# testing sets using a 80/20 split rule
set.seed(110)
samp <- createDataPartition(new_df2$y, p=.80, list=FALSE)
train = new_df2[samp, ] 
test = new_df2[-samp, ]
#distribution of data before balancing
barplot((table(train$y)))

#Rebalancing data via SMOTE

#SMOTE
set.seed(110)
new_train<-SMOTE(y~.,train,perc.over = 100, perc.under=200)
barplot((table(new_train$y)))

#visualise data with PCA to figure out hyperplane, 
bank.pca<- prcomp(new_df2[,-10],center=TRUE,scale=TRUE)
fviz_pca_ind(bank.pca, geom.ind = "point", pointshape = 21, 
             pointsize = 2, 
             fill.ind = new_df2$y, col.ind = "black",
             repel = TRUE)
#No clear separation, however, on the left 1's (those who respond positively to the marketing) are more.
#SVM Kernel selection


#linear kernel
set.seed(110)
svm_linear_mod <- svm(y~., 
                      data=new_train, 
                      method="C-classification", 
                      kernel="linear", 
                      scale=TRUE)

#training performance

svm_linear<-predict(svm_linear_mod,new_train[,-10],type="class")
svm_linear_acc <- confusionMatrix(svm_linear,new_train$y,mode="prec_recall",positive = "1")

#radial kernel
set.seed(110)
svm_radial_mod <- svm(y~., 
                      data=new_train, 
                      method="C-classification", 
                      kernel="radial", 
                      scale=TRUE)

#training performance

svm_radial<-predict(svm_radial_mod,new_train[,-10],type="class")
svm_radial_acc <- confusionMatrix(svm_radial,new_train$y,mode="prec_recall",positive = "1")

#polynomial kernel

set.seed(110)
svm_poly_mod <- svm(y~., 
                    data=new_train, 
                    method="C-classification", 
                    kernel="polynomial", 
                    scale=TRUE)

#training performance

svm_polynomial<-predict(svm_poly_mod,new_train[,-10],type="class")
svm_polynomial_acc <- confusionMatrix(svm_polynomial,new_train$y,mode="prec_recall",positive = "1")


#sigmoid kernel
set.seed(110)
svm_sigmoid_mod <- svm(y~., 
                       data=new_train, 
                       method="C-classification", 
                       kernel="sigmoid", 
                       scale=TRUE)

#training performance

svm_sigmoid<-predict(svm_sigmoid_mod,new_train[,-10],type="class")
svm_sigmoid_acc <- confusionMatrix(svm_sigmoid,new_train$y,mode="prec_recall",positive = "1")

#Check accuracy for different kernels

rbind(Linear_SMOTE=svm_linear_acc$overall[1], 
      Radial_SMOTE=svm_radial_acc$overall[1],
      Polynomial_SMOTE=svm_polynomial_acc$overall[1], 
      Sigmoid_SMOTE=svm_sigmoid_acc$overall[1])


#Hyperparameter tuning
set.seed(110)
tPoly=tune(svm, y~., data=new_train, 
             tunecontrol=tune.control(sampling = "cross"), #default to 10 k cross validation
             kernel="polynomial", scale = TRUE,
             ranges = list(degree = 3:5, cost = 2^(-1:3)))
summary(tPoly)
tPoly$best.parameters


#Evaluate best model on training set

inpred <- predict(tPoly$best.model, new_train[, -10],type="class")
confusionMatrix(inpred, new_train$y, mode="prec_recall",positive="1")



#Apply model to test set

outpred <- predict(tPoly$best.model, test[,-10],type="class")
confusionMatrix(outpred, test$y, mode="prec_recall",positive="1")

# As above  model is overfitting, testing original model (Polynomial kernel with degree 3)
outpred_poly <- predict(svm_poly_mod, test[,-10],type="class")
confusionMatrix(outpred_poly, test$y, mode="everything",positive="1")
summary(svm_poly_mod)


#ROC Comparisons
pROC_obj <- roc(test$y,factor(outpred, 
                              ordered = TRUE),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)
pROC_obj2 <- roc(test$y,factor(outpred_poly, 
                              ordered = TRUE),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


#Plot ROC of both fitted curves
plot(pROC_obj, type = "n") # but don't actually plot the curve
# Add the line
lines(pROC_obj, type="b", pch=21, col="blue", bg="grey")
# Add the line of an other ROC curve
lines(pROC_obj2, type="o", pch=19, col="red")


outpred_complete <- predict(svm_poly_mod, new_df2[,-10],type="class")
confusionMatrix(outpred_complete, new_df2$y, mode="everything",positive="1")

pROC_obj_complete <- roc(new_df2$y,factor(outpred_complete, 
                              ordered = TRUE),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)