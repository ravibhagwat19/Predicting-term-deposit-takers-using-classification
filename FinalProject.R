library(tidyverse)
library(data.table)
library(ggplot2)
library(VIM)
library(mice)
library(caret)
library(dplyr)
library(DMwR)
library(DMwR)


#Importing the dataset
bankdata = read.table('bank-additional-full.csv',sep=';',header = T)

#summary of data
dim(bankdata)
names(bankdata)
str(bankdata)
summary(bankdata)

##There are 41188 obseravtions with 21 variables in this dataset


### Data Pre-Processing ###

#Converting Unknown values to missing values
bankdata[bankdata=="unknown"] <- NA
bankdata$marital = droplevels(bankdata$marital)
bankdata$job = droplevels(bankdata$job)
bankdata$education = droplevels(bankdata$education)
bankdata$default = droplevels(bankdata$default)
bankdata$housing = droplevels(bankdata$housing)
bankdata$loan = droplevels(bankdata$loan)

#Removing default variable as it has 99.9% same observations
table(bankdata$default)
bankdata <- select(bankdata,-c(default))

summary(bankdata)

##Transforming Pdays variable
bankdata$pdays <- cut(bankdata$pdays, breaks= c(0,200,Inf), labels= c('priorContact','NoPriorContact'))
bankdata$pdays<-as.factor(bankdata$pdays)
table(bankdata$pdays)


#Divide age into different groups
for(i in 1 : nrow(bankdata)){
  if (bankdata$age[i] < 20){
    bankdata$age_group[i] = 'Teenagers'
  } else if (bankdata$age[i] < 35 & bankdata$age[i] > 19){
    bankdata$age_group[i] = 'Young Adults'
  } else if (bankdata$age[i] < 60 & bankdata$age[i] > 34){
    bankdata$age_group[i] = 'Adults'
  } else if (bankdata$age[i] > 59){
    bankdata$age_group[i] = 'Senior Citizens'
  }}
bankdata$age_group<-as.factor(bankdata$age_group)

#Replace target variables with yes with 1 and no with 0
bankdata$y<-ifelse(bankdata$y =='yes', 1,0)
bankdata$y<-as.factor(bankdata$y)
table(bankdata$y)

##36548 didn't respond to the campaign and 4640 responded

##Attribute DURATION highly affects the output target (e.g., if duration=0 then y='no').
##Yet, the duration is not known before a call is performed. Also, after the end of the call y is 
##obviously known. Thus, this input should only be included for benchmark purposes and should be 
##discarded if the intention is to have a realistic predictive model.


#Check for Duplicate Rows
sum(duplicated(bankdata))
bankdata = bankdata %>% distinct


#How Many Rows Are Completely Missing Values In All Columns
all.empty = rowSums(is.na(bankdata))==ncol(bankdata)
sum(all.empty)

#How Many Rows Contain Missing Data
sum(!complete.cases(bankdata))

#Missing Value By Variable
sapply(bankdata, function(x) sum(is.na(x)))
aggr_plot <- aggr(bankdata, col=c('green','blue'), numbers=TRUE, sortVars=TRUE, labels=names(df), 
                  cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))


### Exploratory Data Analysis ###

#Variable Frequency Bar charts
par(mfrow=c(2,2))
for(i in 1:length(bankdata))
{barplot(prop.table(table(bankdata[,i])),xlab=names(bankdata[i]), ylab= "Frequency (%)" , col = rainbow(3))}

#Relationship between job and education
(table(bankdata$job, bankdata$education))
ggplot(bankdata, aes(job,education)) + geom_count(color='red') + theme_classic() + 
  ggtitle('job by education frequency') + theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Relationship between job and Target variable (y)
(table(bankdata$job, bankdata$y))
ggplot(bankdata, aes(job,y)) + geom_count(color='red') + theme_classic() + ggtitle('job by subscription')
+ theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Age vs Marital status that subscribes term deposit
ggplot(bankdata, aes(x=age, fill=marital)) + geom_histogram(binwidth = 2, alpha=0.7) +facet_grid(cols = vars(y)) +
expand_limits(x=c(0,100)) +scale_x_continuous(breaks = seq(0,100,10)) +ggtitle("Age Distribution by Marital Status")

#Age_group and Target variable
ggplot (bankdata, aes(x=age_group)) + 
  geom_histogram(color = "blue", fill = "blue", binwidth = 5, stat = 'count') +
  facet_grid(cols=vars(y)) + 
  ggtitle('Age Distribution by Subscription') + ylab('Count') + xlab('Age')

#Subscription based on Number of Contact during the Campaign
ggplot(data=bankdata, aes(x=campaign, fill=y))+geom_histogram()+
ggtitle("Subscription based on Number of Contact during the Campaign")+
xlab("Number of Contact during the Campaign")+xlim(c(min=1,max=30)) +
guides(fill=guide_legend(title="Subscription of Term Deposit"))

bankdata %>% group_by(campaign) %>% summarize(contact.cnt = n(), pct.con.yes = mean(y=="1")*100) %>% 
arrange(desc(contact.cnt)) %>% head() 





##Missing Values Imputation

init=mice(bankdata,maxit=0)
meth = init$method
predM = init$predictorMatrix
meth[c("housing","loan")]="logreg" 
meth[c("job","marital","education")]="polyreg"
imp <- mice(bankdata, method = meth,  predictorMatrix=predM , m = 2) # Impute data
bank_clean <- complete(imp)
summary(bank_clean)
write.csv(bank_clean, "Imputed_data.csv")


### Model Building ###

#Sampling the data into train and test

partition <- createDataPartition(bank_clean$y, p=0.70, list=FALSE)
Train_data <- data2[ partition, ]
Test_data <- data2[ -partition, ]

## Dealing with imbalanced data ##
new_train<-SMOTE(y~.,train_data,perc.over = 100, perc.under=200)
prop.table(table(new_train$y))
