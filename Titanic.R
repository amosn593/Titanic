# We are going to predict the survival rate in the titanic dataset found in kaggle website.
# Go to http://www.kaggle.com/c/titanic-gettingStarted/data to retrieve the data list.

# --------------------------------------------------------------------------------------
# READING IN THE DATA
# -----------------------------------------------------------------------------------------
train <- read.csv('train.csv',na.strings = c("NA",""))
test <- read.csv("test.csv",na.strings = c("NA",""))

# ADDING SURVIVED VARIABLE IN TEST DATA TO ENABLE COMBINING the dataset
test <- data.frame(Survived=rep("None",nrow(test)),test[,])

# COMBINING TRAIN AND TEST DATA
data <- rbind(train,test)

# ---------------------------------------------------------------------------------------
# Converting types on character Variables.
# -------------------------------------------------------------------------------------------
#LOOKING AT structure of the DATAset 
str(data)

data$Survived <- as.factor(data$Survived)
data$Pclass <- as.factor(data$Pclass)
data$SibSp <- as.factor(data$SibSp)
data$Parch <- as.factor(data$Parch)

# ----------------------------------------------------------------------------------------------
# Detecting Missing Values
# ------------------------------------------------------------------------------------------------
summary(data)



# ------------------------------------------------------------------------------------------------
# Imputing Missing Values
# -----------------------------------------------------------------------------------------------

# Embarked variable
table(data$Embarked,useNA = "always")
# there are only two missing values and we can replace them with mode.
data$Embarked[which(is.na(data$Embarked))]="S"

# Age Variable

# ----------------------------------------------------------------------------------------------
# Exploring and Visualizing data
# -----------------------------------------------------------------------------------------------


#TAKING A LONG AT SURVIVED VARIABLE
table(data[1:891,]$Survived)

library(ggplot2)

ggplot(data[1:891,],aes(as.factor(Survived)))+
  geom_bar(color="blue",fill="purple")+
  xlab("Survived")+
  ggtitle("Survived")

#DISTRIBUTION ACROSS CLASSES
table(data[1:891,]$Pclass)

#LOAD GGPLOT2
library(ggplot2)

#RICH FOLKS SURVIVED MORE
ggplot(data[1:891,],aes(x=as.factor(Pclass),fill=factor(Survived)))+
  geom_bar()+
  xlab("Pclass")+
  ylab("Total Count")+
  labs(fill="Survived")

#LOOKING AT NAME VARIABLE
head(as.character(data$Name))

#UNIQUE NAMES IN COMBINED DATA
any(duplicated(data$Name))
length(unique(data$Name))
# data[which(duplicated(data$Name))]
library(dplyr)
dp1 <- filter(data,duplicated(Name))

# There seems to be TWO DUPLICATE VALUES
dp <- as.character(data[which(duplicated(as.character(data$Name))),"Name"])
data[which(data$Name%in%dp1$Name),]
# Looking at the other attributes, it appears that they are only having same name 
# but they are different.

#creating extraction utility to extract titles from name variable
library(stringr)

extractTitle <- function(Name){
  Name <- as.character(Name)
  
  if(length(grep("Miss.",Name))>0){
    return("Miss.")
  }else if(length(grep("Mrs.",Name))>0){
    return("Mrs.")
  }else if (length(grep("Mr.",Name))>0){
    return("Mr.")
  }else if(length(grep("Master.",Name))>0){
    return("Master.")
  }else{
    return("Others.")
  }
}

Titles <- NULL
for(i in 1:nrow(data)){
  Titles <- c(Titles,extractTitle(data[i,"Name"]))
}
table(Titles)

# Adding Titles variable into the dataset.
data$Title <- as.factor(Titles)

ggplot(data[1:891,],aes(x=Title,fill=Survived))+
  geom_bar()+
  facet_wrap(~Pclass)+
  xlab("Title")+
  ylab("Total Count")+
  ggtitle("Titles")

##Distribution of Gender
table(train$Sex)

#Visualize 3-way relationship of sex,survived and pclass
ggplot(data[1:891,],aes(x=Sex,fill=Survived))+
  geom_bar()+
  facet_wrap(~Pclass)+
  xlab("Title")+
  ylab("Total Count")+
  labs(fill="Survived")
#Ok, age and sex seem to be important and predictive

##Looking at age variable
summary(data$Age)
min(data$Age,na.rm = TRUE)
ggplot(data[1:891,],aes(x=Age,fill=Survived))+
  geom_histogram(binwidth = 10)+
  facet_wrap(~Sex+Pclass)+
  xlab("Age")+
  ylab("Total Count")+
  labs(fill="Survived")

#Validate Master is a god proxy for young males
boys <- data[which(data$Title=="Master."),]
summary(boys$Age) 

#Looking at Miss
misses <- data[which(data$Title=="Miss."),]
summary(misses$Age)
ggplot(misses[misses$Survived!="None",],aes(x=Age,fill=Survived))+
  geom_histogram(binwidth = 5)+
  facet_wrap(~Pclass)+
  xlab("Age")+
  ylab("Total Count")+
  labs(fill="Survived")

# Looking at Sibsp variable
summary(data$SibSp)

#Visualize Title by sibsp,pclass
ggplot(data[1:891,],aes(x=SibSp,fill=Survived))+
  geom_bar(width = 0.5)+
  facet_wrap(~Pclass+Title)+
  xlab("Sibsp")+
  ylab("Total Count")+
  labs(fill="Survived")

##Looking at Parch variable
summary(data$Parch)

#Visualize Title by Parch,pclass
ggplot(data[1:891,],aes(x=Parch,fill=Survived))+
  geom_bar(width = 0.5)+
  facet_wrap(~Pclass+Title)+
  xlab("Parch")+
  ylab("Total Count")+
  labs(fill="Survived",Title="Pclass and Title By Parch")
#Feature engineering, creating family size
tsibsp <- c(train$SibSp,test$SibSp)
tparch <- c(train$Parch,test$Parch)
data$Family_Size <- as.factor(tsibsp+tparch+1)
summary(data$Family_Size)

ggplot(data[1:891,],aes(x=Family_Size,fill=Survived))+
  geom_bar(width = 0.5)+
  facet_wrap(~Pclass+Title)+
  ylim(0,300)+
  xlab("Family Size")+
  ggtitle("Pclass and Title By Family Size")
ylab("Total Count")+
  labs(fill="Survived")
##Rule out fare,cabin & Embark

#======================================
### Exploratory Modelling
#======================================
training <- data[1:891,]
#Spliting training data into train and validation dataset
library(caret)
tr <- createDataPartition(training$Survived,p=0.7,list = FALSE)
train1 <- training[tr,]
valid1 <- training[-tr,]



# Random Forest.
library(randomForest)
#Taking Pclass and Title

rf.train.1 <- data[1:891,c("Pclass","Title")]
lab1 <- as.factor(train$Survived)
set.seed(1234)
rf.1 <- randomForest(x=rf.train.1,y=lab1,importance = TRUE,ntree = 100)
rf.1
varImpPlot(rf.1)

#Including SibSp
rf.train.2 <- data[1:891,c("Pclass","Title","SibSp")]
set.seed(1234)
rf.2 <- randomForest(x=rf.train.2,y=lab1,importance = TRUE,ntree = 100)
rf.2
varImpPlot(rf.2)

#Including Parch
rf.train.3 <- data[1:891,c("Pclass","Title","Parch")]
set.seed(1234)
rf.3 <- randomForest(x=rf.train.3,y=lab1,importance = TRUE,ntree = 100)
rf.3
varImpPlot(rf.3)

#Including both Sibsp and Parch
rf.train.4 <- data[1:891,c("Pclass","Title","SibSp","Parch")]
set.seed(1234)
rf.4 <- randomForest(x=rf.train.4,y=lab1,importance = TRUE,ntree = 100)
rf.4
varImpPlot(rf.4)

#Including Family Size
rf.train.5 <- data[1:891,c("Pclass","Title","Family_Size")]
set.seed(1234)
rf.5 <- randomForest(x=rf.train.5,y=lab1,importance = TRUE,ntree = 100)
rf.5
varImpPlot(rf.5)


#Subset test data to make submission
test.submit <- data[892:1309,c("Pclass","Title","Family_Size")]

#MAke predictions
pred.t <- predict(rf.5,test.submit)
table(pred.t)

#Write a csv file for submission
submit.df <- data.frame(PassengerID=seq(892,1309,1),Survived =pred.t)
write.csv(submit.df,file = "RF_201805_1.csv",row.names = FALSE)

# Our score in Kaggle is 0.79904 athough randomforest had a score of 0.8349

#======================================================================
#Cross Validation
#======================================================================

library(doSNOW)

set.seed(1234)
library(caret)
cv.10.fold <- createMultiFolds(lab1,k=10,times = 10)
table(lab1[cv.10.fold[[33]]])

#Set up traincontrol as per above object
tr.c1 <- trainControl(method = "repeatedcv",number = 10,repeats = 10,
                      index = cv.10.fold)

#Set up multiple-Core training with dosnow package
library(parallel)
detectCores()
library(doSNOW)
# cl <- makeCluster(2,type = "SOCK")
# registerDoSNOW(cl)


##set seed
set.seed(1234)

rfc1 <- train(x = rf.train.5,y = lab1,method ='rf',tunelength =3,mtree =1000,
              trControl =tr.c1)

# Shutdown Cluster 
# stopCluster(cl)

# Check out results
rfc1

# rfc1 is pessimistic compared to rf.5
# Lets try 5-folds CV

set.seed(1234)
cv.5.fold <- createMultiFolds(lab1,k=5,times = 10)


##Set up traincontrol
tr.c2 <- trainControl(method = "repeatedcv",number = 5,repeats = 10,
                      index = cv.5.fold)

##set seed
set.seed(1234)

rfc2 <- train(x = rf.train.5,y = lab1,method ='rf',tunelength =3,mtree =1000,
              trControl =tr.c2)
rfc2

# 5- folds CV is not better, lets try 3-fold CV

set.seed(1234)
cv.3.fold <- createMultiFolds(lab1,k=3,times = 10)


##Set up traincontrol
tr.c3 <- trainControl(method = "repeatedcv",number = 3,repeats = 10,
                      index = cv.3.fold)

##set seed
set.seed(1234)

rfc3 <- train(x = rf.train.5,y = lab1,method ='rf',tunelength =3,mtree =1000,
              trControl =tr.c3)
rfc3
#=============================
#Exploratory modeling 2
#============================
library(rpart)
library(rpart.plot)

#create utility function
rpart.cv <- function(seed,training,label,ctr){
  set.seed(seed)
  rpart.cv <- train(x = training,y = label,method = "rpart",tuneLength =30,
                    trControl = ctr)
  return(rpart.cv)
  
}

# Grap feaures 
features <- c("Pclass","Title","Family_Size")
rpart.train.1 <- data[1:891,features]

# Run Model and check results
rpart.1 <- rpart.cv(1234,rpart.train.1,lab1,tr.c3)
rpart.1   

#Plot
prp(rpart.1$finalModel,type =1,extra = 1,under = TRUE)

# Both rpart and rf confirm Title is very important,lets investigate further
table(data$Title)

# Parse out last name and Title 
name.split <- str_split(data$Name,",")
name.split[1]
 
lastname <- sapply(name.split,"[",1)

# Add last name to data dataset 
data$Last.Name <- lastname

# Now for Titles
name.split <- str_split(sapply(name.split,"[",2)," ")
name.split[1]
titles <- sapply(name.split,"[",2)
unique(titles)

# Whats up with title the
data[titles=="the",]

# Lets re-match the titles correctly
titles[titles %in% c("Dona.","the","Lady")] <- "Lady."
titles[titles %in% c("Ms.","Mlle.")] <- "Miss."
titles[titles %in% c("Col.","Capt.","Major.")] <- "Officer."
titles [titles=="Mme."] <- "Mrs"
titles[titles %in% c("Don.","Jonkheer.")] <- "Sir."
table(titles)

# Add new titles to the data  
data$New.Title <- as.factor(titles)

# Visualize new titles
ggplot(data[1:891,],aes(New.Title,fill=Survived))+
  geom_bar()+
  facet_wrap(~Pclass)+
  ggtitle("New Title by Pclass")

# Collapse the title
index <- which(data$New.Title=="Lady.")
data$New.Title[index] <- "Mrs."
data$New.Title[data$New.Title=="Mrs"] <- "Mrs."
 
indexes <- which(data$New.Title=="Sir."|data$New.Title=="Officer."|
                   data$New.Title=="Rev."|data$New.Title=="Dr.")
data$New.Title[indexes] <- "Mr."
# Visualize new titles
ggplot(data[1:891,],aes(New.Title,fill=Survived))+
  geom_bar()+
  facet_wrap(~Pclass)+
  ggtitle("New Title by Pclass")
# Grap features
feature <- c("Pclass",'Family_Size',"New.Title")
rpart.train.2 <- data[1:891,feature]

# Run Cv and check out results
rpart.2 <- rpart.cv(1234,rpart.train.2,lab1,tr.c3)
rpart.2

# Plot  
prp(rpart.2$finalModel,type = 0,extra = 1,under = TRUE)

# Dive into 1 Class and Mr.
ind.1.mr. <- which(data$Pclass==1 & data$New.Title=="Mr.")
mr.data <- data[ind.1.mr.,]
summary(mr.data)

# One female
mr.data[mr.data$Sex=="female",]
  
# Update New.Title feature 
data$New.Title[data$Sex=="female"& data$New.Title=="Mr."] <- "Mrs."

# Any other gender mix up?
length(which(data$Sex=="female" &(data$New.Title=="Master" | data$New.Title=="Mr.")))
# Refresh 1 class and Mr.  
ind.1.mr. <- which(data$Pclass==1 & data$New.Title=="Mr.")
mr.data <- data[ind.1.mr.,]
summary(mr.data)
feature

# Subset test data and features  features
submit <- data[892:1309,feature]

# Make predictions
preds <- predict(rpart.2$finalModel,submit,type = "class")

# write a csv file 
submit.pred <- data.frame(PassengerId=rep(892:1309),Survived=preds)

# Make submission
write.csv(submit.pred,file = "Rpart.110118.csv",row.names = FALSE)
