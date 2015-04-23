library("caret")
library("ggplot2")
library(doParallel) #Thanks to Saif Shaikh for this speed up!

ProjectDir <- "~/R/Class8/project"
dataURLTrn <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
dataURLTst <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
dataFileTrn <- "pml-training.csv"
dataFileTst <- "pml-testing.csv"
now <- Sys.time()

#system(paste("mkdir -p ", ProjectDir))
setwd(ProjectDir)

#Check to see if the expected data files exist. Download if necessary. Then show the hash signature of the file used.
if (file.exists(dataFileTrn)){
    md5Trn <- system2("md5sum",args=c(dataFileTrn), stdout=TRUE)
    message(paste("Reusing the existing data file with MD5 hash:",md5Trn))        
} else {
    
    download.file(dataURLTrn, dataFileTrn, "wget", quiet = TRUE)
    message(paste("Got a fresh download of the data",now))
}

if (file.exists(dataFileTst)){
    md5Tst <- system2("md5sum",args=c(dataFileTst), stdout=TRUE)
    message(paste("Reusing the existing data file with MD5 hash:",md5Tst))        
} else {
    download.file(dataURLTst, dataFileTst, "wget", quiet = TRUE)
    message(paste("Got a fresh download of the data",now))
}

set.seed(97)

#If our variables are not in the workspace, load up the csv file. Otherwise carry on
if ( ! exists("dfTrn")) {
    dfTrn <- read.csv(dataFileTrn, header=TRUE, sep=",")
}
if ( ! exists("dfTst")) {
    dfTst <- read.csv(dataFileTst, header=TRUE, sep=",")
}

#PCA - Best accuracy I could get with a PCA analysis was about 42%. So I tried something simple. With just a "center and scale"
#preprocess, I hit a 77% accuracy. 
keepers <- sapply(dfTrn,is.numeric)  #Only the numeric columns
predictors <- c("roll_belt","pitch_belt","yaw_belt","total_accel_belt","gyros_belt_x","gyros_belt_y","gyros_belt_z",
           "accel_belt_x","accel_belt_y","accel_belt_z","magnet_belt_x","magnet_belt_y","magnet_belt_z","roll_arm","pitch_arm",
           "yaw_arm","total_accel_arm","gyros_arm_x","gyros_arm_y","gyros_arm_z","accel_arm_x","accel_arm_y","accel_arm_z",
           "magnet_arm_x","magnet_arm_y","magnet_arm_z","roll_dumbbell","pitch_dumbbell","yaw_dumbbell","total_accel_dumbbell",
           "gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z","accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z",
           "magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z","roll_forearm","pitch_forearm","yaw_forearm",
           "total_accel_forearm","gyros_forearm_x","gyros_forearm_y","gyros_forearm_z","accel_forearm_x","accel_forearm_y",
           "accel_forearm_z","magnet_forearm_x","magnet_forearm_y","magnet_forearm_z")
outcome <- "classe"
othervar <- "problem_id"

cl <- makeCluster(detectCores())
registerDoParallel(cl)


#Training
#With the default preProcessing giving me a 79% accuracy, I tried various training options.


#train with keepers and outcome, test without the outcome

tc <- trainControl(method="repeatedcv", allowParallel=TRUE, number=20, repeats=5) #99.5 percent with rf training
fit <- train(classe ~ ., trControl=tc, method="rpart", preProcess=c("center","scale"), data=dfTrn[,c(predictors,outcome)])
P1 <- predict(fit, dfTrn[,c(predictors,outcome)])
P2 <- predict(fit, newdata=dfTst[,c(predictors,othervar)])
confusionMatrix(fit)




# how you built your model (failed attempts)
# pproc1 <- preProcess(log10(abs(dfTrn[,keepers])+1)* sign(dfTrn[,keepers]), method="pca",  na.remove = TRUE, thresh=.80)
# trnPCA <- predict(pproc1, log10(abs(dfTrn[,keepers])+1)* sign(dfTrn[,keepers]))
# tc1 <- trainControl(method="repeatedcv", number=10, repeats=3)
# tc2 <- trainControl(method="repeatedcv", number=30, repeats=9)
# tc3 <- trainControl(method="boot",number=10, repeats=30)
# fit0 <- train(dfTrn$classe ~ ., method="rpart", data=trnPCA)
# fit1 <- train(dfTrn$classe ~ ., trControl=tc1, method="rpart", data=trnPCA)
# fit2 <- train(dfTrn$classe ~ ., trControl=tc2, method="rpart", data=trnPCA)
# fit3 <- train(dfTrn$classe ~ ., trControl=tc3, method="rpart", data=trnPCA)





# how you used cross validation
#tc1 <- trainControl(method="repeatedcv", number=10, repeats=3)  #83%
#tc3 <- trainControl(method="boot",number=10, repeats=3)         #71%
#tc4 <- trainControl(method="repeatedcv", number=30, repeats=3)  #91%
#tc5 <- trainControl(method="repeatedcv", number=10, repeats=9)  #83%
#tc6 <- trainControl(method="repeatedcv", number=15, repeats=15) #88%


# what you think the expected out of sample error is
# and why you made the choices you did.