library("caret")
library("ggplot2")

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
    md5Trn <- system(paste("md5sum", dataFileTrn))
    message(paste("Reusing the existing data file with MD5 hash:",md5Trn))        
} else {
    
    download.file(dataURLTrn, dataFileTrn, "wget", quiet = TRUE)
    message(paste("Got a fresh download of the data",now))
}

if (file.exists(dataFileTst)){
    md5Tst <- system(paste("md5sum", dataFileTst))
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

#PCA
keepers <- sapply(dfTrn,is.numeric)  #Only the numeric columns
pproc <- preProcess(log10(abs(dfTrn[,keepers])+1)* sign(dfTrn[,keepers]), method="pca",  na.remove = FALSE, thresh=.80)
trnPCA <- predict(pproc, log10(abs(dfTrn[,keepers])+1)* sign(dfTrn[,keepers]))
fit <- train(dfTrn$classe ~ ., method="rpart", data=trnPCA)
fin <- fit$finalModel

# how you built your model
# how you used cross validation
# what you think the expected out of sample error is
# and why you made the choices you did.