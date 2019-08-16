library(car)
library(e1071)
hearts <- Hearts_Dummy
traindata <- hearts[ which(hearts$Thal_normal==1), ]
testdata <- hearts[ which(hearts$Thal_normal==0), ]
traindata$Thal_normal <- NULL
testdata$Thal_normal <- NULL
traindata$Count <- NULL
testdata$Count <- NULL

scatterplotMatrix(~MaxHR+Oldpeak+Ca | Thal_normal, data=hearts)
svmfit <- svm(AHD_Yes~., data=traindata, kernel='radial', gamma=1, cost=1)
summary(svmfit)

set.seed(1)
tune.out = tune(svm, AHD_Yes~., data=traindata, kernel='radial', ranges = list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4)))
summary(tune.out)
plot(tune.out)

Prediction <- predict(svmfit, testdata)
Tab <- table(pred=prediction, true=testdata$AHD_Yes[1:135])
Tab
