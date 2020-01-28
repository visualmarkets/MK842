# Call Libraries #
library(data.table)
library(magrittr)
library(future)
library(doFuture)
library(caret)
library(caretEnsemble)

# Execution Plan
registerDoFuture()
plan(multisession, workers = 1) # availableCores() - 1)

# Read in Data
# Training data
trainClean <- readRDS("cleanData.Rds")[["data"]][['trainClean']]
trainClean <- na.omit(trainClean) # Final sanatize before models

# Try ensemble
# sampleData <- sample(1:nrow(trainClean), size = 250000)
# trainClean <- trainClean[sampleData,]

# Testing data
testClean <- readRDS("cleanData.Rds")[["data"]][['testClean']]
testClean <- na.omit(testClean) # Final sanatize before models

# Establish training control variables
trControl <- trainControl(method = "cv",
                          # sampling = "down",
                          number = 1,
                          savePredictions = "final",
                          summaryFunction = twoClassSummary,
                          index = createResample(trainClean$severity, 3),
                          classProbs = TRUE)

# Turn tuning off for now
tuneLength <- 1

# Different Individual training models
xgBoostModel <- 
  train(
    x = trainClean[,.SD, .SDcols = -"severity"],
    y = trainClean$severity,
    method = "xgbTree",
    metric = "ROC",
    tuneLength = tuneLength, 
    trControl = trControl   
  )  

rfModel <-
  train(
    x = trainClean[,.SD, .SDcols = -"severity"],
    y = trainClean$severity,
    method = "rf",
    metric = "Kappa",
    tuneLength = tuneLength,
    trControl = trControl
  )

c5Model <-
  train(
    x = trainClean[,.SD, .SDcols = -"severity"],
    y = trainClean$severity,
    method = "C5.0",
    metric = "ROC",
    tuneLength = tuneLength,
    trControl = trControl   
  )

nnetModel <-
  train(
    x = trainClean[,.SD, .SDcols = -"severity"],
    y = trainClean$severity,
    method = "nnet",
    metric = "Kappa",
    tuneLength = tuneLength,
    trControl = trControl   
  )

rpartModel <-
  train(
    x = trainClean[,.SD, .SDcols = -"severity"],
    y = trainClean$severity,
    method = "rpart",
    metric = "Kappa",
    tuneLength = tuneLength,
    trControl = trControl   
  )

glmModel <-
  train(
    x = trainClean[,.SD, .SDcols = -"severity"],
    y = trainClean$severity,
    method = "glm",
    metric = "Kappa",
    tuneLength = tuneLength,
    trControl = trControl
  )

knnModel <- 
  train(
    x = trainClean[,.SD, .SDcols = -"severity"],
    y = trainClean$severity,
    method = "knn",
    metric = "Kappa",
    tuneLength = tuneLength,
    trControl = trControl
  )

# Prediction and confusion matrix
testCases <- predict(c5Model, testClean)
confusionMatrix(testCases, testClean$severity)

# Caret Ensemble Stack
modelList <- 
  caretList(
    x = trainClean[,.SD, .SDcols = -"severity"],
    y = trainClean$severity,
    metric = "Kappa",
    trControl = trControl,
    tunelength = 1,
    methodList = c("xgbTree", "C5.0", "glm", "glmnet")
  )

# Construct Ensemble Stack
greedyEnsemble <- 
  caretEnsemble(
    modelList, 
    metric = "Kappa",
    trControl = trainControl(
      number          = 4,
      method          = "cv",
      summaryFunction = twoClassSummary,
      savePredictions = "final",
      index = createResample(trainClean$severity, 3),
      classProbs      = TRUE
    )
  )

# Construct Model Stack
greedyStack <- 
  caretStack(
    modelList, 
    metric = "Kappa",
    trControl = trainControl(
      number          = 4,
      method          = "cv",
      summaryFunction = twoClassSummary,
      savePredictions = "final",
      index = createResample(trainClean$severity, 3),
      classProbs      = TRUE
    )
  )

# Summaries 
summary(greedyEnsemble)
summary(greedyStack)

# Model Correlation
modelCor(resamples(modelList))

# Run Ensembles with test data
testEnsemble <- predict(greedyEnsemble, testClean)
testStack    <- predict(greedyStack, testClean)

# Make Confusion matricies
confusionMatrix(testEnsemble, testClean$severity)
confusionMatrix(testStack, testClean$severity)
