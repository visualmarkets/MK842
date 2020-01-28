vizData <- trainData[1:1000, ]

dummyModel <- dummyVars(~ day_of_week, vizData)
dummyData <- predict(dummyModel, vizData)

test <- cbind(vizData, dummyData)

caret::featurePlot(
  x        = trainClean[1:1000,.(end_hour, pressure_in, precipitation_in)], 
  y        = vizData$severity,
  plot = "pairs"
)
