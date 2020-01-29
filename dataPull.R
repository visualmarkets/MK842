# Call Libraries
library(readr)
library(lubridate)
library(data.table)
library(magrittr)
library(future)
library(doFuture)
library(caret)
library(naniar)
library(skimr)

# Execution Plan
registerDoFuture()
plan(multisession, workers = availableCores() - 1)

# Source Data Schema
source("schema.R")

# Read in Data with Schema
rawData <- 
  read_csv(
    file = "US_Accidents_Dec19.csv", 
    col_types = dataSchema) %>% 
  janitor::clean_names(case = "snake") %>% 
  as.data.table() # Convert to Data.Table

# Case when for severity.
rawData[as.numeric(severity) <= 4, severityHolder := "Low"]
rawData[as.numeric(severity) > 4, severityHolder := "High"]

rawData[,.N, by = severityHolder]

# Add Features
rawData[,
        `:=`(
          severity = factor(severityHolder),
          severityHolder = NULL,
          day_of_week = weekdays(start_time),
          start_hour = lubridate::hour(start_time),
          end_hour   = lubridate::hour(end_time),
          precipitation_in = NULL
        )]

# Make Train and Test Vector
partition <- createDataPartition(rawData[,severity], p = 0.75, list = FALSE)

# Split Dataset into train and test
trainData <- rawData[ partition,]
testData  <- rawData[-partition,]

# Build cleaning model
preProcessModel <- 
  preProcess(
    trainData, 
    method = c("zv", "YeoJohnson")
  )

trainData <- predict(preProcessModel, trainData)

# Select Dummy Vars
dummyVars <- ~ weather_condition + side + state + roundabout + railway + amenity + give_way + day_of_week + junction + bump + turning_loop + traffic_signal + traffic_calming + sunrise_sunset +  wind_direction + crossing + no_exit + stop

# Model for creating dummy vars
dummyModel <- dummyVars(dummyVars, trainData, fullRank = TRUE) # Make model for dummy vars
dummyData  <- predict(dummyModel, trainData)                    # Make dummy data

dummyVars <- formula.tools::get.vars(dummyVars) # Transform from model spec into character vector

# Add in Dummy Variables and get rid of old variables
trainData <- cbind(trainData, dummyData)[,.SD, .SDcols = -c(dummyVars, "distance_mi", "start_time", "end_time", "zipcode", "end_hour", "station")]

trainClean <- DMwR::SMOTE(severity ~ ., data  = trainData[1:100000,]) # Run SMOTE to rebalance the classes between "High" and "Low" 
# trainData <- downSample(y = trainData$severity, x = trainData[,.SD, .SDcols = -c("severity")], list = FALSE, yname = "severity")

#  Make Test Dummy Vars
testData  <- predict(preProcessModel, testData)  # Apply cleaning model from training set to test set
testDummy <- predict(dummyModel, testData)
testData  <- cbind(testData, testDummy)[,.SD, .SDcols = -c(dummyVars, "distance_mi", "start_time", "end_time", "zipcode", "end_hour", "station")] # Bind dummy variables and remove originals

# View Missings
# vizSample    <- sample(1:nrow(trainData), size = 5000)
# visMiss      <- vis_miss(trainData[vizSample,])     # Create Viz of missing data
# visMissClean <- vis_miss(trainClean[vizSample,])
# ggMissUpset  <- gg_miss_upset(trainData[vizSample,]) # Create viz of missing data relationships

# View Results #
# dataViewRaw         <- skimr::skim(trainData)          # View summary of raw data
# dataViewClean       <- skimr::skim(trainClean)         # View summary of centered and scaled data
# dataViewCleanNaOmit <- skimr::skim(na.omit(trainData)) # View summary of totally cleaned data

# Save clean data to file for others
saveRDS(
  object = 
    list(
      data = list(
        rawData    = rawData,
        trainClean = trainData,
        testClean  = testData
      )
    ),
  file = "cleanData.Rds"
)
