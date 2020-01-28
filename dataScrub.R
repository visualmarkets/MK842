# Call Libraries #
library(readr)
library(lubridate)
library(data.table)
library(magrittr)
library(future)
library(doFuture)
library(caret)
library(naniar)
library(skimr)

# Execution Plan #
registerDoFuture()
plan(multisession, workers = availableCores() - 1)

# Source Data Schema #
source("schema.R")

# Read in Data with Schema #
rawData <- 
  read_csv(
    file = "US_Accidents_May19.csv.zip", 
    col_types = dataSchema) %>% 
  janitor::clean_names(case = "snake") %>% 
  as.data.table() # Convert to Data.Table

# Case when for severity.
rawData[as.numeric(severity) <= 4, severityHolder := "Low"]
rawData[as.numeric(severity) > 4, severityHolder := "High"]

# Add Features #
rawData[,
        `:=`(
          severity         = factor(severityHolder),
          severityHolder   = NULL,
          day_of_week      = weekdays(start_time),
          start_hour       = lubridate::hour(start_time),
          end_hour         = lubridate::hour(end_time),
          duration         = as.numeric(end_time - start_time)
        )]

rawData[,.N, by = severity]

dataSample <- sample(1:nrow(rawData), size = 200000)
rawData <- rawData[dataSample,]

# Make Train and Test Vector #
partition <- createDataPartition(rawData[,severity], p = 0.75, list = FALSE)

# Split Dataset into train and test #
trainData <- rawData[ partition,]
testData  <- rawData[-partition,]

# Build cleaning model #
preProcessModel <- 
  preProcess(
    trainData, 
    method = c("center", "scale", "knnImpute")
  )

# Apply cleaning model #
trainClean <- predict(preProcessModel, trainData) # Apply cleaning model to train dataset
testClean  <- predict(preProcessModel, testData)  # Apply cleaning model from training set to test set

omittedData <- na.omit(trainClean, invert = TRUE)

# View Missings
vizSample    <- sample(1:nrow(trainData), size = 10000)
visMiss      <- vis_miss(trainData[vizSample,])     # Create Viz of missing data
visMissClean <- vis_miss(trainClean[vizSample,])
ggMissUpset  <- gg_miss_upset(trainData[vizSample,]) # Create viz of missing data relationships

# View Results #
dataViewRaw         <- skimr::skim(trainData)          # View summary of raw data
dataViewClean       <- skimr::skim(trainClean)         # View summary of centered and scaled data
dataViewCleanNaOmit <- skimr::skim(na.omit(trainData)) # View summary of totally cleaned data

# Save clean data to file for others #
saveRDS(
  object = 
    list(
      data = list(
        rawData     = rawData,
        trainClean  = trainClean,
        testClean   = testClean,
        omittedData = omittedData
      ),
      summaries = list(
        dataViewRaw         = dataViewRaw, 
        dataViewClean       = dataViewClean,
        dataViewCleanNaOmit = dataViewCleanNaOmit
      )
    ),
  file = "cleanData.Rds"
)
