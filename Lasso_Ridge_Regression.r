############################# problem1 ############################################################
library(readr)

# Load the Dataset
startups_data <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Lasso Ridge Regression\\50_Startups.csv")

#EDA
startups_data <- startups_data[-4]

# Reorder the variables
startups_data <- startups_data[,c(4,1,2,3)]

install.packages("glmnet")
library(glmnet)

#taking coloumns name
colnames(startups_data)

#passing input and output variables
x <- model.matrix(Profit ~ ., data = startups_data)
y <- startups_data$Profit

grid <- 10^seq(10, -2, length = 100)
grid

# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)


# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)

######################################### problem2 ###############################################
library(readr)

# Load the Data set 
computer_data <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Lasso Ridge Regression\\Computer_Data.csv")

#EDA
computer_data <- computer_data[-1]

install.packages("glmnet")
library(glmnet)

#taking coloumns name
colnames(computer_data)

#passing input and output variables
x <- model.matrix(price ~ ., data = computer_data)
y <- computer_data$price

grid <- 10^seq(10, -2, length = 100)
grid
# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)


# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)

################################ problem3 ##########################################################
library(readr)

# Load the Data set 
car_data <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Lasso Ridge Regression\\ToyotaCorolla.csv")

#EDA
car_data <- car_data[, -c(1,2,5,6,8,10,11,12,15,19:38)]

install.packages("glmnet")
library(glmnet)

#taking coloumns name
colnames(car_data)

#passing input and output variables
x <- model.matrix(Price ~ ., data = car_data)
y <- car_data$Price

grid <- 10^seq(10, -2, length = 100)
grid
# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)


# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)

###################################### problem4 ######################################################
library(readr)

# Load the Data set 
life_expectencey_data <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Lasso Ridge Regression\\Life_expectencey_LR.csv")

#EDA
life_expectencey_data <- life_expectencey_data[, -c(1,3)]

#Omitting NA values
sum(is.na(life_expectencey_data))
life_expectencey_data <- na.omit(life_expectencey_data)

install.packages("glmnet")
library(glmnet)

#taking coloumns name
colnames(life_expectencey_data)

#passing input and output variables
x <- model.matrix(Life_expectancy ~ ., data = life_expectencey_data)
y <- life_expectencey_data$Life_expectancy

grid <- 10^seq(10, -2, length = 100)
grid
# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)


# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)

########################################### END #####################################################