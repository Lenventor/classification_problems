rm(list = ls())  # remove the existing environment


setwd()

## Load utils.R and discriminant_analysis.R

source("utils.R")
source("discriminant_analysis.R")


## Load the training and test data
train <- Load_data("./data/digits_train.txt")
test <- Load_data("./data/digits_test.txt")

x_train <- train$x
y_train <- train$y

x_test <- test$x
y_test <- test$y

### Visualization 
## uncomment the following command to visualize the first ten digits in the training data. 
Plot_digits(1:10, x_test)



start_time = Sys.time()
priors <- Comp_priors(y_train,10)
means <- Comp_cond_means(x_train,y_train,10)
covs <- Comp_cond_covs(x_train,y_train,10,"LDA")

posterior = Predict_posterior(x_test,priors,means,covs,"LDA")
labels = Predict_labels(posterior)
end_time = Sys.time()
true_labels = labels - 1
errors = y_test-true_labels
error_rate = length(which(errors!=0))/length(y_test)
end_time-start_time
error_rate
#Error rate: 10.23% in 15.28s

start_time = Sys.time()
priors = Comp_priors(y_train,10)
means <- Comp_cond_means(x_train,y_train,10)
covs <- Comp_cond_covs(x_train,y_train,10,"QDA")

posterior = Predict_posterior(x_test,priors,means,covs,"QDA")
labels = Predict_labels(posterior)
end_time = Sys.time()
true_labels = labels - 1
errors = y_test-true_labels
error_rate = length(which(errors!=0))/length(y_test)
end_time-start_time
error_rate
#error_rate: 4.08% in 15.47s

start_time = Sys.time()
priors = Comp_priors(y_train,10)
means <- Comp_cond_means(x_train,y_train,10)
covs <- Comp_cond_covs(x_train,y_train,10,"NB")

posterior = Predict_posterior(x_test,priors,means,covs,"NB")
labels = Predict_labels(posterior)
end_time = Sys.time()
true_labels = labels - 1
errors = y_test-true_labels
error_rate = length(which(errors!=0))/length(y_test)
end_time-start_time
error_rate
#Error Rate: 17.03% in 15.41s

library(MASS)
start_time = Sys.time()
lda_fit = lda(y_test~x_test)
lda_pred = predict(lda_fit, newdata = test)
end_time = Sys.time()
mean(lda_pred$class!=y_test)
end_time-start_time
#error rate 9.45% in 0.1051s

start_time = Sys.time()
qda_fit = qda(y_test~x_test)
qda_pred = predict(qda_fit, newdata = test)
end_time = Sys.time()
mean(qda_pred$class!=y_test)
end_time-start_time
#Error rate 1% in 0.1415s

