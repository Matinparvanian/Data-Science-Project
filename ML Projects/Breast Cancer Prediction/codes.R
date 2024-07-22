rm(list = ls())
ls()
################################
library(ggplot2)
library(dplyr)
library(reshape2)
library(MASS)
library(car)
library(glmnet)
library(pROC)
library(e1071)
library(class) 
library(gridExtra)
################################

setwd("E:\\2023-2024A\\Statistical Learning\\data")
Cancer = read.csv("breast-cancer.csv")
str(Cancer)
dim(Cancer) 
attach(Cancer)
#################################

sum(is.na(Cancer))


missing_counts = data.frame(
  Variable = names(Cancer),
  Count = colSums(!is.na(Cancer)),
  Percentage = colMeans(!is.na(Cancer)) * 100
)

ggplot(missing_counts, aes(x = Variable, y = Count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Number and Percentage of Non-Missing Values",
       x = "Variable",
       y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

##############################################################

Cancer = Cancer[-1]
#############################

Cancer %>%
  group_by(diagnosis) %>%
  summarise(n = n()) %>%
  mutate(Percentage = round(n/sum(n)*100, 1)) %>%
  ggplot(aes(x="", y=n, fill = factor(diagnosis))) +
  geom_bar(width = 1, color = "white", alpha = 0.5, stat = "identity") +
  coord_polar("y", start=0) +
  labs(fill = "diagnosis", x = "", y = "") +
  theme_void() +
  geom_text(aes(y = n/1.3, label = paste0(Percentage, "%")), color = "white", size = 4) +
  scale_fill_manual(values = c('lightskyblue', 'gold'), labels = c("benign", "malignant"))


Cancer %>%
  group_by(diagnosis) %>%
  summarise(n = n()) %>%
  mutate(number = n) %>%
  ggplot(aes(x = factor(diagnosis), y = n, fill = factor(diagnosis))) +
  geom_bar(stat = "identity", color = "white", alpha = 0.5) +
  labs(x = "Count", y = "", fill = "diagnosis") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  geom_text(aes(label = paste0(number)), position = position_stack(vjust = 0.5), color = "black", size = 4) +
  scale_fill_manual(values = c('lightskyblue', 'gold'), labels = c("benign", "malignant")) +
  coord_flip()

############################################################################################################################

variables = names(Cancer)[2:11] 
plots = list() 
for (i in 1:length(variables)) {
  variable = variables[i]
  
  p = ggplot(Cancer, aes(x = .data[[variable]], fill = factor(diagnosis))) +
    geom_density(alpha = 0.5) +
    scale_fill_manual(values = c("lightskyblue", "gold"), name = "diagnosis") +
    labs(x = variable) +
    theme_classic()
  
  plots[[i]] = p  
}
grid.arrange(grobs = plots, ncol = 2)

variables = names(Cancer)[12:21] 
plots = list() 
for (i in 1:length(variables)) {
  variable <- variables[i]
  
  p = ggplot(Cancer, aes(x = .data[[variable]], fill = factor(diagnosis))) +
    geom_density(alpha = 0.5) +
    scale_fill_manual(values = c("lightskyblue", "gold"), name = "diagnosis") +
    labs(x = variable) +
    theme_classic()
  
  plots[[i]] = p  
}
grid.arrange(grobs = plots, ncol = 2)


variables = names(Cancer)[22:31] 
plots = list() 
for (i in 1:length(variables)) {
  variable <- variables[i]
  
  p = ggplot(Cancer, aes(x = .data[[variable]], fill = factor(diagnosis))) +
    geom_density(alpha = 0.5) +
    scale_fill_manual(values = c("lightskyblue", "gold"), name = "diagnosis") +
    labs(x = variable) +
    theme_classic()
  
  plots[[i]] = p  
}
grid.arrange(grobs = plots, ncol = 2)
################################################################################
Cancer_normalized = Cancer
Cancer_normalized[names(Cancer)[-1]] = scale(Cancer_normalized[names(Cancer)[-1]])
head(Cancer)
################################################################################
data_melted = melt(Cancer_normalized, id.vars = "diagnosis", variable.name = "features", value.name = "value")

# Create the boxplot
ggplot(data_melted, aes(x = features, y = value, fill = diagnosis)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "features", y = "value") +
  guides(fill = guide_legend(title = "diagnosis"))

######################################################################
Cancer$diagnosis[Cancer$diagnosis=='M']=1
Cancer$diagnosis[Cancer$diagnosis=='B']=0
attach(Cancer)
table(diagnosis)
###########################################
Cancer$diagnosis = as.numeric(as.character(Cancer$diagnosis))
set.seed(123)
test_index = sample(nrow(Cancer), 0.2 * nrow(Cancer))
train = Cancer[-test_index, ]
test = Cancer[test_index, ]

# Fit a logistic regression model
model = glm(diagnosis ~ ., data = train, family = "binomial")

# Function to calculate performance metrics
calculate_metrics = function(predictions, actual) {
  confusion_matrix = table(predictions, actual)
  true_positive = confusion_matrix[2, 2]
  false_positive = confusion_matrix[1, 2]
  false_negative = confusion_matrix[2, 1]
  true_negative = confusion_matrix[1, 1]
  
  recall = true_positive / (true_positive + false_negative)
  precision = true_positive / (true_positive + false_positive)
  f1_score = 2 * precision * recall / (precision + recall)
  accuracy = (true_positive + true_negative) / sum(confusion_matrix)
  
  return(data.frame(
    Metric = c("Recall", "Precision", "F1 Score", "Accuracy"),
    Value = c(recall, precision, f1_score, accuracy)
  ))
}

thresholds = c(0.3, 0.4, 0.5, 0.6)
results_train = data.frame()
results_test = data.frame()

for (threshold in thresholds) {
  # Train set performance
  pred_class_train = ifelse(predict(model, newdata = train, type = "response") > threshold, 1, 0)
  results_train_threshold = calculate_metrics(pred_class_train, train$diagnosis)
  results_train_threshold$Set = "Train"
  results_train_threshold$Threshold = threshold
  results_train = rbind(results_train, results_train_threshold)
  
  # Test set performance
  pred_class_test = ifelse(predict(model, newdata = test, type = "response") > threshold, 1, 0)
  results_test_threshold = calculate_metrics(pred_class_test, test$diagnosis)
  results_test_threshold$Set = "Test"
  results_test_threshold$Threshold = threshold
  results_test = rbind(results_test, results_test_threshold)
}

# Output performance metrics
print(results_train)
print(results_test)
table(pred_class_test, test$diagnosis)
roc_obj = roc(test$diagnosis,pred_class_test)
plot(roc_obj, main = "ROC Curve", print.auc = TRUE,lty = 2 , col = 'gold')



# Perform backward selection
backward_model = step(model, direction = "backward")

backward_pred_class_train = ifelse(predict(backward_model, newdata = train, type = "response") > 0.5, 1, 0)
backward_results_train = calculate_metrics(backward_pred_class_train, train$diagnosis)
backward_results_train$Set = "Train"

  
backward_pred_class_test = ifelse(predict(backward_model, newdata = test, type = "response") > 0.5, 1, 0)
backward_results_test = calculate_metrics(backward_pred_class_test, test$diagnosis)
backward_results_test$Set = "Test"



# Output performance metrics
print(backward_results_train)
print(backward_results_test)
table(backward_pred_class_test, test$diagnosis)
roc_obj = roc(test$diagnosis, backward_pred_class_test)
plot(roc_obj, main = "ROC Curve", print.auc = TRUE,lty = 2 , col = 'gold')
#################################################################################################################
# Prepare the data for ridge regression
x_train = model.matrix(diagnosis ~ ., data = train)[,-1]
y_train = train$diagnosis

x_test = model.matrix(diagnosis ~ ., data = test)[,-1]
y_test = test$diagnosis

# Perform ridge regression with cross-validation
cv_model = cv.glmnet(x_train, y_train, family = "binomial", alpha = 0)

# Get the best lambda value from cross-validation
best_lambda = cv_model$lambda.min

# Fit ridge regression model with the best lambda
ridge_model = glmnet(x_train, y_train, family = "binomial", alpha = 0, lambda = best_lambda)


# Train set performance
ridge_pred_class_train = ifelse(predict(ridge_model, newx = x_train, type = "response") > 0.5, 1, 0)
ridge_results_train = calculate_metrics(ridge_pred_class_train, y_train)
ridge_results_train$Set = "Train"

# Test set performance
ridge_pred_class_test = ifelse(predict(ridge_model, newx = x_test, type = "response") > 0.5, 1, 0)
ridge_results_test = calculate_metrics(ridge_pred_class_test, y_test)
ridge_results_test$Set = "Test"

# Output performance metrics
print(ridge_results_train)
print(ridge_results_test)
best_lambda 
table(ridge_pred_class_test, y_test)
coef_values = coef(ridge_model)
roc_obj = roc(test$diagnosis,ridge_pred_class_test)
plot(roc_obj, main = "ROC Curve", print.auc = TRUE,lty = 2 , col = 'gold')


cv_model = cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
# Get the best lambda value from cross-validation
best_lambda = cv_model$lambda.min

lasso_model = glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda)
lasso_pred_class_train = ifelse(predict(ridge_model, newx = x_train, type = "response") > 0.5, 1, 0)
lasso_results_train = calculate_metrics(ridge_pred_class_train, y_train)
lasso_results_train$Set = "Train"

# Test set performance
lasso_pred_class_test = ifelse(predict(lasso_model, newx = x_test, type = "response") > 0.5, 1, 0)
lasso_results_test = calculate_metrics(lasso_pred_class_test, y_test)
lasso_results_test$Set = "Test"

# Output performance metrics
print(lasso_results_train)
print(lasso_results_test)
best_lambda
table(lasso_pred_class_test, y_test)
coef_values = coef(lasso_model)
roc_obj = roc(test$diagnosis,lasso_pred_class_test)
plot(roc_obj, main = "ROC Curve", print.auc = TRUE,lty = 2 , col = 'gold')

################################################################################

LDA_model = lda(diagnosis ~ ., data = train)

LDA_pred_train = predict( LDA_model, newdata = train)$class
LDA_pred_test = predict( LDA_model, newdata = test)$class

LDA_train_result = calculate_metrics(LDA_pred_train, train$diagnosis)
LDA_test_result = calculate_metrics(LDA_pred_test, test$diagnosis)

# Output performance metrics
print(LDA_train_result)
print(LDA_test_result)
table(LDA_pred_test, y_test)

# Calculate the predicted probabilities for LDA
LDA_pred_prob_test = predict(LDA_model, newdata = test)$posterior[, 1]
roc_obj = roc(test$diagnosis, LDA_pred_prob_test)
plot(roc_obj, main = "ROC Curve", print.auc = TRUE, ylim = c(0, 1), lty = 2, col = 'gold')

plot(LDA_model)
plot(LDA_model, type="density")

################################################################################

QDA_model = qda(diagnosis ~ ., data = train)

QDA_pred_train = predict( QDA_model, newdata = train)$class
QDA_pred_test = predict( QDA_model, newdata = test)$class

QDA_train_result = calculate_metrics(QDA_pred_train, train$diagnosis)
QDA_test_result = calculate_metrics(QDA_pred_test, test$diagnosis)

# Output performance metrics
print(QDA_train_result)
print(QDA_test_result)
table(QDA_pred_test, y_test)

# Calculate the predicted probabilities for LDA
QDA_pred_prob_test = predict(QDA_model, newdata = test)$posterior[, 1]
roc_obj = roc(test$diagnosis, QDA_pred_prob_test)
plot(roc_obj, main = "ROC Curve", print.auc = TRUE,lty = 2 , col = 'gold')

###############################################################################

# Fit Naive Bayes model
nb_model = naiveBayes(diagnosis ~ ., data = train)

nb_pred_train = predict(nb_model, newdata = train, type = 'class')
nb_pred_test = predict(nb_model, newdata = test , type = 'class')


NB_train_result = calculate_metrics(nb_pred_train, train$diagnosis)
NB_test_result = calculate_metrics(nb_pred_test, test$diagnosis)

print(NB_train_result)
print(NB_test_result)
table(nb_pred_test, test$diagnosis)


# Calculate predicted probabilities for train and test sets
nb_probs_test = predict(nb_model, newdata = test, type = 'raw')

# Create ROC curve objects
roc_obj = roc(test$diagnosis, nb_probs_test[,1])
plot(roc_obj, main = "ROC Curve", print.auc = TRUE,lty = 2 , col = 'gold')

################################################################################
k_values = c(2 , 4 , 6 , 8 , 10 , 12 , 14 , 16 , 18 , 20)
KNN_results = list()

for (k in k_values) {
  KNN_model = knn(train[, -1], test[, -1], train[, 1], k)
  KNN_results[[as.character(k)]] = calculate_metrics(KNN_model, test$diagnosis)
}

for (k in k_values) {
  cat("Metrics for k =", k, ":\n")
  print(KNN_results[[as.character(k)]])
  cat("\n")
}


K = c(2, 4, 6, 8, 10, 12 , 14 , 16 , 18 , 20)
Accuracy = c(KNN_results$'2'[4,2],
              KNN_results$'4'[4,2],
              KNN_results$'6'[4,2],
              KNN_results$'8'[4,2],
              KNN_results$'10'[4,2],
              KNN_results$'12'[4,2],
              KNN_results$'14'[4,2],
              KNN_results$'16'[4,2],
              KNN_results$'18'[4,2],
              KNN_results$'20'[4,2])

F1_Score = c(KNN_results$'2'[3,2],
              KNN_results$'4'[3,2],
              KNN_results$'6'[3,2],
              KNN_results$'8'[3,2],
              KNN_results$'10'[3,2],
              KNN_results$'12'[3,2],
              KNN_results$'14'[3,2],
              KNN_results$'16'[3,2],
              KNN_results$'18'[3,2],
              KNN_results$'20'[3,2])

Precision = c(KNN_results$'2'[2,2],
               KNN_results$'4'[2,2],
               KNN_results$'6'[2,2],
               KNN_results$'8'[2,2],
               KNN_results$'10'[2,2],
               KNN_results$'12'[2,2],
               KNN_results$'14'[2,2],
               KNN_results$'16'[2,2],
               KNN_results$'18'[2,2],
               KNN_results$'20'[2,2])


Recall = c(KNN_results$'2'[1,2],
            KNN_results$'4'[1,2],
            KNN_results$'6'[1,2],
            KNN_results$'8'[1,2],
            KNN_results$'10'[1,2],
            KNN_results$'12'[1,2],
            KNN_results$'14'[1,2],
            KNN_results$'16'[1,2],
            KNN_results$'18'[1,2],
            KNN_results$'20'[1,2])

# Create a line plot
plot(K, Accuracy, type = "b", pch = 16, col = "blue",
     xlab = "K", ylab = "Metrics", main = "Performance Metrics" , ylim = c(0.7 , 0.98))
lines(K, F1_Score, type = "b", pch = 16, col = "red")
lines(K, Precision, type = "b", pch = 16, col = "green")
lines(K, Recall, type = "b", pch = 16, col = "orange")

# Add a legend
legend("bottomright", legend = c("Accuracy", "F1 Score", "Precision", "Recall"),
       col = c("blue", "red", "green", "orange"), lty = 1, pch = 16)



k = 18
KNN_model = knn(train[, -1], test[, -1], train[, 1], k)
KNN_probs = as.numeric(KNN_model == 1)
KNN_results$'18'
table(KNN_probs , test$diagnosis)
roc_obj = roc(test$diagnosis, KNN_probs)
plot(roc_obj, main = "ROC Curve", print.auc = TRUE, lty = 2, col = 'gold')



############################################################################################
results = data.frame(
  Model = c("Naive Bayes", "Logistic Regression", "Backward", "QDA", "KNN", "LASSSO Regression", "LDA", "Ridge Regression"),
  Recall = c(0.79, 0.81, 0.86,0.93,0.96,0.96,1,1),
  Precision = c(0.81,0.78, 0.75,0.87,0.84,0.818,0.66, 0.818),
  F1_Score = c(0.805,0.8, 0.806,0.90,0.903,0.88,0.8,0.9),
  Accuracy = c(0.88,0.88, 0.89,0.94,0.94,0.93,0.902,0.946),
  AUC = c(0.954,0.856, 0.854,0.973,0.918,0.903,0.975, 0.909)
)

# Reshape the data into long format
results_long = reshape2::melt(results, id.vars = "Model", variable.name = "Metric")

# Plot the results using a line plot
ggplot(results_long, aes(x = Model, y = value, color = Metric, group = Metric)) +
  geom_line() +
  geom_point() +
  labs(y = "Value", x = "Model", color = "Metric", ) +
  ggtitle("Comparison of Models") +
  theme(plot.title = element_text(hjust = 0.5))





