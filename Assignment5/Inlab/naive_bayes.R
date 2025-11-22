library(e1071)

data <- read.table("2020_bn_nb_data.txt", header=TRUE, stringsAsFactors=TRUE)

for (col in names(data)) {
  data[[col]] <- as.factor(data[[col]])
}

# Function to print clean table
format_accuracy_table <- function(acc_vec, title) {
  cat("\n==============================\n")
  cat(paste("   ", title, "\n"))
  cat("==============================\n")
  df <- data.frame(Run = 1:length(acc_vec),
                   Accuracy = round(acc_vec, 4))
  print(df, row.names = FALSE)
  cat("\nMean Accuracy:", round(mean(acc_vec), 4), "\n")
  cat("==============================\n\n")
}

acc_nb <- c()

set.seed(123)

for (i in 1:20) {
  idx <- sample(1:nrow(data), 0.7*nrow(data))
  train <- data[idx,]
  test  <- data[-idx,]

  model_nb <- naiveBayes(QP ~ ., data=train)
  pred <- predict(model_nb, test)

  acc <- mean(pred == test$QP)
  acc_nb <- c(acc_nb, acc)
}

format_accuracy_table(acc_nb, "Naive Bayes Classifier (20 Runs)")
