library(bnlearn)

data <- read.table("2020_bn_nb_data.txt", header=TRUE, stringsAsFactors=TRUE)

for (col in names(data)) {
  data[[col]] <- as.factor(data[[col]])
}

# Function for neat accuracy table
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

acc_bn <- c()

set.seed(123)

for (i in 1:20) {
  idx <- sample(1:nrow(data), 0.7*nrow(data))
  train <- data[idx,]
  test  <- data[-idx,]

  # Learn BN on training set
  bn_struct <- hc(train)
  bn_fit <- bn.fit(bn_struct, train)

  # Predict QP using BN
  pred <- predict(bn_fit, node="QP", data=test)

  acc <- mean(pred == test$QP)
  acc_bn <- c(acc_bn, acc)
}

format_accuracy_table(acc_bn, "Bayesian Network Classifier (20 Runs)")
