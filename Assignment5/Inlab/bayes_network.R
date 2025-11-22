library(bnlearn)
library(gRain)

# Load data
data <- read.table("2020_bn_nb_data.txt", header=TRUE, stringsAsFactors=TRUE)

# Convert to factors
for (col in names(data)) {
  data[[col]] <- as.factor(data[[col]])
}

# Learn BN structure using hill-climbing
bn_model <- hc(data)

# Save Bayesian Network structure plot
dir.create("outputs", showWarnings=FALSE)

jpeg("outputs/bn_structure_plot.jpg", width=900, height=600)
plot(bn_model)
dev.off()

cat("\n--- Bayesian Network Structure Learned ---\n")
print(bn_model)

# Fit CPTs
fitted_bn <- bn.fit(bn_model, data)
grain_bn <- as.grain(fitted_bn)

# Evidence for PH100 prediction
evidence <- list(EC100="DD", IT101="CC", MA101="CD")

# Query PH100 distribution
result <- querygrain(grain_bn, nodes="PH100", evidence=evidence)

cat("\n--- Predicted PH100 Grade Distribution ---\n")
print(result)
