library(imager)

load("230401.Rdata")

img <- load.image("poster.jpg")

x <- make_feature(img)

lin_pred <- sum(x * beta_hat)

prob <- class_prob(lin_pred)

prediction <- classify(prob)

cat("Probability:", prob, "\n")
cat("Prediction:", prediction, "\n")