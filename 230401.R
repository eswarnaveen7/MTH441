# ============================================
# MTH441 Assignment 2 - Poster Auto Training 
# ============================================

library(imager)
library(glmnet)
library(parallel)

# =====================================================
# 1. make_feature 
# =====================================================
make_feature <- function(img, resize_dim = c(128, 128), color_bins = 16) {
  if (!inherits(img, "cimg")) stop("make_feature expects an imager cimg object.")
  
  resized <- tryCatch({
    imager::resize(img, size_x = resize_dim[1], size_y = resize_dim[2])
  }, error = function(e) {
    tryCatch(imager::imresize(img, resize_dim[1], resize_dim[2]),
             error = function(e2) stop("Resize failed."))
  })
  
  resized <- resized / (max(resized) + 1e-9)
  
  channels <- imager::imsplit(resized, "c")
  if (length(channels) == 1) {
    ch_r <- channels[[1]]; ch_g <- channels[[1]]; ch_b <- channels[[1]]
  } else {
    ch_r <- channels[[1]]; ch_g <- channels[[2]]; ch_b <- channels[[3]]
  }
  
  vec_hist <- c()
  for (ch in list(ch_r, ch_g, ch_b)) {
    vals <- as.vector(ch)
    h <- hist(vals, breaks = seq(0, 1, length.out = color_bins + 1),
              plot = FALSE)$counts
    h <- h / sum(h)
    vec_hist <- c(vec_hist, h)
  }
  
  gray <- suppressWarnings(imager::grayscale(resized))
  gray_v <- as.vector(gray)
  g_mean <- mean(gray_v)
  g_sd <- sd(gray_v)
  g_skew <- mean((gray_v - g_mean)^3) / (g_sd^3 + 1e-9)
  g_kurt <- mean((gray_v - g_mean)^4) / (g_sd^4 + 1e-9)
  
  gx <- imager::imgradient(gray, "x")
  gy <- imager::imgradient(gray, "y")
  mag <- sqrt((as.numeric(gx))^2 + (as.numeric(gy))^2)
  edge_density <- mean(mag > 0.1)
  edge_mean <- mean(mag)
  edge_sd <- sd(mag)
  
  small <- imager::resize(gray, size_x = 16, size_y = 16)
  small_vec <- as.numeric(small)
  
  feat <- c(1, vec_hist, g_mean, g_sd, g_skew, g_kurt,
            edge_density, edge_mean, edge_sd, small_vec)
  as.numeric(feat)
}

# =====================================================
# 2. Probability & classify 
# =====================================================
class_prob <- function(lin_pred) 1 / (1 + exp(-lin_pred))

classify <- function(p_or_linpred, threshold = 0.5, input_is_prob = TRUE) {
  if (!input_is_prob) p <- class_prob(p_or_linpred) else p <- p_or_linpred
  ifelse(p >= threshold, "1", "0")
}

# =====================================================
# 3. Load dataset
# =====================================================
comedy_dir   <- "D:/SEM 5/MTH441/Assignment/testing/posters/comedy"
thriller_dir <- "D:/SEM 5/MTH441/Assignment/testing/posters/thriller"

comedy_files   <- list.files(comedy_dir,   pattern="\\.jpg$|\\.jpeg$|\\.png$", full.names=TRUE)
thriller_files <- list.files(thriller_dir, pattern="\\.jpg$|\\.jpeg$|\\.png$", full.names=TRUE)

cat("Found", length(comedy_files), "comedy and", length(thriller_files), "thriller posters.\n")

extract_safe <- function(path) {
  tryCatch(make_feature(load.image(path)),
           error=function(e) {cat("Skipping", basename(path), "\n"); NULL})
}

# parallel feature extraction
n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
clusterExport(cl, c("extract_safe", "make_feature"))
clusterEvalQ(cl, library(imager))

comedy_features   <- parLapply(cl, comedy_files, extract_safe)
thriller_features <- parLapply(cl, thriller_files, extract_safe)
stopCluster(cl)

comedy_features   <- Filter(Negate(is.null), comedy_features)
thriller_features <- Filter(Negate(is.null), thriller_features)

X_train <- do.call(rbind, c(comedy_features, thriller_features))
y_train <- c(rep(0, length(comedy_features)), rep(1, length(thriller_features)))

X_train[!is.finite(X_train)] <- 0
X_train[is.na(X_train)] <- 0

cat("✅ Extracted features for", length(y_train), "posters.\n")

# =====================================================
# 4. LASSO (Correct intercept handling)
# =====================================================
cat("Training LASSO logistic regression model (no scaling, correct intercept)...\n")

penalty <- c(0, rep(1, ncol(X_train)-1))   # do NOT penalize the intercept column (the 1)

cvfit <- cv.glmnet(
  X_train, y_train,
  family = "binomial",
  alpha = 1,
  intercept = FALSE,         # IMPORTANT
  standardize = FALSE,
  penalty.factor = penalty
)

best_lambda <- cvfit$lambda.min

fit <- glmnet(
  X_train, y_train,
  family = "binomial",
  alpha = 1,
  lambda = best_lambda,
  intercept = FALSE,         # IMPORTANT
  standardize = FALSE,
  penalty.factor = penalty
)

beta_hat <- as.numeric(fit$beta)
nonzero  <- sum(beta_hat != 0)

cat("🎯 Model trained with", nonzero, "non-zero coefficients.\n")
cat("Length of beta_hat =", length(beta_hat), "\n")

# =====================================================
# 5. Find optimal threshold (Youden J)
# =====================================================
lin_train <- as.numeric(X_train %*% beta_hat)
p_train <- class_prob(lin_train)

grid <- seq(0.01, 0.99, by=0.01)
youden <- sapply(grid, function(t) {
  pred <- as.integer(p_train >= t)
  TPR  <- mean(pred[y_train==1] == 1)
  TNR  <- mean(pred[y_train==0] == 0)
  TPR + TNR - 1
})

best_threshold <- grid[which.max(youden)]
cat(sprintf("✅ Optimal threshold (Youden J): %.3f\n", best_threshold))

# Modify classify() to use the learned threshold
classify <- function(p_or_linpred, threshold = best_threshold, input_is_prob = TRUE) {
  if (!input_is_prob) p <- class_prob(p_or_linpred) else p <- p_or_linpred
  ifelse(p >= threshold, "1", "0")
}

# =====================================================
# 6. Save RData
# =====================================================
save(make_feature, beta_hat, class_prob, classify,
     file = "230401.Rdata")

cat(sprintf("✅ Saved   230401.Rdata with threshold %.3f and %d active coefficients.\n",
            best_threshold, nonzero))