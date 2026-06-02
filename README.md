
---

# Movie Poster Classification (Comedy vs. Thriller)

MTH441: Linear Regression and ANOVA — Assignment 2 

This repository contains the implementation of a binary classification framework designed to predict whether a given movie poster corresponds to a **Comedy (`0`)** or a **Thriller (`1`)**. The methodology utilizes a custom image feature extraction pipeline linked to a **LASSO-penalized Logistic Regression Model (Generalized Linear Model framework)** estimated via Coordinate Descent.

---

## 📋 Statistical Formulation & Problem Setup

The objective is to estimate a regularized parameter vector $\hat{\beta}$ to classify a poster based on an extracted design matrix row vector $x_i \in \mathbb{R}^p$.

### 1. The Link Function

We assume a Bernoulli conditional distribution $Y_i \mid x_i \sim \text{Bernoulli}(p_i)$. We model the success probability $p_i = \mathbb{P}(Y_i = 1 \mid x_i)$ using the Logit link function:

$$\log\left(\frac{p_i}{1 - p_i}\right) = \sum_{j=1}^{p} x_{ij}\beta_j = x_i^T\beta$$

### 2. Parameter Estimation via LASSO Penalty

To handle the high-dimensional feature space ($p > n$ risk) and introduce sparsity, the coefficient vector $\hat{\beta}$ is estimated by maximizing the penalized log-likelihood objective function:

$$\hat{\beta} = \arg\max_{\beta} \left\{ \sum_{i=1}^{n} \left[ y_i (x_i^T\beta) - \log(1 + e^{x_i^T\beta}) \right] - \lambda \sum_{j=2}^{p} |\beta_j| \right\}$$

> **Note on Intercept Handling:** In accordance with standard regression theory, the intercept parameter ($\beta_1$) is **not** subjected to shrinkage ($L_1$ regularization). This is explicitly managed in the estimation phase by mapping a custom `penalty.factor` vector where the first element is set to `0`.

---

## 🛠️ Feature Extraction Matrix Design

The `make_feature()` function maps a raw 3-dimensional `imager` array into a single row vector $x_i \in \mathbb{R}^p$:

* **Intercept Term:** Appends a structural leading `1` to serve as the design matrix column for the baseline intercept parameter ($\beta_1$).
* **Color Distribution (RGB Histograms):** Computes a 16-bin normalized relative frequency histogram independently across the Red, Green, and Blue channels ($16 \times 3 = 48$ continuous covariates).
* **Grayscale Sample Moments:** Converts the image to grayscale to extract global distribution parameters: Empirical Mean ($\mu$), Standard Deviation ($\sigma$), Skewness ($\gamma_1$), and Kurtosis ($\gamma_2$) (4 continuous covariates).
* **Gradient Vectors & Edge Intensity:** Approximates spatial derivatives using image gradients ($G_x, G_y$). Computes the mean gradient magnitude, standard deviation, and an indicator density metric where $\mathbb{I}(\text{Magnitude} > 0.1)$ to capture high-frequency visual transitions (3 covariates).
* **Spatial Low-Frequency Downsampling:** Resizes the grayscale representation into a compact $16 \times 16$ spatial grid, flattening it into $256$ localized intensity covariates to capture macro-level spatial geometry and layout configurations.

---

## 💾 Model Artifacts (Workspace Deliverables)

The grading execution architecture loads your specific `.Rdata` workspace container, which must contain exactly these 4 specific structural objects:

| Object Name | Mathematical/Statistical Class | Description |
| --- | --- | --- |
| `make_feature` | Function mapping: $\text{Image} \to x_i \in \mathbb{R}^p$ 

 | Transforms a raw image array into an aligned numeric covariate row vector.

 |
| `beta_hat` | Estimated Parameter Vector: $\hat{\beta} \in \mathbb{R}^p$ 

 | The sparse regularized regression coefficient vector obtained at the optimal tuning parameter $\lambda_{\min}$.

 |
| `class_prob` | Link Evaluation Function: $\mathbb{R} \to [0, 1]$ 

 | Evaluates the inverse logit link (logistic cumulative distribution function) given the linear predictor $x_i^T\hat{\beta}$ to yield $\hat{p}_i$.

 |
| `classify` | Decision Boundary Operator: $[0, 1] \to \{"0", "1"\}$ 

 | Maps the continuous conditional probability $\hat{p}_i$ into a discrete classification choice using a threshold optimized via **Youden's J Statistic**.

 |

---

## 📈 Optimization & Evaluation Sequence

The final performance metric evaluates models based on minimizing the **sample misclassification rate** (empirical zero-one loss) on $N=100$ unobserved test posters.

The automated testing script replicates the professor's local grading matrix using this sequential functional composition:

```R
library(imager)
poster <- load.image("test_image.jpg")    # Grader inputs empirical data point
load("230401.Rdata")                      # Imports your estimated parameter space

x_i            <- make_feature(poster)    # 1. Map data to covariate space
linear_pred    <- sum(x_i * beta_hat)     # 2. Compute inner product / linear predictor
p_i            <- class_prob(linear_pred) # 3. Map linear predictor to probability space via link
predicted_label<- classify(p_i)           # 4. Apply decision boundary rule ("0" or "1")

```
