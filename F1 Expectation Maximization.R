# Quite fast implementation of Faron's expected F1 maximization using Rcpp and R
library(inline)
library(Rcpp)
Sys.setenv("PKG_CXXFLAGS"="-std=c++11")

# Input: p: item reorder probabilities (sorted), p_none: none probability (0 if not specified)
# Output: matrix[2][n + 1] out: out[0][j] - F1 score with top j products and None
#                               out[1][j] - F1 score with top j products
cppFunction(
  'NumericMatrix get_expectations(NumericVector p, double p_none) {
  // Assuming p is sorted, p_none == 0 if not specified
  int n = p.size();
  NumericMatrix expectations = NumericMatrix(2, n + 1);
  double DP_C[n + 2][n + 1];
  std::fill(DP_C[0], DP_C[0] + (n + 2) * (n + 1), 0);
  if (p_none == 0.0) {
  p_none = std::accumulate(p.begin(), p.end(), 1.0, [](double &a, double &b) {return a * (1.0 - b);});
  }
  DP_C[0][0] = 1.0;
  for (int j = 1; j < n; ++j)
  DP_C[0][j] = (1.0 - p[j - 1]) * DP_C[0][j - 1];
  for (int i = 1; i < n + 1; ++i) {
  DP_C[i][i] = DP_C[i - 1][i - 1] * p[i - 1];
  for (int j = i + 1; j < n + 1; ++j)
  DP_C[i][j] = p[j - 1] * DP_C[i - 1][j - 1] + (1.0 - p[j - 1]) * DP_C[i][j - 1];
  }
  double DP_S[2 * n + 1];
  double DP_SNone[2 * n + 1];
  for (int i = 1; i < (2 * n + 1); ++i) {
  DP_S[i] = 1.0 / (1.0 * i);
  DP_SNone[i] = 1.0 / (1.0 * i + 1);
  }
  for (int k = n; k >= 0; --k) {
  double f1 = 0.0;
  double f1None = 0.0;
  for (int k1 = 0; k1 < (n + 1); ++k1) {
  f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1];
  f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1];
  }
  for (int i = 1; i < (2 * k - 1); ++i) {
  DP_S[i] = (1 - p[k - 1]) * DP_S[i] + p[k - 1] * DP_S[i + 1];
  DP_SNone[i] = (1 - p[k - 1]) * DP_SNone[i] + p[k - 1] * DP_SNone[i + 1];
  }
  expectations(0, k) = f1None + 2 * p_none / (2.0 + k);
  expectations(1, k) = f1;
  }
  return expectations;
  }'
)

# Input: ps - item reorder probabilities, prods - item ids
# Output: reordered items string (as required in submission)
exact_F1_max_none <- function(ps, prods) {
  prods <- as.character(prods)
  perm <- order(ps, decreasing = T)
  ps <- ps[perm]
  prods <- prods[perm]
  expectations <-  get_expectations(ps, 0.0)
  max_idx <-  which.max(expectations)
  add_none <- max_idx %% 2 == 1
  size <- as.integer(max(0, max_idx - 1) / 2)
  if (size == 0) {
    return("None")
  }
  else {
    if (add_none)
      return(paste(c(prods[1:size], "None"), collapse = " "))
    else 
      return(paste(prods[1:size], collapse = " "))
  }
}

library(data.table)
sub <- fread("C:/Users/WTG/kaggle/sub.csv")

submission <- sub %>%
       group_by(order_id) %>%
       summarise(products = exact_F1_max_none(reordered_prob, product_id))
