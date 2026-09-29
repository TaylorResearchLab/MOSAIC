krylov_ritz_convergence_operator <- function(A, n, m_max = 16, tol = 1e-6) {
  if (!is.function(A))
    stop("A must be a function: A(x)")
  if (missing(n))
    stop("You must provide dimension n.")
  # Initial vector
  v <- rnorm(n)
  v <- v / sqrt(sum(v^2))
  V <- matrix(0, n, m_max)
  H <- matrix(0, m_max, m_max)
  V[,1] <- v
  ritz_history <- vector("list", m_max)
  # Arnoldi iteration
  for (m in 1:m_max) {
    w <- A(V[,m])
    if (length(w) != n || any(!is.finite(w))) {
      stop("Operator A(x) returned an invalid vector.")
    }
    for (j in 1:m) {
      H[j,m] <- sum(V[,j] * w)
      w <- w - H[j,m] * V[,j]
    }
    if (m < m_max) {
      H[m+1,m] <- sqrt(sum(w^2))
      if (H[m+1,m] != 0)
        V[,m+1] <- w / H[m+1,m]
    }
    Hm <- H[1:m,1:m,drop=FALSE]
    ritz_history[[m]] <- eigen(Hm, symmetric = FALSE)$values
  }
  # Convergence scores
  convergence_scores <- vector("list", m_max)
  for (m in 2:m_max) {
    prev <- ritz_history[[m-1]]
    curr <- ritz_history[[m]]
    convergence_scores[[m]] <- sapply(curr, function(lambda)
      min(Mod(lambda - prev)))
  }
  # Final Ritz decomposition
  Hm <- H[1:m_max,1:m_max,drop=FALSE]
  eig <- eigen(Hm, symmetric = FALSE)
  final_vals <- eig$values
  Y <- eig$vectors
  # Ritz vectors in the original space
  X <- V %*% Y
  # Normalize
  norms <- sqrt(colSums(Mod(X)^2))
  norms[norms == 0] <- 1
  X <- sweep(X, 2, norms, "/")
  # Convergence score
  scores <- sapply(final_vals, function(lambda) {
    deltas <- c()
    for (m in 2:m_max) {
      curr <- ritz_history[[m]]
      idx <- which.min(Mod(curr - lambda))
      deltas <- c(deltas,
                  Mod(curr[idx] - lambda))
    }
    mean(deltas)
  })
  ordering <- order(scores)
  list(
    eigenvalues       = final_vals[ordering],
    eigenvectors      = X[, ordering, drop = FALSE],
    convergence_score = scores[ordering],
    raw_ritz          = ritz_history
  )
}
