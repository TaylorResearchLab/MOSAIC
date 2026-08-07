library(igraph)
library(Matrix)
#set.seed(41)
#Generate Graph and Adjacency Matrix
g<-graph_from_edgelist(as.matrix(E[,1:2]),directed = FALSE)
A<-as_adjacency_matrix(g, sparse = TRUE)
#Sparse row-normalized random-walk matrix
row_sums<-rowSums(A)
row_sums[row_sums == 0]<-1
P<-Diagonal(x = 1/row_sums) %*% A
P<-as(P, "dgCMatrix")  # ensure sparse
#Multi-step diffusion operator
T_rw<-3           #number of steps
r_dim<-16         #embedding dimension
alpha<-0.9999    #damping factor
multi_step_op<-function(x, args = NULL) {
  y<-alpha*P %*% x
  result<-y
  for (t in 2:T_rw) {
    y<-alpha*P %*% y
    result<-result + y
  }
  return(as.numeric(result))
}
#Decomposition & embedding extraction
eig<-krylov_ritz_convergence_operator(multi_step_op, n = nrow(P), m_max = r_dim, tol = 1e-6)
eig$values<-Re(eig$eigenvalues)
eig$vectors<-Re(eig$eigenvectors)
r_dim_actual<-length(eig$values)
vals<-sqrt(pmax(eig$values[1:r_dim_actual], 1e-12))
Y<-scale(eig$vectors[, 1:r_dim_actual, drop = FALSE] %*% diag(vals))
