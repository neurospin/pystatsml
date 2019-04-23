set.seed(42)

# http://www.statmethods.net/advstats/matrix.html

### Dot product and Euclidean norm

a = c(2, 1)
b = c(1, 1)

euclidian <- function(x){
  return(sqrt(x %*% x)[1])
}

euclidian(a)

euclidian(a - b)

b %*% (a / euclidian(a))

X = matrix(rnorm(100 * 2), 100, 2)
dim(X)
X %*% (a / euclidian(a))

### Compute row means and store them into a vector

row_means = function(X) {
  means = NULL
  for (i in 1:dim(X)[1]) {
    means = c(means, mean(X[i, ]))
  }
  return(means)
}
row_means(X)

## version 2, using built-in accessors

row_means <- function(X) {
  n <- nrow(X)
  means <- numeric(n)
  for (i in 1:n)
    means[i] <- mean(X[i, ])

  return(means)
}

## version 3, using apply
apply(X, 1, mean)

## version 4, more efficient
rowMeans(X)

### Covariance matrix and Mahalanobis norm

N = 100
mu = c(1, 1)
Cov = matrix(c(1, .8,
               .8, 1), 2, 2)

library(MASS)
X = mvrnorm(N, mu, Cov)

xbar = colMeans(X)
print(xbar)

Xc = (X - xbar)

colMeans(Xc)

S = 1 / (N - 1) * (t(Xc)  %*% Xc)
print(S)


Sinv = solve(S)

x = X[1, ]

mahalanobis <- function(x, xbar, Sinv){
  xc = x - xbar
  return(sqrt( (xc %*% Sinv) %*% xc))
}


dist = matrix(nrow = N, ncol = 2)

for(i in 1:nrow(X)){
    dist[i, 1] = mahalanobis(X[i, ], xbar, Sinv)
    dist[i, 2] = euclidian(X[i, ] - xbar)
}
colnames(dist) = c("Mahalanobis", "Euclidian")

print(dist[1:10, ])

x = X[1, ]

print(sqrt(stats::mahalanobis(X[1, ], xbar, Sinv, inverted = TRUE)) -
      mahalanobis(X[1, ], xbar, Sinv))
