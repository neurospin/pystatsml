######
## PCA
######

# Write a class `BasicPCA` with two methods `fit(X)` that estimates the data mean
# and principal components directions. `transform(X)` that project a new the data
# into the principal components.
# 
# Check that your `BasicPCA` pfermed simillarly than the one from sklearn:
#   `from sklearn.decomposition import PCA`


BasicPCA <- function(X, scale=FALSE){
  obj = list()
  Xc <- scale(X, center=TRUE, scale=scale)
  obj$mean <- attr(Xc, "scaled:center")
  s <- svd(Xc, nu = 0)
  # v [K x P] a matrix whose columns contain the right singular vectors of x
  obj$V = s$v
  obj$var = 1 / (nrow(X) - 1) * s$d ^2
  return(obj)
}

BasicPCA.transform <- function(obj, X){
  Xc <- scale(X, center=obj$mean, scale=FALSE)
  return(Xc %*% obj$V)
}

# https://tgmstat.wordpress.com/2013/11/28/computing-and-visualizing-pca-in-r/
# dataset
n_samples = 10
experience = rnorm(n_samples)
salary = 1500 + experience + .5 * rnorm(n_samples)
other = rnorm(n_samples)
X = cbind(experience, salary, other)

# Optional: standardize data
Xcs = scale(X, center=TRUE, scale=FALSE)
attr(Xcs, "scaled:center") = NULL
attr(Xcs, "scaled:scale") = NULL

basic_pca = BasicPCA(Xcs)
BasicPCA.transform(basic_pca, Xcs)

# PCA with prcomp
pca = prcomp(Xcs, center=TRUE, scale.=FALSE)
names(pca)

# Compare
all(pca$rotation == basic_pca$V)
all(predict(pca, Xcs) == BasicPCA.transform(basic_pca, Xcs))

# "https://raw.github.com/neurospin/pystatsml/master/data/iris.csv"
# 
# Describe the data set. Should the dataset been standardized ?
# 
# Retrieve the explained variance ratio. Determine $K$ the number of components.
# 
# Print the $K$ principal components direction and correlation of the $K$ principal
# components with original variables. Interpret the contribution of original variables
# into the PC.
# 
# Plot samples projected into the $K$ first PCs.
# 
# Color samples with their species.
#

url = 'ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/data/iris.csv'
data = read.csv(url)
#setwd("/home/ed203246/git/pystatsml/notebooks")
data = read.csv("../data/iris.csv")

# Describe the data set. Should the dataset been standardized ?

summary(data)
# sepal_length    sepal_width     petal_length    petal_width          species  
# Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100   setosa    :50  
# 1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300   versicolor:50  
# Median :5.800   Median :3.000   Median :4.350   Median :1.300   virginica :50  
# Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199                  
# 3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800                  
# Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500 

numcols = colnames(data)[unlist(lapply(data, is.numeric))]
apply(data[, numcols], 2, sd)
#sepal_length  sepal_width petal_length  petal_width 
#0.8280661    0.4358663    1.7652982    0.7622377 


# Describe the structure of correlation among variables.
X = data[, numcols]
cor(X)

# Compute a PCA with the maximum number of compoenents.
Xcs = scale(X, center=TRUE, scale=TRUE)
attr(Xcs, "scaled:center") = NULL
attr(Xcs, "scaled:scale") = NULL
apply(Xcs, 2, sd)
apply(Xcs, 2, mean)

#Compute a PCA with the maximum number of compoenents.
pca = prcomp(Xcs)

# Variance ratio by component
(pca$sdev ** 2) / sum(pca$sdev ** 2)
#[1] 0.729624454 0.228507618 0.036689219 0.005178709

# cumulative explained variance
cumsum(pca$sdev ** 2) / sum(pca$sdev ** 2)

# K = 2
names(pca)
pca$rotation

PC = predict(pca, Xcs)
t(cor(Xcs, PC[, 1:2]))
# sepal_length sepal_width petal_length petal_width
# PC1    0.8901688  -0.4601427   0.99155518  0.96497896
# PC2   -0.3608299  -0.8827163  -0.02341519 -0.06399985

data = cbind(data, PC)

# Plot samples projected into the K first PCs
# Color samples with their species.
library(ggplot2)

qplot(PC1, PC2, data=data, colour=species)

####################################################################
## MDS
####################################################################

##############
## eurodist ##
##############

# Perform similar analysis on eurodist dataset using R, using:
# - MDS: cmdscale.
# - Euclidian parwise distance: dist
#
#url = 'ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/data/eurodist.csv'
#data = read.csv(url)

setwd("~/git/pystatsml/notebooks")
#url = 'ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/data/eurodist.csv'
data = read.csv("../data/eurodist.csv")

city = data[["city"]]
D = data[, 2:ncol(data)]

print(data[1:5, 1:5])

# Arbitrary choice of K=2 components
mds = cmdscale(D, k=2, , eig=T)

# Recover coordinates of the cities in Euclidian referential whose orientation is arbitrary.
print(as.matrix(dist(mds$points))[1:5, 1:5])

plot(mds$points[,1], -mds$points[,2])
text(mds$points[,1], -mds$points[,2], city, cex=0.8)


# Apply MDS using cmdscale
k_range = 1:(min(5, dim(D)-1))
stress <- rep(0, max.k)
for (kk in k_range){
  mds <- cmdscale(D, k=kk, eig=T)
  stress[kk] = (sum((D - as.matrix(dist(mds$points))) ^ 2)) ^ 0.5
}
plot(k_range, stress, type="l", xlab="k", ylab="stress")
#cbind(1:max.k,P.k)

# Ressources
# http://people.stat.sc.edu/Hitchcock/chapter5_R_examples.txt

##########
## iris ##
##########

# Perform similar analysis on eurodist dataset using R, using:
# - MDS: cmdscale.
# - Euclidian parwise distance: dist
#
#url = 'ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/data/iris.csv'
#data = read.csv(url)

setwd("~/git/pystatsml/notebooks")
#url = 'ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/data/iris.csv'
data = read.csv("../data/iris.csv")

species = data[["species"]]
X = scale(data[, 1:4])
attr(X, "scaled:center") = NULL
attr(X, "scaled:scale") = NULL
D = as.matrix(dist(X))
print(D[1:5, 1:5])

# Select K
k_range = 1:(min(5, dim(D)-1))
stress <- rep(0, max.k)
for (kk in k_range){
  mds <- cmdscale(D, k=kk, eig=T)
  stress[kk] = (sum((D - as.matrix(dist(mds$points))) ^ 2)) ^ 0.5
}
plot(k_range, stress, type="l", xlab="k", ylab="stress")

K = 2 # components
mds = cmdscale(D, k=K , eig=T)

# Recover coordinates of the cities in Euclidian referential whose orientation is arbitrary.
print(as.matrix(dist(mds$points))[1:5, 1:5])

plot(mds$points[,1], -mds$points[,2], col=species)

# PCA with prcomp
pca = prcomp(X, center=TRUE, scale.=FALSE)
names(pca)
PC = predict(pca, X)[, 1:K]

# Compute correlation between PCA and MDS components
cor(cbind(mds$points, PC))

#     1.000000e+00  1.551000e-16 1.000000e+00  4.766625e-16
#     1.551000e-16  1.000000e+00 4.474091e-16 -1.000000e+00
# PC1 1.000000e+00  4.474091e-16 1.000000e+00  1.842964e-16
# PC2 4.766625e-16 -1.000000e+00 1.842964e-16  1.000000e+00


####################################################################
## isomap
####################################################################
install.packages("vegan")

s_curve = read.csv("../data/s_curve.csv")
colnames(s_curve)

X = as.matrix(s_curve[, c("x", "y", "z")])
color = s_curve[["color"]]
D <- dist(X, method="euclidean")

library(vegan)

iso = isomap(D, ndim=2, k=10)

#install.packages("ggplot2")
library(ggplot2)

qplot(iso$points[,1], iso$points[,2], col=color) + scale_colour_gradientn(colours=rainbow(4))
scale_fill_distiller(palette = "Spectral")