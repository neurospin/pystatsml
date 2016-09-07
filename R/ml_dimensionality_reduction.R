## PCA

# https://tgmstat.wordpress.com/2013/11/28/computing-and-visualizing-pca-in-r/


user1 = data.frame(name=c('eric', 'sophie'),
                     age=c(22, 48), gender=c('M', 'F'),
                     job=c('engineer', 'scientist'))
  
user2 = data.frame(name=c('alice', 'john', 'peter', 'julie', 'christine'),
                   age=c(19, 26, 33, 44, 35), gender=c('F', 'M', 'M', 'F', 'F'),
                   job=c("student", "student", 'engineer', 'scientist', 'scientist'))

user3 = rbind(user1, user2)

salary = data.frame(name=c('alice', 'john', 'peter', 'julie'), salary=c(22000, 2400, 3500, 4300))

user = merge(user3, salary, by="name", all=TRUE)

user[(user$gender == 'F') & (user$job == 'scientist'), ]

summary(user)

types = NULL
for(n in colnames(user)){
  types = rbind(types, data.frame(var=n, 
                                  type=typeof(user[[n]]),
                                  isnumeric=is.numeric(user[[n]])))
}

typeof(user['name'])
typeof(user['name'])

link = 'https://raw.github.com/neurospin/pystatsml/master/data/salary_table.csv'
X = read.csv(url(link))

X = read.csv(curl(link))

install.packages("RCurl")

install.packages(c("curl", "httr"))
require(RCurl)
myCsv <- getURL("https://gist.github.com/raw/667867/c47ec2d72801cfd84c6320e1fe37055ffe600c87/test.csv")
WhatJDwants <- read.csv(textConnection(myCsv))


urlfile<-'https://raw.github.com/aronlindberg/latent_growth_classes/master/LGC_data.csv'
dsin<-read.csv(curl(link))

library(curl)

x <- read.csv( curl("https://raw.githubusercontent.com/trinker/dummy/master/data/gcircles.csv") )

download.file("https://raw.github.com/aronlindberg/latent_growth_classes/master/LGC_data.csv", 
              destfile = "/tmp/test.csv", method = "curl")

download.file(link,  destfile = "/tmp/test.csv", method = "curl")
