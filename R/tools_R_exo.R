# Set current working directory
WD = "."
setwd(WD)

df = read.csv("../datasets/iris.csv")

# Print column names
colnames(df)

# Get numerical columns
unlist(lapply(df, is.numeric))
num_cols = colnames(df)[unlist(lapply(df, is.numeric))]

# For each species compute the mean of numerical columns and store it in a stats table like:
stats = NULL
for (grp in levels(df$species)) {
  m = as.matrix(df[df$species == grp, num_cols])
  line = data.frame(species = grp, as.list(colMeans(m)))
  stats = rbind(stats, line)
}

print(stats)

## shorter version
aggregate(. ~ species, data = df, mean)

## Merging database
user1 = data.frame(name = c("eric", "sophie"),
                   age = c(22, 48),
                   gender = c("M", "F"),
                   job = c("engineer", "scientist"))
user2 = data.frame(name = c("alice", "john", "peter", "julie", "christine"),
                   age = c(19, 26, 33, 44, 35),
                   gender = c("F", "M", "M", "F", "F"),
                   job = c("student", "student", "engineer", "scientist", "scientist"))
user3 = rbind(user1, user2)
salary = data.frame(name = c("alice", "john", "peter", "julie"),
                    salary = c(2200, 2400, 3500, 4300))

user = merge(user3, salary, by = "name", all = TRUE)


df = user

fillmissing_with_mean <- function(df){
  num_cols = colnames(df)[unlist(lapply(df, is.numeric))]
  for (n in num_cols) {
    x = df[, n]
    df[is.na(x), n] = mean(x[!is.na(x)])  # mean(x, na.rm=TRUE)
  }
  return(df)
}

user_imputed = fillmissing_with_mean(user)

write.csv(user_imputed, "users_imputed.csv", row.names = FALSE)
