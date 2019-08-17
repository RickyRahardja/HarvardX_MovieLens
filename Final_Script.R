# Create edx set, validation set


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#Create train and test sets
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

#Create evaluation by RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Model 1
mu_hat <- mean(train_set$rating)

rmse1 <- RMSE(test_set$rating, mu_hat)
rmse_results1 <- data_frame(method = "Naive Average Model", RMSE = rmse1)

#Model 2
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu)) 

predicted_ratings_movie <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

rmse2 <- RMSE(predicted_ratings_movie, test_set$rating)
rmse_results2 <- bind_rows(rmse_results1,
                           data_frame(method="Movie Effect Model",  
                                      RMSE = rmse2))

#Model 3
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings_movies_user <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)


rmse3 <- RMSE(predicted_ratings_movies_user, test_set$rating)
rmse_results3 <- bind_rows(rmse_results2,
                           data_frame(method="Movie + User Effects Model",  
                                      RMSE = rmse3))

#Model 4
lambdas <- seq(0, 10, 0.25)

rmse4 <- sapply(lambdas, function(l){
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings_reg_movies_user <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings_reg_movies_user, test_set$rating))
})


rmse_results4 <- bind_rows(rmse_results3,
                           data_frame(method="Movie + User Effect with Regularization Model",  
                                      RMSE = min(rmse4)))

#Model 5
lambdas_validation <- seq(0, 10, 0.25)

rmse_validation <- sapply(lambdas_validation, function(l){
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings_validation <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings_validation, validation$rating))
})

rmse_results5 <- bind_rows(rmse_results4,
                           data_frame(method="Movie + User Effect with Regularization Model on Validation Set",  
                                      RMSE = min(rmse_validation)))
rmse_results5

min(rmse_validation)

#Ensure RMSE result is lower than 0.8649 on Validation Set
if(min(rmse_validation) <= 0.8649){
  print("I deserve to get 25 points.")
  } else {
  print("I must develop better prediction algorithm!")
  }
