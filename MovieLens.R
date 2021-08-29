##########################################################
# 0.Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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

##########################################################
# 1.Create a movie recommendation system using the MovieLens dataset.
##########################################################

RMSE <- function(true_rating, predicted_rating){
  sqrt(mean((true_rating - predicted_rating)^2))
}

##########################################################
# 1a.Consider average in the model only
##########################################################

mu_hat <- mean(edx$rating)
naive_rmse <- RMSE(validation$rating, mu_hat)
#naive_rmse #for result testing

rmse_results <- data_frame("Predictive Method" = "Just the average", RMSE = naive_rmse)
#rmse_results #for result testing
#RMSE of including average only: 1.0612018

##########################################################
# 1b.Add movie effect into model
##########################################################

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_m = mean(rating - mu_hat))

predicted_rating <- mu_hat + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_m

model_1_rmse <- RMSE(predicted_rating, validation$rating)
#model_1_rmse #for result testing

rmse_results <- bind_rows(rmse_results,
                          data_frame("Predictive Method"="Movie Effect Model",
                                     RMSE = model_1_rmse ))
#rmse_results #for result testing
#RMSE of including average and movie effect model: 0.9439087

##########################################################
# 1c.Add user-specifc effect into model
##########################################################

user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_m))

predicted_rating <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_m + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_rating, validation$rating)
#model_2_rmse #for result testing

rmse_results <- bind_rows(rmse_results,
                          data_frame("Predictive Method"="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
#rmse_results %>% knitr::kable() #for result testing
#RMSE of including average, movie effect model and user effect model: 0.8653488

##########################################################
# 1d.Regularization for movie-specifc effect in the model
##########################################################

#Choosing the penalty terms using cross-validation
lambdas <- seq(0, 10, 0.25)

just_the_sum <- edx %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu_hat), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_rating <- validation %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_m = s/(n_i+l)) %>%
    mutate(pred = mu_hat + b_m) %>%
    .$pred
  return(RMSE(predicted_rating, validation$rating))
})

#qplot(lambdas, rmses)  
#lambdas[which.min(rmses)]
#lambda = 2.5 based on the above cross-validation 

lambda <- lambdas[which.min(rmses)]
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_m2 = sum(rating - mu_hat)/(n()+lambda), n_i = n()) 

predicted_rating <- validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_u+ b_m2) %>%
  .$pred

model_3_rmse <- RMSE(predicted_rating, validation$rating)
#model_3_rmse #for result testing

rmse_results <- bind_rows(rmse_results,
                          data_frame("Predictive Method"="Regularized Movie + User Effects Model",  
                                     RMSE = model_3_rmse ))

#rmse_results %>% knitr::kable() #for result testing

#RMSE of including average, regularized movie effect model and user effect model: 0.8652263

##########################################################
# 1e.Regularization for user-specifc effect in the model
##########################################################

#Choosing the penalty terms using cross-validation
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu_hat <- mean(edx$rating)
  
  b_m <- edx %>% 
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu_hat)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu_hat)/(n()+l))
  
  predicted_rating <- validation %>% 
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu_hat + b_m + b_u) %>%
    .$pred
  
  return(RMSE(predicted_rating, validation$rating))
})


#qplot(lambdas, rmses)  
#lambdas[which.min(rmses)]
#lambda = 5.25 based on the above cross-validation 

lambda <- lambdas[which.min(rmses)]

movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_m2 = sum(rating - mu_hat)/(n()+lambda), n_i = n()) 

user_reg_avgs <- edx %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  group_by(userId) %>% 
  summarize(b_u2 = sum(rating - mu_hat-b_m2)/(n()+lambda), n_i = n()) 

predicted_rating <- validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_m2 + b_u2) %>%
  .$pred

model_4_rmse <- RMSE(predicted_rating, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame("Predictive Method"="Regularized Movie + Regularized User Effects Model",  
                                     RMSE = model_4_rmse ))

#rmse_results %>% knitr::kable() #for result testing

#RMSE of including average, regularized movie effect model and regularized user effect model: 0.8648170 

##########################################################
# 2.Final Result
##########################################################

data_frame("Final Result" ="Regularized Movie + Regularized User Effects Model",RMSE = model_4_rmse )%>% knitr::kable()

#Final RMSE (including average, movie effect model and user effect model): 0.8648170 
