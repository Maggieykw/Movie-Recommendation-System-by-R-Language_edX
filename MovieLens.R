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

train_set <- edx
test_set <- validation

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

##########################################################
# 1a.Consider average in the model only
##########################################################

mu_hat <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mu_hat)
#naive_rmse #for result testing

#rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse) #for result testing
#rmse_results #for result testing
#RMSE of including average only: 1.0612018

##########################################################
# 1b.Add movie effect into model
##########################################################

movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

predicted_ratings <- mu_hat + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
#model_1_rmse #for result testing

#rmse_results <- bind_rows(rmse_results, #for result testing
                          #data_frame(method="Movie Effect Model",
                                     #RMSE = model_1_rmse ))
#rmse_results #for result testing
#RMSE of including average and movie effect model: 0.9439087

##########################################################
# 1c.Add user-specifc effect into model
##########################################################

user_avgs <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
#model_2_rmse #for result testing

#rmse_results <- bind_rows(rmse_results, #for result testing
                          #data_frame(method="Movie + User Effects Model",  
                                     #RMSE = model_2_rmse ))
#rmse_results %>% knitr::kable() #for result testing
#RMSE of including average, movie effect model and user effect model: 0.8292477 (which is the final RMSE result as well)

##########################################################
# 2.Final Result
##########################################################

data_frame("Final Result" ="Movie + User Effects Model",RMSE = model_2_rmse )%>% knitr::kable()

#Final RMSE (including average, movie effect model and user effect model): 0.8292477 
