library(tidymodels)
library(embed)
library(ggplot2)
library(recipes)
library(vroom)
library(themis)
library(discrim)
library(keras)
library(tensorflow)


train_data <- vroom("C:/Users/sfolk/Desktop/STAT348/GGG/train.csv")
test_data <- vroom("C:/Users/sfolk/Desktop/STAT348/GGG/test.csv")

my_recipe <- recipe(type~., data = train_data) %>% 
  step_mutate_at(color, fn = factor) %>% 
  step_mutate(id, feature = id) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>% 
  step_smote(all_outcomes(), neighbors=3) %>% 
  step_range(all_numeric_predictors(), min=0, max=1)


#Naive Bayes Model

nb_model <- naive_Bayes(Laplace=tune(),
                        smoothness=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

nb_workflow <- workflow() %>% 
  add_model(nb_model) %>% 
  add_recipe(my_recipe)

tuning_grid <- grid_regular(Laplace(), smoothness(), levels = 20)


folds <- vfold_cv(train_data, v = 10, repeats=1)

cv_results <- nb_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(roc_auc, accuracy))

best_tune <- cv_results %>% select_best(metric='roc_auc')

final_workflow <- nb_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

nb_preds <- predict(final_workflow, 
                    new_data = test_data,
                    type = 'class')

nb_submission <- nb_preds %>% 
  bind_cols(., test_data) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class) 


vroom_write(x = nb_submission, file = "C:/Users/sfolk/Desktop/STAT348/GGG/NB_Predictions.csv", delim = ",")


