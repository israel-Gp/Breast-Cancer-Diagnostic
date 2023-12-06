# Libraries --------------------------------------------------------------

## Required Libraries ------------------------------------------------------

packages.to.install <- c(
  # General Purpose
  
  'tidyverse',
  # Tidyverse
  'janitor',
  # Data Cleaning
  'caret',
  # for machine learning
  'furrr',
  # Parallel mapping
  'broom',
  # For tidying base and tidyverse operations
  'lubridate',
  # For date-time variable manipulation
  'ggcorrplot',
  # For ggplot2 correlation plots
  'ggrepel',
  # GGplot2 repel texts
  'doParallel',
  # Parrallel Processing with CARET
  'report',
  # Streamlined reports
  'broom',
  
  # GGplot addons
  'ggridges',
  'ggh4x',
  'patchwork',
  'scales',
  
  # Nerural Plots
  'NeuralNetTools',
  
  # Feature Selection
  
  'Boruta',
  # Variable Importance wrappe
  'xgboost',
  # Boruta xgboost parsing in Boruta
  'varrank',
  #Variable rank based on mutual information
  'infotheo' # Entropy & Mutual Information
)

## Download Missing Libraries ----------------------------------------------

# Parse missing packages
missing.packages <-
  packages.to.install[!(packages.to.install %in% installed.packages()[, "Package"])]

if (length(missing.packages))
  install.packages(missing.packages, dependencies = TRUE, repos = 'http://cran.us.r-project.org')

# Load Base Libraries -----------------------------------------------------

library(janitor)
library(broom)
library(caret)
library(furrr)
library(lubridate)
library(ggcorrplot)
library(doParallel)
library(scales)
library(ggridges)
library(patchwork)
library(ggh4x)
library(Boruta)
library(varrank)
library(infotheo)
library(report)
library(broom)
library(tidyverse)

theme_set(theme_bw())


# Source Data -------------------------------------------------------------

# Creators
# William Wolberg
#
# Olvi Mangasarian
#
# Nick Street
#
# W. Street

# Data is the results of a breast mass biopsy
# Data has 2 bare variables and 10 repeated variables

# Download Data -----------------------------------------------------------

data_url <-
  'https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip'

download <- 'breast+cancer+wisconsin+diagnostic.zip'
if (!file.exists(download))
  download.file(data_url,
                download)

wdbc_data <- 'wdbc.data'
if (!file.exists(wdbc_data))
  unzip(download, wdbc_data)

wdbc_names <- 'wdbc.names'
if (!file.exists(wdbc_names))
  unzip(download, wdbc_names)

# Read Data ---------------------------------------------------------------

wdbc_raw <- read_csv(wdbc_data, col_names = FALSE)

wdbc_names_raw <- read_csv(wdbc_names, col_names = FALSE) %>%
  setNames(c('names')) %>%
  pull(names)

wdbc_names_raw
# Variable names start with [alnum]{1}\\)
# some variables will need to repeate
# This matched the source documentations

# Extract Names -----------------------------------------------------------

base_names <- wdbc_names_raw %>%
  str_subset('(?i)^[:alnum:]{1}\\)') %>%
  str_subset('\\:', negate = TRUE) %>%
  str_remove_all('^(?i)^[:alnum:]{1}\\)') %>%
  str_remove_all('\\(.*\\)') %>%
  str_squish()

core_names <- base_names[1:2]
repeat_names <- base_names[-(1:2)]

repeat_names_n <- (ncol(wdbc_raw) - 2) / 10

repeat_names <- rep(repeat_names, repeat_names_n)

# Clean Data --------------------------------------------------------------

wdbc <- wdbc_raw %>%
  # Set names
  setNames(c(core_names, repeat_names)) %>%
  # Clean Names
  clean_names() %>%
  # Set Classes
  mutate(
    id_number = as_factor(id_number),
    diagnosis = case_when(diagnosis == 'B' ~ 'Benign',
                          TRUE ~ 'Malignant') %>%
      as_factor()
  ) %>%
  # Relocate response to left side
  relocate(diagnosis)

# Partition Data ----------------------------------------------------------

# Partition to 64 Train/ 26 Test/ 10 Holdout

nrow(wdbc) * c(0.7, 0.2, 0.1)

# Expected sizes are ~400/114/57

glimpse(wdbc)
# Only categorical variable is the response variable
# no balancing of categorical variables required

## Holdout Set ------------------------------------------------------------

set.seed(2140, sample.kind = 'Rounding')
holdout_index <-
  createDataPartition(wdbc$diagnosis,
                      times = 1,
                      p = 0.1,
                      list = FALSE) %>%
  as.vector()

holdout_set <- slice(wdbc, holdout_index)
remaining_set <- slice(wdbc, -holdout_index)

## Train & Test Set -------------------------------------------------------

set.seed(2145, sample.kind = 'Rounding')
train_index <-
  createDataPartition(
    remaining_set$diagnosis,
    times = 1,
    p = 0.8,
    list = FALSE
  ) %>%
  as.vector()

train_set <- slice(remaining_set, train_index)
test_set <- slice(remaining_set, -train_index)

list(holdout_set = holdout_set,
     train_set = train_set,
     test_set = test_set) %>% map_vec(nrow)

# Remove remaining_set
rm(remaining_set)

# Explore Data ------------------------------------------------------------

# Exploration is limited to the training data

## Response Distribution --------------------------------------------------

train_set %>%
  count(diagnosis, sort = TRUE) %>%
  mutate(prop = n / sum(n))

# Over half of the results of FNA analysis breast mass result in a benign diagnosis

train_set %>%
  count(diagnosis, sort = TRUE) %>%
  mutate(prop = n / sum(n),
         diagnosis = fct_reorder(diagnosis, prop, .desc = TRUE)) %>%
  ggplot(aes(diagnosis, prop, fill = diagnosis, label = label_percent()(prop))) +
  geom_col(show.legend = FALSE) +
  geom_text(position = position_stack(vjust = 0.5)) +
  xlab('Diagnosis') +
  ylab('Proportion') +
  ggtitle('Breast Mass Biopsy Diagnosis')

## Numeric Predictors -----------------------------------------------------

train_set_tidy <- train_set %>%
  pivot_longer(
    cols = where(is.numeric),
    names_to = 'numeric_predictor',
    names_transform = list(numeric_predictor = as.factor)
  ) %>%
  mutate(
    predictor_group = str_remove(numeric_predictor, '_[:digit:]+$') %>%
      str_squish() %>%
      as.factor(),
    group_n = str_extract(numeric_predictor, '[:digit:]+$') %>%
      replace_na('1') %>%
      as_factor()
  )

train_set_tidy %>%
  ggplot(aes(value, fill = numeric_predictor)) +
  geom_density(show.legend = FALSE,
               alpha = 0.25,
               aes(value, after_stat(ndensity))) +
  geom_histogram(
    show.legend = FALSE,
    color = '#000000',
    alpha = 0.25,
    aes(value, after_stat(ndensity), fill = numeric_predictor)
  ) +
  facet_wrap2( ~ numeric_predictor,
               scales = 'free',
               axes = 'all',
               ncol = 3) +
  xlab('Value') +
  ylab('Density') +
  ggtitle('Numeric Predictor Distirbution')

# Similar measure variables have similar distributions
# With major deviations being quite notable (texture, smoothness, perimeter, concavity)
# Those that have similar distributions have somewhat different spreads.
# There may be very strong correlations between similar measures

numeric_predictor_means <- train_set_tidy %>%
  group_by(numeric_predictor) %>%
  summarise(mean = mean(value, na.rm = TRUE))

numeric_predictor_diagnosis_means <- train_set_tidy %>%
  group_by(numeric_predictor, diagnosis) %>%
  summarise(mean = mean(value, na.rm = TRUE)) %>%
  ungroup()

train_set_tidy %>%
  ggplot() +
  geom_density(
    show.legend = FALSE,
    alpha = 0.25,
    color = '#000000',
    linewidth = 2,
    aes(value, after_stat(ndensity))
  ) +
  geom_density(show.legend = FALSE,
               alpha = 0.25,
               aes(value, after_stat(ndensity), fill = diagnosis)) +
  geom_histogram(
    color = '#000000',
    alpha = 0.25,
    aes(value, after_stat(ndensity), fill = diagnosis)
  ) +
  geom_vline(
    data = numeric_predictor_means,
    aes(xintercept = mean),
    linetype = 'dashed',
    linewidth = 1.5
  ) +
  geom_vline(
    data = numeric_predictor_diagnosis_means,
    aes(xintercept = mean, color = diagnosis),
    linetype = 'dashed',
    linewidth = 1.5
  ) +
  facet_wrap2( ~ numeric_predictor,
               scales = 'free',
               axes = 'all',
               ncol = 3) +
  scale_y_continuous('Density', labels = comma) +
  scale_x_continuous('Value', labels = comma) +
  guides(fill = guide_legend('Diagnosis'), color = guide_legend('Diagnosis Mean')) +
  ggtitle('Numeric Predictor Distirbution grouped by Biopsy Diagnosis')

# When grouping by diagnosis vastly different distributions are observed
# Malignant when trends higher
# area, compactness, concave points, perimeter, radius
# Other features has some differences but they are not easily noticeable
# Variables with clear difference in means may be adequate predictors

# Difference in means -----------------------------------------------------

numeric_predictor_t_test <- train_set_tidy %>%
  pivot_wider(
    id_cols = numeric_predictor,
    names_from = diagnosis,
    values_from = value,
    values_fn = list
  ) %>%
  clean_names() %>%
  mutate(t.test = map2(malignant, benign, \(x, y) tidy(t.test(x, y)))) %>%
  select(-all_of(c('malignant', 'benign'))) %>%
  unnest(cols = contains('t.test')) %>%
  rowwise() %>%
  mutate(includes_zero = between(0, conf.low, conf.high)) %>%
  ungroup() %>%
  filter(!isTRUE(includes_zero)) %>%
  arrange(desc(estimate)) %>%
  mutate(rank = rank(-estimate))

numeric_predictor_t_test

# The top 3 mean diffrences are area_3, area and perimiter_3
# This mates the plot interpretation
# Proper feature selection may filter out some due to correlation or linearity

# Feature Selection -------------------------------------------------------


## Non Zero Variance ------------------------------------------------------

train_set_nzv <-
  nearZeroVar(
    train_set,
    saveMetrics = TRUE,
    names = TRUE,
    allowParallel = TRUE
  )

train_set_nzv

# There are no Near Zero Variance Variables

## Correlation ------------------------------------------------------------

train_correlation <- cor(select(train_set, where(is.numeric)))
train_correlation_pmap <-
  cor_pmat(select(train_set, where(is.numeric)))

train_corr_p <- ggcorrplot(
  train_correlation,
  type = 'lower',
  ggtheme = ggplot2::theme_bw,
  title = 'Biopsy Diagnosis Numeric Variable Pearson Correlations',
  hc.order = TRUE,
  lab = TRUE,
  p.mat = train_correlation_pmap
)

train_corr_p

# There are a very large amount of correlations
# There may be very stron non-lienear correlations
# To have a better scope rerun as spearman

train_correlation_s <-
  cor(select(train_set, where(is.numeric)), method = 'spearman')
train_correlation_S_pmap <-
  cor_pmat(select(train_set, where(is.numeric)), method = 'spearman')

train_corr_s <- ggcorrplot(
  train_correlation_s,
  type = 'lower',
  ggtheme = ggplot2::theme_bw,
  title = 'Biopsy Diagnosis Numeric Variable Pearson Correlations',
  hc.order = TRUE,
  lab = TRUE,
  p.mat = train_correlation_S_pmap
)

train_corr_p / train_corr_s

# There are even larger amount of predictors which are non-linearly correlated

# Will remove based on spearman

spearman_remove <- findCorrelation(train_correlation_s,
                                   verbose = TRUE,
                                   names = TRUE)

spearman_remove

train_set <- train_set %>%
  select(-all_of(spearman_remove))

## Linear Combos ----------------------------------------------------------

lcombos <- findLinearCombos(select(train_set,where(is.numeric)))

lcombos
# no linear combos

## Boruta Analysis --------------------------------------------------------

train_set <- train_set %>%
  select(-all_of('id_number'))

# GGplot Boruta Plot
boruta_interpret <-
  function(x, title = NULL, subtitle = NULL) {
    decisions <- tibble(variable = names(x$finalDecision),
                        decision = as.character(x$finalDecision))
    
    importance <- as_tibble(x$ImpHistory) %>%
      pivot_longer(cols = everything(),
                   names_to = 'variable')
    
    data <- left_join(importance, decisions) %>%
      replace_na(list(decision = 'Metric')) %>%
      mutate(across(where(is.character), as.factor)) %>%
      mutate(variable = fct_reorder(variable, value, .desc = FALSE))
    
    plot <- data %>%
      ggplot(aes(variable, value, fill = decision)) +
      geom_boxplot(alpha = 0.25) +
      geom_jitter(position = position_jitterdodge()) +
      scale_y_continuous('Importance') +
      xlab('Predictor') +
      guides(fill = guide_legend('Decision')) +
      ggtitle(title, subtitle = subtitle) +
      coord_flip()
    
    return(plot)
    
  }

set.seed(756, sample.kind = 'Rounding')
boruta_fs <- Boruta(diagnosis  ~ .,
                    data = train_set,
                    doTrace = 3,
                    maxRuns = 10000)

boruta_interpret(
  boruta_fs,
  'Biopsy Diagnosis Boruta Feature Selection',
  'Biopsy Diagnosis festure selection via Boruta analysis with random forest variable importance'
)

# Boruta found no attributes to be unimportant
# Considering that these are all numeric predicxtors the XGBoost variant of Boruta will
# be applied

set.seed(30, sample.kind = 'Rounding')
boruta_fs_xgb <- Boruta(
  diagnosis  ~ .,
  data = train_set,
  doTrace = 3,
  getImp = getImpXgboost,
  maxRuns = 10000
)

boruta_interpret(
  boruta_fs_xgb,
  'Biopsy Diagnosis Boruta Feature Selection',
  'Biopsy Diagnosis festure selection via Boruta analysis with XGBoost variable importance'
)

boruta_fs_xgb

# Boruta with XGboost found 6 important variables
# Concave points being the largest contributor

# XGboost predictions may be better suited as it reduces the feature space significantly
# XGboost variant will be used

train_set <- train_set %>%
  select(all_of(c(
    'diagnosis', getSelectedAttributes(boruta_fs_xgb)
  )))

# Data Preparation --------------------------------------------------------

# A model that applies PCA and data preparation will be compared to
# a model without these applied

data_pre_process <-
  preProcess(
    train_set,
    method = c('center', 'scale', 'YeoJohnson', 'pca'),
    # PCA cutoff
    thresh = 0.95
  )

data_pre_process

# PCA reduces 1 predictor with a variance threshold of 0.95
train_set_pca <- as_tibble(predict(data_pre_process, train_set))

# Train Model -------------------------------------------------------------

# 2 Core Models will be trained
# with each data set
# a total of 4 models

# The core model are
# a) Neural Network
# b) eXtreme Gradient Boosting via xgbLinear

# As cancer diagnosis is a sensitive issue the model will be trained for
# sensitivity

## Training Parameters ----------------------------------------------------

nnet_hyp <- expand.grid(size = seq(from = 1, to = 50, by = 1),
                        decay = seq(from = 0.1, to = 1, by = 0.1))

xgb_hyp <- expand.grid(
  nrounds = c(50,100), 
  eta = 0.3,
  alpha = 0:1,
  lambda = 0:1
)

## Train Control ----------------------------------------------------------

train_control <- trainControl(
  method = 'repeatedcv',
  number = 10,
  repeats = 5,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  allowParallel = TRUE
)

## Parallel Processing Settings -------------------------------------------

cores <- detectCores() - 1

# Train Models ------------------------------------------------------------

# Start Parallel Processing
cl <- makePSOCKcluster(cores)
registerDoParallel(cl)

# Neural Network
set.seed(121, sample.kind = 'Rounding')
nnet <- train(
  diagnosis ~ .,
  data = train_set,
  method = 'nnet',
  metric = 'Sens',
  trControl = train_control,
  tuneGrid = nnet_hyp
)

# Neural Network PCA
set.seed(125, sample.kind = 'Rounding')
nnet_pca <- train(
  diagnosis ~ .,
  data = train_set_pca,
  method = 'nnet',
  metric = 'Sens',
  trControl = train_control,
  tuneGrid = nnet_hyp
)

# XGBoost
set.seed(234, sample.kind = 'Rounding')
xbg <- train(
  diagnosis ~ .,
  data = train_set,
  method = 'xgbLinear',
  metric = 'Sens',
  trControl = train_control,
  tuneGrid = xgb_hyp
)

# XGBoost PCA
set.seed(700, sample.kind = 'Rounding')
xgb_pca <- train(
  diagnosis ~ .,
  data = train_set_pca,
  method = 'xgbLinear',
  metric = 'Sens',
  trControl = train_control,
  tuneGrid = xgb_hyp
)

# Stop Parallel Processing
stopCluster(cl)
registerDoSEQ()

# Test Models -------------------------------------------------------------

nnet_pred <- predict(nnet, test_set)
nnet_pca_pred <- predict(nnet_pca,predict(data_pre_process, test_set))
xgb_pred <- predict(xbg, test_set)
xgb_pca_pred <- predict(xgb_pca,predict(data_pre_process, test_set))

# NNET
confusionMatrix(nnet_pred,test_set$diagnosis)
# Sensitivity : 0.9211, Specificity : 0.9844

# NNET PCA
confusionMatrix(nnet_pca_pred,test_set$diagnosis)
# Sensitivity : 0.9211, Specificity : 0.9844

# No change in predictive power NNET PCA not required

# XGBoost
confusionMatrix(xbg_pred,test_set$diagnosis)
# Sensitivity : 0.8947, Specificity : 0.9688

# XGBoost PCA
confusionMatrix(xgb_pca_pred,test_set$diagnosis)
# Sensitivity : 0.9211, Specificity : 0.9531

model_resamples <- resamples(list(nnet = nnet,nnet_pca = nnet_pca,xbg = xbg,xbg_pca = xgb_pca))

bwplot(model_resamples)

model_sensitivity <- list(nnet = nnet_pred,nnet_pca = nnet_pca_pred,xbg = xbg_pred,xbg_pca = xgb_pca_pred) %>% 
  map_vec(sensitivity,test_set$diagnosis)

model_specificity <- list(nnet = nnet_pred,nnet_pca = nnet_pca_pred,xbg = xbg_pred,xbg_pca = xgb_pca_pred) %>% 
  map_vec(specificity,test_set$diagnosis)

model_performance <- tibble(model = c('Neural Network','Neural Network PCA','xGBoost','xGBoost PCA'),
                            sensitivity = model_sensitivity,
                            specificity = model_specificity
                            )

model_performance %>% 
  arrange(desc(sensitivity))

# While resmples shows that NNET PCA has the highest sensitivity
# The Golden Model is a Neural Network without PCA as it has with a high Sensitivity and a high specificity
# The values of very high for a classifier
# the performance will be tested on the holdout set

# Gold Model Performance --------------------------------------------------

nnet_hold_pred <- predict(nnet, holdout_set)

confusionMatrix(nnet_hold_pred,holdout_set$diagnosis)

# The model has lower than test set sensitivity but still high for a classifier
# The model is considered accurately trained considering the small sample
# size

NeuralNetTools::plotnet(nnet)
summary(nnet)

# Model consists of 6 predictors, 1 hidden layer, 12 neurons and two bias values