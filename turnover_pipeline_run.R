#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
})

# ---- LOAD PIPELINE DEFINITIONS ----
source("R/turnover_pipeline.R")

# ---- ARG PARSER ----
parser <- ArgumentParser(description = "Run turnover model pipeline")

parser$add_argument(
  "--input",
  required = TRUE,
  help = "Input CSV file"
)

parser$add_argument(
  "--output",
  required = TRUE,
  help = "Output folder"
)

parser$add_argument(
  "--model",
  default = "default",
  choices = c("default", "alt"),
  help = "Choose predictor set: 'default' or 'alt'"
)

args <- parser$parse_args()

# ---- READ INPUT DATA ----
raw_data <- read.csv(args$input)

# ---- RUN MODEL SUITE ----
results <- run_turnover_model_suite(
  raw_data,
  model_type = args$model
)

# ---- AUGMENT RESULTS ----
aug <- save_and_augment_turnover_results(
  results      = results,
  raw_data     = raw_data,
  file_prefix  = "turnover",
  save_dir     = args$output,
  model_type   = args$model
)

# ---- FILE SUFFIX FOR OUTPUTS ----
suffix <- if (args$model == "alt") "_alt" else ""

# ---- FIT FINAL MODELS ON FULL DATA ----
cat("Fitting final RF and XGB models on full dataset...\n")

# Get the stored recipe and data
rec <- results$recipe_used
data <- results$model_data

# Build model specs and workflows
rf_spec <- rand_forest(
  mode = "classification",
  mtry = results$rf_results$best_config$mtry,
  min_n = results$rf_results$best_config$min_n,
  trees = 1000
) %>% set_engine("ranger", importance = "impurity")

rf_wf <- workflow() %>% add_model(rf_spec) %>% add_recipe(rec)
rf_fit <- fit(rf_wf, data = data)

xgb_spec <- boost_tree(
  mode = "classification",
  trees = 1000,
  learn_rate     = results$xgb_results$best_config$learn_rate,
  tree_depth     = results$xgb_results$best_config$tree_depth,
  min_n          = results$xgb_results$best_config$min_n,
  loss_reduction = results$xgb_results$best_config$loss_reduction,
  sample_size    = results$xgb_results$best_config$sample_size
) %>% set_engine("xgboost")

xgb_wf <- workflow() %>% add_model(xgb_spec) %>% add_recipe(rec)
xgb_fit <- fit(xgb_wf, data = data)

# Store the fitted models back into results
results$rf_results$final_fit  <- rf_fit
results$xgb_results$final_fit <- xgb_fit

cat("Final models fitted and stored in results object.\n")


# ---- SAVE MODEL SUITE ----
output_file <- file.path(args$output, paste0("turnover_results", suffix, ".rds"))
saveRDS(results, output_file)
cat("Saved model suite to:", output_file, "\n")

# ---- SAVE COMPARISON TABLE ----
comparison_csv <- file.path(args$output, paste0("model_comparison_metrics", suffix, ".csv"))
write.csv(results$comparison_table, comparison_csv, row.names = FALSE)
cat("Saved model comparison metrics to:", comparison_csv, "\n")

# ---- SAVE HYPERPARAMETERS SUMMARY ----
hyperparams <- bind_rows(
  results$glmnet_results$best_config %>%
    mutate(Model = "Penalised logistic (glmnet)"),
  results$rf_results$best_config %>%
    mutate(Model = "Random forest (ranger)"),
  results$xgb_results$best_config %>%
    mutate(Model = "Gradient boosting (xgboost)")
) %>%
  relocate(Model)

hyperparam_csv <- file.path(args$output, paste0("model_best_hyperparameters", suffix, ".csv"))
write.csv(hyperparams, hyperparam_csv, row.names = FALSE)

cat("Saved model best hyperparameters to:", hyperparam_csv, "\n")


# ---- DONE ----
cat("Turnover pipeline completed successfully (model:", args$model, ").\n")
