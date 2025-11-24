#!/usr/bin/env Rscript

############################################################
# TURNOVER EVALUATION SUITE (Optimised & Auto-Skip Version)
############################################################

suppressPackageStartupMessages({
  library(tidymodels)
  tidymodels_prefer()
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(yardstick)
  library(fastshap)
  library(argparse)
  library(purrr)
  library(gghighlight)
  library(ggborderline)
})

# ----------------- CLI ARGUMENTS -----------------

parser <- ArgumentParser(description = "Turnover model evaluation suite")

parser$add_argument(
  "--input",
  required = TRUE,
  help = "Path to turnover_results.rds produced by turnover_pipeline.R"
)

parser$add_argument(
  "--output",
  required = TRUE,
  help = "Directory to save plots and output"
)

parser$add_argument(
  "--skip-shap",
  action = "store_true",
  help = "Skip SHAP computation (for speed)"
)

parser$add_argument(
  "--skip-pdp",
  action = "store_true",
  help = "Skip PDP computation (for speed)"
)

args <- parser$parse_args()

# ----------------- HELPER: SCRIPT PATH -----------------

get_script_path <- function() {
  cmdArgs <- commandArgs(trailingOnly = FALSE)
  needle <- "--file="
  match <- grep(needle, cmdArgs)
  if (length(match) > 0) {
    return(dirname(normalizePath(sub(needle, "", cmdArgs[match]))))
  } else {
    return(getwd())
  }
}

script_dir <- get_script_path()
source(file.path(script_dir, "R/turnover_pipeline.R"))

# ----------------- LOAD RESULTS -----------------

results <- readRDS(args$input)

############################################################
# 1. Extract CV predictions
############################################################

extract_cv_predictions <- function(results) {
  glmer_pred <- results$glmer_results$predictions %>%
    mutate(
      model = "Mixed-effects logistic (glmer)",
      truth = factor(actual, levels = c(0, 1),
                     labels = c("no_turnover", "turnover")),
      prob = prob
    ) %>%
    select(model, truth, prob)
  
  glmnet_best <- results$glmnet_results$best_config$.config
  glmnet_pred <- results$glmnet_results$tune_results %>%
    collect_predictions() %>%
    filter(.config == glmnet_best) %>%
    mutate(model = "Penalised logistic (glmnet)",
           truth = turnover_factor, prob = .pred_turnover) %>%
    select(model, truth, prob)
  
  rf_best <- results$rf_results$best_config$.config
  rf_pred <- results$rf_results$tune_results %>%
    collect_predictions() %>%
    filter(.config == rf_best) %>%
    mutate(model = "Random forest (ranger)",
           truth = turnover_factor, prob = .pred_turnover) %>%
    select(model, truth, prob)
  
  xgb_best <- results$xgb_results$best_config$.config
  xgb_pred <- results$xgb_results$tune_results %>%
    collect_predictions() %>%
    filter(.config == xgb_best) %>%
    mutate(model = "Gradient boosting (xgboost)",
           truth = turnover_factor, prob = .pred_turnover) %>%
    select(model, truth, prob)
  
  bind_rows(glmer_pred, glmnet_pred, rf_pred, xgb_pred)
}

############################################################
# 2. ROC / Calibration / Confusion Matrix
############################################################

plot_turnover_roc <- function(cv_preds, title_suffix = "") {
  
  roc_df <- cv_preds %>%
    group_by(model) %>%
    roc_curve(truth, prob, event_level = "second") %>%
    ungroup()
  
  roc_df$model <- factor(
    roc_df$model,
    levels = c("Mixed-effects logistic (glmer)",
               "Penalised logistic (glmnet)",
               "Random forest (ranger)",
               "Gradient boosting (xgboost)")
  )
  
  ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity, colour = model)) +
    geom_abline(linetype = "dashed", colour = "grey50") +
    geom_step(linewidth = 1, show.legend = FALSE) +
    coord_equal() +
    scale_x_continuous(
      breaks = seq(0, 1, by = 0.25),
      labels = c("0", "0.25", "0.50", "0.75", "1")
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, by = 0.25),
      labels = c("0", "0.25", "0.50", "0.75", "1")
    ) +
    scale_color_manual(values = c('#DB444B','#006BA2','#3EBCD2','#379A8B')) +
    gghighlight(use_direct_label = FALSE) +
    facet_wrap(~model, nrow = 1) +
    labs(
      x = "1 - Specificity",
      y = "Sensitivity",
      title = paste0("ROC Curves", title_suffix)
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 8),
      panel.grid.minor.x = element_blank(),
      panel.grid.minor.y = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_blank()
    )
}


plot_turnover_calibration <- function(cv_preds, n_bins = 10, title_suffix = "") {
  
  calib_df <- cv_preds %>%
    mutate(
      prob_clipped = pmin(pmax(prob, 1e-6), 1 - 1e-6),
      bin = ntile(prob_clipped, n_bins)
    ) %>%
    group_by(model, bin) %>%
    summarise(
      mean_pred = mean(prob_clipped),
      obs_rate  = mean(truth == "turnover"),
      n = n(),
      .groups = "drop"
    )
  
  calib_df$model <- factor(
    calib_df$model,
    levels = c("Mixed-effects logistic (glmer)",
               "Penalised logistic (glmnet)",
               "Random forest (ranger)",
               "Gradient boosting (xgboost)")
  )
  
  ggplot(calib_df, aes(x = mean_pred, y = obs_rate, colour = model)) +
    geom_abline(linetype = "dashed", colour = "grey50") +
    geom_borderline(
      linewidth = 1,
      bordercolour = "white",
      borderwidth = 0.5,
      show.legend = FALSE
    ) +
    geom_point(
      shape = 21,
      aes(fill = model),
      colour = "white",
      size = 2,
      stroke = 1,
      show.legend = FALSE
    ) +
    gghighlight(use_direct_label = FALSE) +
    scale_color_manual(values = c('#DB444B','#006BA2','#3EBCD2','#379A8B')) +
    scale_fill_manual(values = c('#DB444B','#006BA2','#3EBCD2','#379A8B')) +
    facet_wrap(~model, nrow = 1) +
    labs(
      x = "Mean predicted probability",
      y = "Observed turnover rate",
      colour = "Model",
      title = paste0("Calibration curves", title_suffix, " (binned probabilities)")
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 8),
      panel.grid = element_blank(),
      axis.ticks = element_blank()
    )
}


plot_confusion_matrices <- function(cv_preds, threshold = 0.5, title_suffix = "") {
  preds <- cv_preds %>%
    mutate(
      truth      = factor(truth, levels = c("no_turnover", "turnover")),
      pred_class = factor(ifelse(prob >= threshold, "turnover", "no_turnover"),
                          levels = c("no_turnover", "turnover"))
    )
  
  cm_df <- preds %>%
    group_by(model) %>%
    group_modify(~ {
      cm <- conf_mat(data = .x, truth = truth, estimate = pred_class)
      tibble(cm = list(cm))
    }) %>%
    ungroup() %>%
    mutate(table = purrr::map(cm, ~ as.data.frame(.x$table))) %>%
    select(model, table) %>%
    unnest(table) %>%
    group_by(model) %>%
    mutate(Percent = Freq / sum(Freq) * 100) %>%
    ungroup() %>%
    mutate(Label = paste0(Freq, "\n(", sprintf("%.1f%%", Percent), ")"))
  
  cm_df$model <- factor(
    cm_df$model,
    levels = c("Mixed-effects logistic (glmer)",
               "Penalised logistic (glmnet)",
               "Random forest (ranger)",
               "Gradient boosting (xgboost)")
  )
  
  ggplot(cm_df, aes(x = Prediction, y = Truth, fill = Freq)) +
    geom_tile(show.legend = FALSE) +
    geom_text(aes(label = Label), colour = "black", size = 4) +
    scale_fill_gradient(low = "#FFB200", high = "#EB5B00") +
    facet_wrap(~model, nrow = 1) +
    coord_fixed(ratio = 1) +
    labs(
      title = paste0("Confusion Matrices at Threshold = ", threshold, title_suffix)
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 8),
      panel.grid = element_blank(),
      axis.ticks = element_blank()
    )
}

############################################################
# 4. SHAP values for XGBoost (subsampled, fewer sims)
############################################################

compute_xgb_shap <- function(fitted_xgb,
                             data,
                             nsim = 20,
                             max_rows = 1000) {
  
  booster <- fitted_xgb$fit$fit$fit
  if (!inherits(booster, "xgb.Booster")) {
    stop("Could not extract xgboost booster from tidymodels workflow.")
  }
  
  # SUBSAMPLE rows to speed things up
  if (nrow(data) > max_rows) {
    set.seed(123)
    data <- data[sample(seq_len(nrow(data)), max_rows), , drop = FALSE]
  }
  
  pred_fun <- function(object, newdata) {
    m <- as.matrix(newdata)
    raw_pred <- predict(object, m)
    prob <- 1 / (1 + exp(-raw_pred))
    as.numeric(prob)
  }
  
  fastshap::explain(
    object       = booster,
    X            = data,
    pred_wrapper = pred_fun,
    nsim         = nsim
  )
}

plot_xgb_shap_summary <- function(shap_vals, baked_data, top_n = 20) {
  
  shap_df <- as.data.frame(shap_vals)
  shap_df$row_id <- seq_len(nrow(shap_df))
  
  baked_predictors <- baked_data %>%
    select(where(is.numeric)) %>%
    mutate(row_id = row_number())
  
  shap_long <- shap_df %>%
    pivot_longer(
      cols = -row_id,
      names_to = "feature",
      values_to = "shap"
    )
  
  data_long <- baked_predictors %>%
    pivot_longer(
      cols = -row_id,
      names_to = "feature",
      values_to = "value"
    )
  
  shap_full <- left_join(shap_long, data_long, by = c("row_id", "feature"))
  
  shap_importance <- shap_full %>%
    group_by(feature) %>%
    summarise(mean_abs_shap = mean(abs(shap)), .groups = "drop") %>%
    arrange(desc(mean_abs_shap)) %>%
    slice(1:top_n)
  
  ggplot(
    shap_full %>% filter(feature %in% shap_importance$feature),
    aes(x = shap, y = reorder(feature, abs(shap)), color = value)
  ) +
    geom_jitter(width = 0, height = 0.2, alpha = 0.7, size = 2) +
    scale_color_viridis_c(option = "plasma") +
    labs(
      title = "XGBoost SHAP Summary",
      x = "SHAP value",
      y = "Feature"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      axis.ticks = element_blank(),
      legend.title = element_blank(),
      axis.title.y = element_blank(),
      axis.text.y = element_text(size = 8)
    ) 
}

compute_pdp <- function(fitted_model,
                        data,
                        feature,
                        ptypes,
                        grid_size = 10,
                        sample_rows = 1000) {
  
  if (!feature %in% names(ptypes)) {
    stop(paste0("Feature '", feature, "' not in recipe predictors."))
  }
  
  rng <- range(data[[feature]], na.rm = TRUE)
  
  if (is.integer(ptypes[[feature]])) {
    grid_vals <- seq(from = rng[1], to = rng[2], length.out = grid_size)
    grid_vals <- unique(as.integer(round(grid_vals)))
  } else if (is.numeric(ptypes[[feature]])) {
    grid_vals <- seq(from = rng[1], to = rng[2], length.out = grid_size)
  } else {
    stop("PDP only implemented for numeric/integer predictors.")
  }
  
  pdp_df <- lapply(grid_vals, function(val) {
    # SAMPLE rows instead of using full dataset
    if (nrow(data) > sample_rows) {
      set.seed(123)
      newdata <- data[sample(seq_len(nrow(data)), sample_rows), , drop = FALSE]
    } else {
      newdata <- data
    }
    
    newdata[[feature]] <- val
    
    prob <- predict(fitted_model, new_data = newdata, type = "prob")$.pred_turnover
    
    tibble(
      value     = as.numeric(val),
      mean_prob = mean(prob)
    )
  }) %>% bind_rows()
  
  pdp_df
}

plot_pdp_features <- function(fitted_model,
                              data,
                              features,
                              grid_size = 10,
                              sample_rows = 1000) {
  
  mold   <- workflows::extract_mold(fitted_model)
  ptypes <- mold$blueprint$ptypes$predictors
  
  pdp_list <- lapply(features, function(f) {
    if (!f %in% names(data)) return(NULL)
    
    df <- compute_pdp(
      fitted_model = fitted_model,
      data         = data,
      feature      = f,
      ptypes       = ptypes,
      grid_size    = grid_size,
      sample_rows  = sample_rows
    )
    df$feature <- f
    df
  })
  
  pdp_all <- bind_rows(pdp_list)
  
  # Order facets: top 4 (options), bottom 3 (pressing counts)
  pdp_all$feature <- factor(
    pdp_all$feature,
    levels = c(
      "left_option", "front_option", "right_option", "back_option",  # top row
      "pressing_count_1", "pressing_count_2", "pressing_count_3"     # bottom row
    )
  )
  
  ggplot(pdp_all, aes(x = value, y = mean_prob, group = feature)) +
    geom_line(linewidth = 1, color = '#006BA2') +
    geom_point(shape = 21, colour = "white", fill = '#006BA2', size = 2.5, stroke = 1) +
    facet_wrap(~feature, scales = "free_x", nrow = 2) +
    theme_minimal(base_size = 14) +
    labs(
      title = "XGBoost Partial Dependence",
      x = "Feature value",
      y = "Predicted prob (mean)"
    ) +
    theme(
      strip.text = element_text(size = 8),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_blank(),
      axis.text.x = element_text(size = 8),
      axis.text.y = element_text(size = 8)
    )
}


############################################################
# 3. Main Evaluation Wrapper
############################################################

run_turnover_evaluation_suite <- function(results,
                                          shap_nsim       = 20,
                                          shap_top_n      = 10,
                                          shap_max_rows   = 1000,
                                          pdp_features    = c("pressing_count_1",
                                                              "pressing_count_2",
                                                              "pressing_count_3",
                                                              "front_option",
                                                              "right_option",
                                                              "left_option",
                                                              "back_option"),
                                          pdp_grid_size   = 10,
                                          pdp_sample_rows = 1000,
                                          skip_shap       = FALSE,
                                          skip_pdp        = FALSE) {
  
  cv_preds <- extract_cv_predictions(results)
  roc_plot   <- plot_turnover_roc(cv_preds)
  calib_plot <- plot_turnover_calibration(cv_preds)
  cm_plot    <- plot_confusion_matrices(cv_preds)
  
  # --- Detect existing fits ---
  if (!is.null(results$rf_results$final_fit) && !is.null(results$xgb_results$final_fit)) {
    message("✅ Using existing fitted models from results object...")
    rf_fit  <- results$rf_results$final_fit
    xgb_fit <- results$xgb_results$final_fit
    rec     <- results$recipe_used
  } else {
    message("⚙️  Fitting final RF and XGB models (not found in results)...")
    final_models <- fit_final_rf_xgb(results)
    rf_fit  <- final_models$rf_fit
    xgb_fit <- final_models$xgb_fit
    rec     <- final_models$recipe
  }
  
  prep_rec <- prep(rec)
  baked_data <- bake(prep_rec, results$model_data)
  baked_predictors <- baked_data %>% select(where(is.numeric))
  
  # --- SHAP ---
  xgb_shap_plot <- NULL
  if (!skip_shap) {
    xgb_shap_vals <- compute_xgb_shap(
      fitted_xgb = xgb_fit,
      data       = baked_predictors,
      nsim       = shap_nsim,
      max_rows   = shap_max_rows
    )
    xgb_shap_plot <- plot_xgb_shap_summary(
      shap_vals  = xgb_shap_vals,
      baked_data = baked_predictors,
      top_n      = shap_top_n
    )
  } else {
    message("⏩ Skipping SHAP computation for speed.")
  }
  
  # --- PDP ---
  xgb_pdp_plot <- NULL
  if (!skip_pdp) {
    xgb_pdp_plot <- plot_pdp_features(
      fitted_model = xgb_fit,
      data         = results$model_data,
      features     = pdp_features,
      grid_size    = pdp_grid_size,
      sample_rows  = pdp_sample_rows
    )
  } else {
    message("⏩ Skipping PDP computation.")
  }
  
  list(
    roc_plot         = roc_plot,
    calibration_plot = calib_plot,
    confusion_plot   = cm_plot,
    xgb_shap_plot    = xgb_shap_plot,
    xgb_pdp_plot     = xgb_pdp_plot
  )
}

# --- SHAP ---
xgb_shap_plot <- NULL
xgb_shap_vals <- NULL  # store data


# --- PDP ---
xgb_pdp_plot <- NULL
xgb_pdp_data <- NULL  # store data


# ----------------- RUN + SAVE PLOTS -----------------

plots <- run_turnover_evaluation_suite(
  results,
  skip_shap = args$skip_shap,
  skip_pdp  = args$skip_pdp
)

dir.create(args$output, showWarnings = FALSE, recursive = TRUE)

ggsave(file.path(args$output, "roc_curve.png"), plots$roc_plot, width = 7, height = 5, dpi = 300)
ggsave(file.path(args$output, "calibration_curve.png"), plots$calibration_plot, width = 7, height = 5, dpi = 300)
ggsave(file.path(args$output, "confusion_matrices.png"), plots$confusion_plot, width = 7, height = 5, dpi = 300)

if (!is.null(plots$xgb_shap_plot)) {
  ggsave(file.path(args$output, "shap_summary.png"), plots$xgb_shap_plot, dpi = 300)
  saveRDS(plots$xgb_shap_plot, file.path(args$output, "xgb_shap_plot.rds"))
  cat("Saved SHAP plot object to:", file.path(args$output, "xgb_shap_plot.rds"), "\n")
}

if (!is.null(plots$xgb_pdp_plot)) {
  ggsave(file.path(args$output, "pdp_features.png"), plots$xgb_pdp_plot, dpi = 300)
  saveRDS(plots$xgb_pdp_plot, file.path(args$output, "xgb_pdp_plot.rds"))
  cat("Saved PDP plot object to:", file.path(args$output, "xgb_pdp_plot.rds"), "\n")
}

cat("✅ Evaluation suite completed. Outputs saved to:", args$output, "\n")




############################################################
# END OF EVALUATION SUITE
############################################################
