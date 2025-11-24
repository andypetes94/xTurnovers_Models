############################################################
#  TURNOVER MODELING — FULL FUNCTION SCRIPT (PARALLELIZED)
#  Now supports two model types via `model_type`:
#    - "default": full predictor set
#    - "alt": same, but drops:
#        distance_ball_moved,
#        ball_movement_speed,
#        percent_distance,
#        pass.angle
############################################################

# =========================================================
# 0. PACKAGES
# =========================================================

suppressPackageStartupMessages({
  library(lme4)
  library(dplyr)
  library(pROC)
  library(tidymodels)
  library(ranger)
  library(xgboost)
  library(argparse)   # used by wrapper scripts, safe to load
  
  # === PARALLEL PACKAGES ===
  library(doParallel)
  library(future)
  library(future.apply)
  
  tidymodels_prefer()
})

# =========================================================
# 0a. GLOBAL PARALLEL SETUP
# =========================================================

#n_cores <- parallel::detectCores()
n_workers <- 2L
# if (is.na(n_cores) || n_cores < 2) {
#   n_workers <- 1L
# } else {
#   n_workers <- max(1L, n_cores - 1L)
# }

if (n_workers > 1L) {
  message("Using ", n_workers, " parallel workers.")
  cl <- parallel::makeCluster(n_workers)
  doParallel::registerDoParallel(cl)
  future::plan(future::multicore, workers = n_workers)
} else {
  message("Parallelism disabled (only 1 core detected).")
  future::plan(future::sequential)
}

# =========================================================
# 0b. PREDICTOR SETS (DEFAULT vs ALT)
# =========================================================

get_predictor_vars <- function(model_type = c("default", "alt")) {
  model_type <- match.arg(model_type)
  
  default_predictors <- c(
    "x", "y",
    "distance_ball_moved", "ball_movement_speed",
    "percent_distance", "pass.angle",
    "pressing_count_1", "pressing_count_2", "pressing_count_3",
    "play_pattern.id", "pass.type.name",
    "right_option", "front_option", "left_option", "back_option",
    "player.id", "match_id", "position_group.id"
  )
  
  if (model_type == "default") {
    return(default_predictors)
  }
  
  # ALT MODEL: DROP 4 MOVEMENT VARIABLES
  alt_predictors <- setdiff(
    default_predictors,
    c("distance_ball_moved",
      "ball_movement_speed",
      "percent_distance",
      "pass.angle")
  )
  alt_predictors
}

get_numeric_vars <- function(model_type = c("default", "alt")) {
  model_type <- match.arg(model_type)
  
  default_numeric <- c(
    "x", "y",
    "distance_ball_moved", "ball_movement_speed",
    "percent_distance", "pass.angle",
    "pressing_count_1", "pressing_count_2", "pressing_count_3"
  )
  
  if (model_type == "default") {
    return(default_numeric)
  }
  
  # ALT MODEL: remove those 4 numeric movement vars
  alt_numeric <- setdiff(
    default_numeric,
    c("distance_ball_moved",
      "ball_movement_speed",
      "percent_distance",
      "pass.angle")
  )
  alt_numeric
}

# =========================================================
# 1. DATA PREPARATION FUNCTION
# =========================================================

prepare_turnover_data <- function(raw_data,
                                  outcome_var = "turnover_count",
                                  model_type = c("default", "alt")) {
  
  model_type <- match.arg(model_type)
  
  predictors <- get_predictor_vars(model_type)
  required_model_vars <- c(outcome_var, predictors)
  
  missing_required <- setdiff(required_model_vars, names(raw_data))
  if (length(missing_required) > 0) {
    stop("Missing required variables: ", paste(missing_required, collapse = ", "))
  }
  
  model_data <- raw_data %>%
    dplyr::select(all_of(required_model_vars)) %>%
    na.omit()
  
  model_data <- model_data %>%
    mutate(
      !!outcome_var := as.numeric(.data[[outcome_var]]),
      turnover_factor = factor(.data[[outcome_var]],
                               levels = c(0, 1),
                               labels = c("no_turnover", "turnover"))
    )
  
  cat("Model type:", model_type, "\n")
  cat("Total observations:", nrow(model_data), "\n")
  print(table(model_data[[outcome_var]]))
  
  model_data
}

# =========================================================
# 1a. SUBSAMPLING FOR GLMER
# =========================================================

subsample_for_glmer <- function(model_data,
                                outcome_var = "turnover_count",
                                max_n = 40000,
                                seed = 123) {
  set.seed(seed)
  
  n <- nrow(model_data)
  if (n <= max_n) {
    cat("subsample_for_glmer: n <=", max_n, "→ using full data for glmer.\n")
    return(model_data)
  }
  
  n_classes <- length(unique(model_data[[outcome_var]]))
  per_class <- max_n / n_classes
  
  cat("subsample_for_glmer: total n =", n,
      "→ sampling ~", max_n, "rows for glmer (≈",
      round(per_class), "per class).\n")
  
  model_data %>%
    group_by(across(all_of(outcome_var))) %>%
    group_modify(~ {
      slice_sample(.x, n = min(ceiling(per_class), nrow(.x)), replace = FALSE)
    }) %>%
    ungroup()
}



# =========================================================
# 2. GROUPED CROSS-VALIDATION FUNCTION
# =========================================================

create_grouped_folds <- function(model_data,
                                 v = 5,
                                 group_var = "match_id",
                                 outcome_var = "turnover_count",
                                 seed = 123) {
  set.seed(seed)
  
  n_groups <- length(unique(model_data[[group_var]]))
  cat("Number of unique", group_var, ":", n_groups, "\n")
  
  if (n_groups == 1) {
    cat("Only 1 group found → using stratified v-fold CV.\n")
    folds <- vfold_cv(
      model_data,
      v = min(v, nrow(model_data)),
      strata = all_of(outcome_var)
    )
  } else {
    v_folds <- min(v, n_groups)
    cat("Using v =", v_folds, "grouped CV by", group_var, "\n")
    folds <- group_vfold_cv(
      model_data,
      group = !!sym(group_var),
      v = v_folds
    )
  }
  
  folds
}

# =========================================================
# 3. MIXED-EFFECTS LOGISTIC REGRESSION CV (PARALLEL)
# =========================================================

run_glmer_grouped_cv <- function(model_data,
                                 folds,
                                 outcome_var = "turnover_count",
                                 model_type = c("default", "alt"),
                                 optimizer = "bobyqa",
                                 max_iterations = 2e5,
                                 seed = 123) {
  set.seed(seed)
  model_type <- match.arg(model_type)
  
  numeric_vars <- get_numeric_vars(model_type)
  
  split_indices <- seq_along(folds$splits)
  
  fold_results_list <- future.apply::future_lapply(split_indices, function(i) {
    cat("Fitting glmer fold", i, "...\n")
    
    split_i <- folds$splits[[i]]
    train_data <- rsample::analysis(split_i)
    test_data  <- rsample::assessment(split_i)
    
    train_scaled <- train_data
    test_scaled  <- test_data
    
    # Scale numeric vars
    for (v in numeric_vars) {
      m <- mean(train_data[[v]], na.rm = TRUE)
      s <- sd(train_data[[v]], na.rm = TRUE)
      if (s == 0) s <- 1
      
      train_scaled[[paste0(v, "_scaled")]] <- (train_data[[v]] - m) / s
      test_scaled[[paste0(v, "_scaled")]]  <- (test_data[[v]]  - m) / s
    }
    
    scaled_predictors <- paste0(numeric_vars, "_scaled")
    factor_predictors <- c(
      "as.factor(play_pattern.id)", "as.factor(pass.type.name)",
      "as.factor(right_option)", "as.factor(front_option)",
      "as.factor(left_option)", "as.factor(back_option)"
    )
    all_predictors <- c(scaled_predictors, factor_predictors)
    
    candidate_re <- c("player.id", "match_id", "position_group.id")
    valid_re <- candidate_re[
      sapply(candidate_re, function(v)
        length(unique(train_scaled[[v]])) > 1
      )
    ]
    
    if (length(valid_re) == 0) {
      warning(paste(
        "No random effects with ≥2 levels in fold", i,
        "- skipping glmer for this fold."
      ))
      return(NULL)
    }
    
    random_effects <- paste0("(1 | ", valid_re, ")", collapse = " + ")
    
    model_formula <- as.formula(
      paste(outcome_var, "~", paste(all_predictors, collapse = " + "), "+", random_effects)
    )
    
    fit <- suppressWarnings(
      glmer(
        formula = model_formula,
        data = train_scaled,
        family = "binomial",
        control = glmerControl(
          optimizer = optimizer,
          optCtrl = list(maxfun = max_iterations)
        )
      )
    )
    
    probs <- predict(fit, newdata = test_scaled, type = "response", allow.new.levels = TRUE)
    preds <- ifelse(probs > 0.5, 1, 0)
    actual <- test_data[[outcome_var]]
    
    accuracy <- mean(preds == actual)
    tp <- sum(preds == 1 & actual == 1)
    tn <- sum(preds == 0 & actual == 0)
    fp <- sum(preds == 1 & actual == 0)
    fn <- sum(preds == 0 & actual == 1)
    
    precision <- ifelse(tp + fp > 0, tp / (tp + fp), NA)
    recall    <- ifelse(tp + fn > 0, tp / (tp + fn), NA)
    specificity <- ifelse(tn + fp > 0, tn / (tn + fp), NA)
    f1 <- ifelse(!is.na(precision + recall) && precision + recall > 0,
                 2 * precision * recall / (precision + recall),
                 NA)
    
    auc_val <- tryCatch(as.numeric(pROC::auc(actual, probs)), error = function(e) NA)
    brier <- mean((probs - actual)^2)
    
    fold_metrics <- data.frame(
      fold = i,
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      specificity = specificity,
      f1 = f1,
      auc = auc_val,
      brier_class = brier
    )
    
    predictions <- data.frame(
      fold = i, actual = actual, prob = probs, pred = preds
    )
    
    list(
      fold_metrics = fold_metrics,
      predictions  = predictions
    )
  })
  
  non_null <- Filter(Negate(is.null), fold_results_list)
  
  if (length(non_null) == 0) {
    stop("glmer CV failed: no folds could be fitted with valid random effects.")
  }
  
  fold_df <- bind_rows(lapply(non_null, `[[`, "fold_metrics"))
  all_pred <- bind_rows(lapply(non_null, `[[`, "predictions"))
  
  summary_row <- fold_df %>%
    summarise(
      Model       = "Mixed-effects logistic (glmer)",
      roc_auc     = mean(auc, na.rm = TRUE),
      accuracy    = mean(accuracy, na.rm = TRUE),
      sens        = mean(recall, na.rm = TRUE),
      spec        = mean(specificity, na.rm = TRUE),
      f_meas      = mean(f1, na.rm = TRUE),
      brier_class = mean(brier_class, na.rm = TRUE)
    )
  
  list(
    fold_metrics = fold_df,
    summary      = summary_row,
    predictions  = all_pred
  )
}

# =========================================================
# 4. RECIPE BUILDERS + METRICS FUNCTION
# =========================================================
#  - glmnet: one-hot, no player/match IDs
#  - RF/XGB: factors handled, no player/match IDs

build_turnover_recipe <- function(model_data,
                                  model_type = c("default", "alt")) {
  model_type <- match.arg(model_type)
  
  predictors <- get_predictor_vars(model_type)
  predictors <- setdiff(predictors, c("player.id", "match_id"))  # drop IDs
  
  formula_all <- as.formula(
    paste("turnover_factor ~", paste(predictors, collapse = " + "))
  )
  
  recipe(formula_all, data = model_data) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_novel(all_nominal_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE)
}

build_turnover_recipe_glmnet <- function(model_data,
                                         model_type = c("default", "alt")) {
  model_type <- match.arg(model_type)
  
  predictors <- get_predictor_vars(model_type)
  predictors <- setdiff(predictors, c("player.id", "match_id"))  # no IDs
  
  formula_glmnet <- as.formula(
    paste("turnover_factor ~", paste(predictors, collapse = " + "))
  )
  
  recipe(formula_glmnet, data = model_data) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_novel(all_nominal_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE)
}

build_turnover_recipe_rf <- function(model_data,
                                     model_type = c("default", "alt")) {
  model_type <- match.arg(model_type)
  
  predictors <- get_predictor_vars(model_type)
  predictors <- setdiff(predictors, c("player.id", "match_id"))
  
  formula_rf <- as.formula(
    paste("turnover_factor ~", paste(predictors, collapse = " + "))
  )
  
  recipe(formula_rf, data = model_data) %>%
    step_zv(all_predictors()) %>%
    step_unknown(all_nominal_predictors()) %>%
    step_other(all_nominal_predictors(), threshold = 0.01) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_mutate_at(all_nominal_predictors(), fn = as.factor)
}

build_turnover_recipe_xgb <- function(model_data,
                                      model_type = c("default", "alt")) {
  model_type <- match.arg(model_type)
  
  predictors <- get_predictor_vars(model_type)
  predictors <- setdiff(predictors, c("player.id", "match_id"))
  
  formula_xgb <- as.formula(
    paste("turnover_factor ~", paste(predictors, collapse = " + "))
  )
  
  recipe(formula_xgb, data = model_data) %>%
    step_zv(all_predictors()) %>%
    step_unknown(all_nominal_predictors()) %>%
    step_other(all_nominal_predictors(), threshold = 0.01) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
    step_normalize(all_numeric_predictors())
}

default_turnover_metrics <- function() {
  metric_set(
    roc_auc,
    accuracy,
    sens,
    spec,
    f_meas,
    brier_class
  )
}

# =========================================================
# 5. GLMNET (PENALISED LOGISTIC REGRESSION)
# =========================================================

tune_glmnet_model <- function(turnover_rec,
                              cv_folds,
                              metrics_to_use,
                              grid_size = 20,
                              seed = 123) {
  set.seed(seed)
  
  logit_spec <- logistic_reg(
    mode    = "classification",
    penalty = tune(),
    mixture = tune()
  ) %>%
    set_engine("glmnet")
  
  logit_wf <- workflow() %>%
    add_model(logit_spec) %>%
    add_recipe(turnover_rec)
  
  logit_res <- tune_grid(
    logit_wf,
    resamples = cv_folds,
    grid      = grid_size,
    metrics   = metrics_to_use,
    control   = control_grid(save_pred = TRUE)
  )
  
  logit_best <- select_best(logit_res, metric = "roc_auc")
  
  logit_metrics_best <- logit_res %>%
    collect_metrics() %>%
    filter(.config == logit_best$.config) %>%
    select(.metric, mean) %>%
    tidyr::pivot_wider(names_from = .metric, values_from = mean) %>%
    mutate(Model = "Penalised logistic (glmnet)") %>%
    relocate(Model)
  
  list(
    tune_results = logit_res,
    best_config  = logit_best,
    metrics_wide = logit_metrics_best
  )
}

# =========================================================
# 6. RANDOM FOREST (RANGER)
# =========================================================

tune_rf_model <- function(turnover_rec,
                          model_data,
                          cv_folds,
                          metrics_to_use,
                          grid_size = 20,
                          seed = 123,
                          num_threads = NULL) {
  set.seed(seed)
  
  if (is.null(num_threads)) {
    num_threads <- max(1L, parallel::detectCores() - 1L)
  }
  
  rf_spec <- rand_forest(
    mode  = "classification",
    mtry  = tune(),
    min_n = tune(),
    trees = 1000
  ) %>%
    set_engine("ranger",
               importance   = "impurity",
               num.threads  = num_threads)
  
  rf_wf <- workflow() %>%
    add_model(rf_spec) %>%
    add_recipe(turnover_rec)
  
  prep_rec <- prep(turnover_rec)
  baked    <- bake(prep_rec, model_data)
  num_predictors <- ncol(baked) - 1
  
  cat("Final predictors after preprocessing (RF):", num_predictors, "\n")
  
  rf_params <- extract_parameter_set_dials(rf_spec) %>%
    update(mtry = mtry(c(1, num_predictors)))
  
  rf_res <- tune_grid(
    rf_wf,
    resamples  = cv_folds,
    grid       = grid_size,
    param_info = rf_params,
    metrics    = metrics_to_use,
    control    = control_grid(save_pred = TRUE)
  )
  
  rf_best <- select_best(rf_res, metric = "roc_auc")
  
  rf_metrics_best <- rf_res %>%
    collect_metrics() %>%
    filter(.config == rf_best$.config) %>%
    select(.metric, mean) %>%
    tidyr::pivot_wider(names_from = .metric, values_from = mean) %>%
    mutate(Model = "Random forest (ranger)") %>%
    relocate(Model)
  
  list(
    tune_results = rf_res,
    best_config  = rf_best,
    metrics_wide = rf_metrics_best
  )
}

# =========================================================
# 7. GRADIENT BOOSTING (XGBOOST)
# =========================================================

tune_xgb_model <- function(turnover_rec,
                           cv_folds,
                           metrics_to_use,
                           grid_size = 20,
                           seed = 123,
                           num_threads = NULL) {
  set.seed(seed)
  
  if (is.null(num_threads)) {
    num_threads <- max(1L, parallel::detectCores() - 1L)
  }
  
  xgb_spec <- boost_tree(
    mode = "classification",
    trees = 1000,
    learn_rate     = tune(),
    tree_depth     = tune(),
    min_n          = tune(),
    loss_reduction = tune(),
    sample_size    = tune()
  ) %>%
    set_engine("xgboost", nthread = num_threads)
  
  xgb_wf <- workflow() %>%
    add_model(xgb_spec) %>%
    add_recipe(turnover_rec)
  
  xgb_res <- tune_grid(
    xgb_wf,
    resamples = cv_folds,
    grid      = grid_size,
    metrics   = metrics_to_use,
    control   = control_grid(save_pred = TRUE)
  )
  
  xgb_best <- select_best(xgb_res, metric = "roc_auc")
  
  xgb_metrics_best <- xgb_res %>%
    collect_metrics() %>%
    filter(.config == xgb_best$.config) %>%
    select(.metric, mean) %>%
    tidyr::pivot_wider(names_from = .metric, values_from = mean) %>%
    mutate(Model = "Gradient boosting (xgboost)") %>%
    relocate(Model)
  
  list(
    tune_results = xgb_res,
    best_config  = xgb_best,
    metrics_wide = xgb_metrics_best
  )
}

# =========================================================
# 8. COMBINE MODEL SUMMARIES
# =========================================================

combine_model_comparison <- function(glmer_summary,
                                     glmnet_metrics,
                                     rf_metrics,
                                     xgb_metrics) {
  
  glmer_summary_aligned <- glmer_summary %>%
    rename(
      roc_auc     = roc_auc,
      accuracy    = accuracy,
      sens        = sens,
      spec        = spec,
      f_meas      = f_meas,
      brier_class = brier_class
    )
  
  comparison_table <- bind_rows(
    glmer_summary_aligned,
    glmnet_metrics,
    rf_metrics,
    xgb_metrics
  ) %>%
    mutate(across(where(is.numeric), ~ round(.x, 3)))
  
  comparison_table
}

# =========================================================
# 9. MASTER WRAPPER FUNCTION
# =========================================================

run_turnover_model_suite <- function(raw_data,
                                     outcome_var = "turnover_count",
                                     model_type = c("default", "alt"),
                                     grid_size = 20,
                                     seed = 123,
                                     glmer_max_n = 40000,
                                     glmer_v = 3,
                                     ml_v = 5) {
  
  set.seed(seed)
  model_type <- match.arg(model_type)
  
  # STEP 1: Data prep (full data)
  model_data <- prepare_turnover_data(raw_data, outcome_var, model_type)
  
  # STEP 1a: Subsample for glmer
  glmer_data <- subsample_for_glmer(
    model_data,
    outcome_var = outcome_var,
    max_n       = glmer_max_n,
    seed        = seed
  )
  
  # STEP 2a: CV folds for ML models
  cv_folds_ml <- create_grouped_folds(
    model_data,
    v          = ml_v,
    group_var  = "match_id",
    outcome_var = outcome_var,
    seed       = seed
  )
  
  # STEP 2b: CV folds for glmer
  cv_folds_glmer <- create_grouped_folds(
    glmer_data,
    v          = glmer_v,
    group_var  = "match_id",
    outcome_var = outcome_var,
    seed       = seed
  )
  
  # STEP 3: glmer CV
  glmer_cv <- run_glmer_grouped_cv(
    glmer_data,
    cv_folds_glmer,
    outcome_var = outcome_var,
    model_type  = model_type,
    seed        = seed
  )
  
  # STEP 4: recipes + metrics
  turnover_rec_glmnet <- build_turnover_recipe_glmnet(model_data, model_type)
  turnover_rec_rf     <- build_turnover_recipe_rf(model_data, model_type)
  turnover_rec_xgb    <- build_turnover_recipe_xgb(model_data, model_type)
  
  metrics_to_use      <- default_turnover_metrics()
  
  # STEP 5: ML models
  glmnet_res <- tune_glmnet_model(
    turnover_rec_glmnet, cv_folds_ml, metrics_to_use,
    grid_size = grid_size, seed = seed
  )
  
  rf_res <- tune_rf_model(
    turnover_rec_rf, model_data, cv_folds_ml, metrics_to_use,
    grid_size   = grid_size,
    seed        = seed,
    num_threads = n_workers
  )
  
  xgb_res <- tune_xgb_model(
    turnover_rec_xgb, cv_folds_ml, metrics_to_use,
    grid_size   = grid_size,
    seed        = seed,
    num_threads = n_workers
  )
  
  # STEP 6: Comparison
  comparison <- combine_model_comparison(
    glmer_cv$summary,
    glmnet_res$metrics_wide,
    rf_metrics = rf_res$metrics_wide,
    xgb_metrics = xgb_res$metrics_wide
  )
  
  list(
    model_data        = model_data,
    glmer_data        = glmer_data,
    
    # recipes
    recipe_glmnet     = turnover_rec_glmnet,
    recipe_rf         = turnover_rec_rf,
    recipe_xgb        = turnover_rec_xgb,
    
    # CV folds
    cv_folds_ml       = cv_folds_ml,
    cv_folds_glmer    = cv_folds_glmer,
    
    # model results
    glmer_results     = glmer_cv,
    glmnet_results    = glmnet_res,
    rf_results        = rf_res,
    xgb_results       = xgb_res,
    
    # comparison
    comparison_table  = comparison,
    
    # store model_type + a general recipe for later use
    model_type        = model_type,
    recipe_used       = build_turnover_recipe(model_data, model_type)
  )
}

# =========================================================
# 10. SAVE & AUGMENT RESULTS (with model_type suffix)
# =========================================================

save_and_augment_turnover_results <- function(
    results,
    raw_data,
    file_prefix = "turnover",
    save_dir,
    model_type = c("default", "alt")
) {
  
  model_type <- match.arg(model_type)
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, recursive = TRUE)
  }
  
  suffix <- if (identical(model_type, "alt")) "_alt" else ""
  
  augmented <- raw_data
  
  # 1. GLMER predictions (based on glmer_data subset)
  glmer_pred <- results$glmer_results$predictions
  if (!is.null(glmer_pred) && nrow(glmer_pred) == nrow(augmented)) {
    augmented$xTurnover_glmer <- glmer_pred$prob
    augmented$pred_glmer      <- ifelse(glmer_pred$prob >= 0.5, 1, 0)
  } else {
    augmented$xTurnover_glmer <- NA_real_
    augmented$pred_glmer      <- NA_integer_
  }
  
  # 2. GLMNET predictions
  glmnet_best <- results$glmnet_results$best_config$.config
  
  glmnet_pred <- results$glmnet_results$tune_results %>%
    collect_predictions() %>%
    filter(.config == glmnet_best) %>%
    arrange(.row)
  
  augmented$xTurnover_glmnet <- glmnet_pred$.pred_turnover
  augmented$pred_glmnet      <- ifelse(glmnet_pred$.pred_turnover >= 0.5, 1, 0)
  
  # 3. RANDOM FOREST predictions
  rf_best <- results$rf_results$best_config$.config
  
  rf_pred <- results$rf_results$tune_results %>%
    collect_predictions() %>%
    filter(.config == rf_best) %>%
    arrange(.row)
  
  augmented$xTurnover_rf <- rf_pred$.pred_turnover
  augmented$pred_rf      <- ifelse(rf_pred$.pred_turnover >= 0.5, 1, 0)
  
  # 4. XGBOOST predictions
  xgb_best <- results$xgb_results$best_config$.config
  
  xgb_pred <- results$xgb_results$tune_results %>%
    collect_predictions() %>%
    filter(.config == xgb_best) %>%
    arrange(.row)
  
  augmented$xTurnover_xgb <- xgb_pred$.pred_turnover
  augmented$pred_xgb      <- ifelse(xgb_pred$.pred_turnover >= 0.5, 1, 0)
  
  # 5. Ensemble
  augmented$xTurnover_ensemble <- rowMeans(
    cbind(
      augmented$xTurnover_glmer,
      augmented$xTurnover_glmnet,
      augmented$xTurnover_rf,
      augmented$xTurnover_xgb
    ),
    na.rm = TRUE
  )
  augmented$pred_ensemble <- ifelse(augmented$xTurnover_ensemble >= 0.5, 1, 0)
  
  # 6. Save final augmented dataframe with suffix
  output_csv <- file.path(
    save_dir,
    paste0(file_prefix, suffix, "_augmented_data.csv")
  )
  write.csv(augmented, output_csv, row.names = FALSE)
  
  list(
    augmented_data = augmented,
    augmented_file = output_csv
  )
}


############################################################
# END OF FUNCTION DEFINITIONS
############################################################
