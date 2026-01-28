# 02_lgb_train_validate.R
# LightGBM top-K training + external validation (fixed threshold = 0.6)

packages <- c(
  "tidyverse", "tidymodels", "bonsai", "lightgbm",
  "readxl", "stringi", "openxlsx", "rlang"
)
invisible(lapply(setdiff(packages, rownames(installed.packages())), install.packages))
invisible(lapply(packages, library, character.only = TRUE))

set.seed(42)

# ================== User params ==================
THRESHOLD   <- 0.6
KS_TARGET   <- c(9, 50, 100)

EXT_SETS <- list(
  VAR  = list(group_xlsx = "Group_Vad.xlsx",  abund_xlsx = "9479_abundance.xlsx"),
  FENG = list(group_xlsx = "Group_Screen.xlsx", abund_xlsx = "9479_abundance.xlsx")
)

# ================== Helpers ==================
normalize_id <- function(x) {
  x %>%
    stringr::str_replace_all("\\u00A0", " ") %>%
    stringr::str_squish() %>%
    stringi::stri_trans_general("Fullwidth-Halfwidth")
}

load_external_set <- function(group_xlsx, abund_xlsx, POS_CLASS, NEG_CLASS,
                              sheet_group = 1, sheet_abund = 1) {
  group_df <- readxl::read_excel(group_xlsx, sheet = sheet_group, col_types = "text") %>%
    transmute(SampleID = normalize_id(SampleID), Group = Group)
  
  abund_df <- readxl::read_excel(abund_xlsx, sheet = sheet_abund, col_types = "text") %>%
    mutate(SampleID = normalize_id(SampleID))
  
  species_cols <- setdiff(colnames(abund_df), "SampleID")
  abund_df <- abund_df %>%
    mutate(across(all_of(species_cols), ~ suppressWarnings(as.numeric(.)))) %>%
    mutate(across(all_of(species_cols), ~ tidyr::replace_na(., 0)))
  
  dat <- group_df %>%
    inner_join(abund_df, by = "SampleID") %>%
    distinct() %>%
    tidyr::drop_na(Group)
  
  dat$Group <- factor(dat$Group, levels = c(POS_CLASS, NEG_CLASS))
  dat
}

metrics_at_threshold <- function(df_prob, truth_col, prob_col, threshold, pos, neg) {
  df_cls <- df_prob %>%
    mutate(
      .pred_class = factor(ifelse(.data[[prob_col]] >= threshold, pos, neg),
                           levels = c(pos, neg))
    )
  
  tibble(
    threshold = threshold,
    sens = yardstick::sens(df_cls, truth = !!sym(truth_col), estimate = .pred_class, event_level = "first")$.estimate,
    spec = yardstick::spec(df_cls, truth = !!sym(truth_col), estimate = .pred_class, event_level = "first")$.estimate,
    acc  = yardstick::accuracy(df_cls, truth = !!sym(truth_col), estimate = .pred_class)$.estimate,
    ppv  = yardstick::ppv(df_cls, truth = !!sym(truth_col), estimate = .pred_class, event_level = "first")$.estimate,
    npv  = yardstick::npv(df_cls, truth = !!sym(truth_col), estimate = .pred_class, event_level = "first")$.estimate
  )
}

# ================== Load training data ==================
# Training set files (edit as needed)
TRAIN_GROUP_XLSX  <- "Group-Train.xlsx"
TRAIN_GROUP_SHEET <- 1
TRAIN_ABUND_XLSX  <- "9479_abundance.xlsx"
TRAIN_ABUND_SHEET <- 1

# Ranking file (feature importance)
RANK_XLSX  <- "rfe_importance.xlsx"
RANK_SHEET <- 1

# Load training labels
train_group_df <- readxl::read_excel(TRAIN_GROUP_XLSX, sheet = TRAIN_GROUP_SHEET, col_types = "text") %>%
  transmute(SampleID = normalize_id(SampleID), Group = Group)

# Load training abundance
train_abund_df <- readxl::read_excel(TRAIN_ABUND_XLSX, sheet = TRAIN_ABUND_SHEET, col_types = "text") %>%
  mutate(SampleID = normalize_id(SampleID))

stopifnot("SampleID" %in% colnames(train_abund_df))
train_species_cols <- setdiff(colnames(train_abund_df), "SampleID")

train_abund_df <- train_abund_df %>%
  mutate(across(all_of(train_species_cols), ~ suppressWarnings(as.numeric(.)))) %>%
  mutate(across(all_of(train_species_cols), ~ tidyr::replace_na(., 0)))

# Merge training set
dat <- train_group_df %>%
  inner_join(train_abund_df, by = "SampleID") %>%
  distinct() %>%
  tidyr::drop_na(Group)

# Set class levels
neg_classes <- setdiff(unique(dat$Group), POS_CLASS)
if (length(neg_classes) != 1) stop("Binary classification only: check labels and POS_CLASS.")
NEG_CLASS <- as.character(neg_classes)

dat$Group <- factor(dat$Group, levels = c(POS_CLASS, NEG_CLASS))

# ================== Load ranking and build rf_rank ==================
species_importance <- readxl::read_excel(RANK_XLSX, sheet = RANK_SHEET)

if (all(c("var", "Mean_Importance") %in% colnames(species_importance))) {
  rf_rank <- species_importance %>%
    filter(!is.na(var)) %>%
    arrange(desc(Mean_Importance)) %>%
    pull(var)
} else if (all(c("var", "importance") %in% colnames(species_importance))) {
  rf_rank <- species_importance %>%
    filter(!is.na(var)) %>%
    arrange(desc(importance)) %>%
    pull(var)
} else {
  stop("Ranking file must contain 'var' and either 'Mean_Importance' or 'importance'.")
}

# Keep only ranked features that exist in training abundance columns
rf_rank <- intersect(rf_rank, train_species_cols)
if (length(rf_rank) < max(KS_TARGET)) {
  stop("Not enough ranked features available for KS_TARGET after intersecting with training columns.")
}


# ================== Tune per-K ==================
tune_lgb_for_k <- function(K, dat, rf_rank) {
  sel_vars <- rf_rank[1:K]
  train_k  <- dat %>% select(Group, all_of(sel_vars))
  
  rec <- recipe(Group ~ ., data = train_k) %>%
    step_zv(all_predictors()) %>%
    prep()
  
  train_baked <- bake(rec, new_data = NULL)
  
  folds <- vfold_cv(train_baked, v = 5, strata = Group)
  metrics_set <- metric_set(roc_auc)
  
  model_lgb <- boost_tree(
    mode = "classification",
    engine = "lightgbm",
    trees = 2000,
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune(),
    sample_size = tune()
  ) %>%
    set_engine(
      "lightgbm",
      objective = "binary",
      metric = "auc",
      boosting = "gbdt",
      early_stopping_round = 50,
      num_threads = max(1, parallel::detectCores() - 1),
      verbosity = -1
    )
  
  wf <- workflow() %>%
    add_model(model_lgb) %>%
    add_formula(Group ~ .)
  
  hp <- parameters(
    finalize(mtry(), train_baked),
    min_n(range = c(5, 50)),
    tree_depth(range = c(2, 10)),
    learn_rate(range = c(-3, -1)),
    loss_reduction(range = c(-3, 0)),
    sample_prop(range = c(0.6, 1.0))
  )
  
  grid <- grid_random(hp, size = 20)
  
  tuned <- tune_grid(
    wf,
    resamples = folds,
    grid = grid,
    metrics = metrics_set,
    control = control_grid(save_pred = TRUE, verbose = TRUE)
  )
  
  best_params <- select_best(tuned, metric = "roc_auc")
  best_auc    <- show_best(tuned, metric = "roc_auc", n = 1)
  
  list(
    K = K,
    sel_vars = sel_vars,
    recipe = rec,
    best_params = best_params,
    best_auc = best_auc
  )
}

fit_final_lgb <- function(dat, sel_vars, rec, best_params) {
  train_best <- dat %>% select(Group, all_of(sel_vars))
  train_baked <- bake(rec, new_data = train_best)
  
  model_final <- boost_tree(
    mode = "classification",
    engine = "lightgbm",
    trees = 2000,
    mtry = best_params$mtry,
    min_n = best_params$min_n,
    tree_depth = best_params$tree_depth,
    learn_rate = best_params$learn_rate,
    loss_reduction = best_params$loss_reduction,
    sample_size = best_params$sample_size
  ) %>%
    set_engine(
      "lightgbm",
      objective = "binary",
      metric = "auc",
      boosting = "gbdt",
      early_stopping_round = 50,
      num_threads = max(1, parallel::detectCores() - 1),
      verbosity = -1
    )
  
  wf <- workflow() %>% add_model(model_final) %>% add_formula(Group ~ .)
  fit(wf, train_baked)
}

# ================== Run tuning + training eval ==================
all_results <- lapply(KS_TARGET, function(K) tune_lgb_for_k(K, dat, rf_rank))

summary_tbl <- purrr::map_dfr(all_results, function(res) {
  tibble(
    K = res$K,
    cv_mean_auc = res$best_auc$mean,
    cv_std_err  = res$best_auc$std_err
  )
})
openxlsx::write.xlsx(summary_tbl, "lgb_cv_auc_summary.xlsx", asTable = TRUE)

train_eval <- list()

for (res in all_results) {
  K <- res$K
  fit_final <- fit_final_lgb(dat, res$sel_vars, res$recipe, res$best_params)
  
  train_best <- dat %>% select(Group, all_of(res$sel_vars))
  train_baked <- bake(res$recipe, new_data = train_best)
  
  prob_col <- paste0(".pred_", POS_CLASS)
  
  pred_train <- predict(fit_final, new_data = train_baked, type = "prob") %>%
    bind_cols(train_baked %>% select(Group))
  
  auc_train <- roc_auc(pred_train, truth = Group, !!sym(prob_col), event_level = "first")$.estimate
  met_thr   <- metrics_at_threshold(pred_train, "Group", prob_col, THRESHOLD, POS_CLASS, NEG_CLASS)
  
  train_eval[[as.character(K)]] <- tibble(
    dataset = "train",
    K = K,
    auc = auc_train
  ) %>% bind_cols(met_thr)
  
  saveRDS(fit_final, file = paste0("lgb_model_K", K, ".rds"))
}

train_eval_tbl <- bind_rows(train_eval)
openxlsx::write.xlsx(train_eval_tbl, "lgb_train_eval_threshold_0p6.xlsx", asTable = TRUE)

# ================== External validation ==================
ext_eval <- list()

for (nm in names(EXT_SETS)) {
  ext <- EXT_SETS[[nm]]
  test_df <- load_external_set(ext$group_xlsx, ext$abund_xlsx, POS_CLASS, NEG_CLASS)
  
  for (res in all_results) {
    K <- res$K
    fit_final <- readRDS(paste0("lgb_model_K", K, ".rds"))
    
    test_best <- test_df %>% select(Group)
    
    miss <- setdiff(res$sel_vars, colnames(test_df))
    for (v in miss) test_best[[v]] <- 0
    
    # Add existing columns in the correct order
    for (v in res$sel_vars) {
      if (v %in% colnames(test_df)) test_best[[v]] <- test_df[[v]]
    }
    
    test_best <- test_best %>% select(Group, all_of(res$sel_vars))
    
    test_baked <- bake(res$recipe, new_data = test_best)
    
    prob_col <- paste0(".pred_", POS_CLASS)
    pred_test <- predict(fit_final, new_data = test_baked, type = "prob") %>%
      bind_cols(test_baked %>% select(Group))
    
    auc_test <- roc_auc(pred_test, truth = Group, !!sym(prob_col), event_level = "first")$.estimate
    met_thr  <- metrics_at_threshold(pred_test, "Group", prob_col, THRESHOLD, POS_CLASS, NEG_CLASS)
    
    ext_eval[[paste(nm, K, sep = "_")]] <- tibble(
      dataset = nm,
      K = K,
      auc = auc_test
    ) %>% bind_cols(met_thr)
    
    pred_export <- pred_test %>%
      mutate(
        SampleID = test_df$SampleID,
        Pred_Prob_POS = .data[[prob_col]],
        Pred_Label = ifelse(.data[[prob_col]] >= THRESHOLD, POS_CLASS, NEG_CLASS)
      ) %>%
      relocate(SampleID, Group, Pred_Prob_POS, Pred_Label)
    
    openxlsx::write.xlsx(
      pred_export,
      paste0("lgb_", nm, "_K", K, "_predictions_threshold_0p6.xlsx"),
      asTable = TRUE
    )
  }
}

ext_eval_tbl <- bind_rows(ext_eval)
openxlsx::write.xlsx(ext_eval_tbl, "lgb_external_eval_threshold_0p6.xlsx", asTable = TRUE)
