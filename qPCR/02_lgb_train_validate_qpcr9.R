# 02_lgb_train_validate_qpcr9.R
# LightGBM model for qPCR 9-species panel
# Outputs:
# - lgb_cv_roc.pdf
# - lgb_valid_roc.pdf
# - lgb_valid_test_predictions.xlsx

source("00_common.R")

# -------- Paths (edit if needed) --------
TRAIN_XLSX <- "Train_286.xlsx"
VALID_XLSX <- "Feng_1129.xlsx"

# -------- Read data --------
train_raw <- read_qpcr_table(TRAIN_XLSX, id_col = "SampleID", group_col = "Group", pred_vars = pred_vars)
valid_raw <- read_qpcr_table(VALID_XLSX, id_col = "SampleID", group_col = "Group", pred_vars = pred_vars)

train_df <- train_raw %>% dplyr::select(Group, all_of(pred_vars))
valid_df <- valid_raw %>% dplyr::select(Group, all_of(pred_vars))

# -------- Recipe (minimal) --------
rec <- recipe(Group ~ ., data = train_df) %>%
  step_zv(all_predictors()) %>%
  prep()

train_baked <- bake(rec, new_data = train_df)
valid_baked <- bake(rec, new_data = valid_df)

# -------- Model spec --------
# Requires: bonsai + lightgbm
suppressPackageStartupMessages({
  library(bonsai)
  library(lightgbm)
})

lgb_spec <- boost_tree(
  mode = "classification",
  engine = "lightgbm",
  trees = 1000,
  mtry = min(6, length(pred_vars)),
  min_n = 10,
  learn_rate = 0.05,
  tree_depth = 6
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

wf <- workflow() %>% add_model(lgb_spec) %>% add_formula(Group ~ .)

# -------- CV ROC --------
cv_out <- cv_fit_and_predict(wf, train_baked, v = 5, repeats = 1, strata_col = "Group")
prob_col <- paste0(".pred_", POS_CLASS)

cv_pred <- cv_out$pred %>%
  mutate(Group = factor(Group, levels = c(POS_CLASS, NEG_CLASS)))

plot_roc_with_ci(
  df_prob = cv_pred,
  truth_col = "Group",
  prob_col = prob_col,
  title = "LightGBM CV ROC (qPCR 9-species)",
  out_pdf = paste0(format(Sys.Date(), "%Y%m%d"), ".lgb.cvROC.pdf")
)

# -------- Fit final model --------
fit_final <- fit(wf, data = train_baked)

# -------- Validation ROC + predictions --------
pred_valid_prob <- predict(fit_final, new_data = valid_baked, type = "prob") %>%
  pull(!!sym(prob_col))

valid_prob_df <- valid_baked %>%
  mutate(Group = factor(Group, levels = c(POS_CLASS, NEG_CLASS))) %>%
  bind_cols(tibble(!!prob_col := pred_valid_prob))

plot_roc_with_ci(
  df_prob = valid_prob_df,
  truth_col = "Group",
  prob_col = prob_col,
  title = "LightGBM Validation ROC (qPCR 9-species)",
  out_pdf = paste0(format(Sys.Date(), "%Y%m%d"), ".lgb.validROC.pdf")
)

export_validation_predictions(
  valid_df = valid_raw,
  prob_pos = pred_valid_prob,
  out_xlsx = paste0(format(Sys.Date(), "%Y%m%d"), ".lgb.valid_test_predictions.xlsx"),
  id_col = "SampleID",
  truth_col = "Group",
  threshold = THRESHOLD
)

