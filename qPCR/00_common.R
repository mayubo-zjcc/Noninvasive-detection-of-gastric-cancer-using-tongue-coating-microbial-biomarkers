# 00_common.R
# Common utilities for qPCR 9-species models (XGB/LGB/RF)
# - CV ROC: pooled out-of-fold predictions + DeLong AUC CI
# - Validation ROC: DeLong AUC CI
# - Export validation predictions

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(readxl)
  library(openxlsx)
  library(rlang)
  library(pROC)
  library(ggplot2)
  library(stringi)
})

set.seed(42)

# ------------------ User config ------------------
THRESHOLD <- 0.6
POS_CLASS <- "GC"
NEG_CLASS <- "NGC"

# Fixed 9-species panel (column names must match your Excel header)
pred_vars <- c(
  "Dialister_pneumosintes",
  "Treponema_vincentii",
  "Fusobacterium_nucleatum",
  "Actinomyces_ICM47",
  "Neisseria_oral",
  "Actinomyces_SGB17132",
  "Veillonella_rogosae",
  "Streptococcus_infantis",
  "Actinomyces_graevenitzii"
)

# ------------------ Helpers ------------------

normalize_id <- function(x) {
  x %>%
    stringr::str_replace_all("\\u00A0", " ") %>%
    stringr::str_squish() %>%
    stringi::stri_trans_general("Fullwidth-Halfwidth")
}

read_qpcr_table <- function(path, sheet = 1,
                            id_col = "SampleID",
                            group_col = "Group",
                            pred_vars = pred_vars,
                            pos_class = POS_CLASS,
                            neg_class = NEG_CLASS) {
  df <- readxl::read_excel(path, sheet = sheet)
  
  # Basic checks
  if (!group_col %in% colnames(df)) stop("Missing column: ", group_col, " in ", path)
  
  # Normalize SampleID if exists
  if (id_col %in% colnames(df)) {
    df <- df %>% mutate(!!id_col := normalize_id(.data[[id_col]]))
  }
  
  # Keep only required columns if present
  missing_vars <- setdiff(pred_vars, colnames(df))
  if (length(missing_vars) > 0) {
    stop("Missing predictors in ", path, ": ", paste(missing_vars, collapse = ", "))
  }
  
  # Enforce binary factor levels (pos first for yardstick event_level = 'first')
  df <- df %>%
    mutate(
      !!group_col := factor(.data[[group_col]], levels = c(pos_class, neg_class))
    )
  
  # Coerce predictors to numeric and replace NA with 0
  df <- df %>%
    mutate(across(all_of(pred_vars), ~ suppressWarnings(as.numeric(.)))) %>%
    mutate(across(all_of(pred_vars), ~ tidyr::replace_na(., 0)))
  
  df
}

metrics_at_threshold <- function(df_prob, truth_col, prob_col,
                                 threshold = THRESHOLD,
                                 pos_class = POS_CLASS,
                                 neg_class = NEG_CLASS) {
  df_cls <- df_prob %>%
    mutate(.pred_class = factor(ifelse(.data[[prob_col]] >= threshold, pos_class, neg_class),
                                levels = c(pos_class, neg_class)))
  
  tibble(
    threshold = threshold,
    sens = yardstick::sens(df_cls, truth = !!sym(truth_col), estimate = .pred_class, event_level = "first")$.estimate,
    spec = yardstick::spec(df_cls, truth = !!sym(truth_col), estimate = .pred_class, event_level = "first")$.estimate,
    acc  = yardstick::accuracy(df_cls, truth = !!sym(truth_col), estimate = .pred_class)$.estimate,
    ppv  = yardstick::ppv(df_cls, truth = !!sym(truth_col), estimate = .pred_class, event_level = "first")$.estimate,
    npv  = yardstick::npv(df_cls, truth = !!sym(truth_col), estimate = .pred_class, event_level = "first")$.estimate
  )
}

roc_auc_ci_delong <- function(truth, prob_pos, pos_class = POS_CLASS, neg_class = NEG_CLASS) {
  # pROC expects levels = c(neg, pos)
  y <- truth
  if (!is.factor(y)) y <- factor(y)
  y <- factor(y, levels = c(neg_class, pos_class))
  
  roc_obj <- pROC::roc(
    response  = y,
    predictor = prob_pos,
    levels    = c(neg_class, pos_class),
    direction = "<",
    quiet     = TRUE
  )
  auc_val <- as.numeric(pROC::auc(roc_obj))
  ci_val  <- as.numeric(pROC::ci.auc(roc_obj, conf.level = 0.95)) # low, mid, high
  
  list(roc_obj = roc_obj, auc = auc_val, ci = ci_val)
}

plot_roc_with_ci <- function(df_prob, truth_col, prob_col,
                             title, out_pdf,
                             pos_class = POS_CLASS,
                             neg_class = NEG_CLASS,
                             label_x = 0.65, label_y = 0.15) {
  roc_tbl <- yardstick::roc_curve(
    df_prob,
    truth = !!sym(truth_col),
    !!sym(prob_col),
    event_level = "first"
  )
  
  auc_ci <- roc_auc_ci_delong(
    truth   = df_prob[[truth_col]],
    prob_pos = df_prob[[prob_col]],
    pos_class = pos_class,
    neg_class = neg_class
  )
  
  auc_label <- sprintf("AUC = %.4f\n95%% CI: %.4f–%.4f",
                       auc_ci$auc, auc_ci$ci[1], auc_ci$ci[3])
  
  p <- ggplot(roc_tbl, aes(x = 1 - specificity, y = sensitivity)) +
    geom_line(linewidth = 1.1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
    coord_equal() +
    annotate("label", x = label_x, y = label_y, label = auc_label,
             hjust = 0, vjust = 0, size = 4, label.size = 0.3) +
    labs(
      title = title,
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)"
    ) +
    theme_minimal(base_size = 12)
  
  ggsave(out_pdf, plot = p, width = 7, height = 5.5, dpi = 600)
  invisible(list(plot = p, auc = auc_ci$auc, ci = auc_ci$ci))
}

cv_fit_and_predict <- function(wf, data, v = 5, repeats = 1, strata_col = "Group") {
  # Repeated CV if repeats > 1, otherwise plain CV
  if (repeats > 1) {
    folds <- rsample::vfold_cv(data, v = v, repeats = repeats, strata = !!sym(strata_col))
  } else {
    folds <- rsample::vfold_cv(data, v = v, strata = !!sym(strata_col))
  }
  
  res <- tune::fit_resamples(
    wf,
    resamples = folds,
    metrics = yardstick::metric_set(yardstick::roc_auc),
    control = tune::control_resamples(save_pred = TRUE)
  )
  
  pred_cv <- tune::collect_predictions(res)
  list(res = res, pred = pred_cv)
}

export_validation_predictions <- function(valid_df, prob_pos, out_xlsx,
                                          id_col = "SampleID",
                                          truth_col = "Group",
                                          pos_class = POS_CLASS,
                                          neg_class = NEG_CLASS,
                                          threshold = THRESHOLD) {
  out <- valid_df %>%
    mutate(
      Pred_Prob_POS = prob_pos,
      Pred_Label = ifelse(prob_pos >= threshold, pos_class, neg_class)
    ) %>%
    {
      if (id_col %in% colnames(.)) {
        dplyr::relocate(., all_of(id_col), all_of(truth_col), Pred_Prob_POS, Pred_Label)
      } else {
        dplyr::relocate(., all_of(truth_col), Pred_Prob_POS, Pred_Label)
      }
    }
  
  openxlsx::write.xlsx(out, out_xlsx, asTable = TRUE)
  invisible(out)
}
