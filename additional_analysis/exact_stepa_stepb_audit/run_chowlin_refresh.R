suppressPackageStartupMessages({
  library(tempdisagg)
})

args <- commandArgs(trailingOnly = TRUE)
base_dir <- if (length(args) >= 1) args[[1]] else "."
out_dir <- file.path(base_dir, "out")
inputs <- read.csv(file.path(out_dir, "chowlin_inputs.csv"), check.names = FALSE)
sets <- read.csv(file.path(out_dir, "chowlin_indicator_sets.csv"), stringsAsFactors = FALSE)

years <- sort(unique(inputs$Year))
annual <- aggregate(annual_gerd ~ Year, data = inputs, FUN = function(x) x[[1]])
annual <- annual[match(years, annual$Year), ]
y_ts <- ts(annual$annual_gerd, start = min(years), frequency = 1)

fit_set <- function(set_name) {
  topics <- sets$topic[sets$set == set_name]
  environment <- new.env(parent = globalenv())
  environment$y_ts <- y_ts
  terms <- character(length(topics))
  for (index in seq_along(topics)) {
    variable <- paste0("indicator_", index)
    environment[[variable]] <- ts(
      inputs[[topics[[index]]]],
      start = c(min(inputs$Year), min(inputs$Month[inputs$Year == min(inputs$Year)])),
      frequency = 12
    )
    terms[[index]] <- variable
  }
  formula <- as.formula(paste("y_ts ~", paste(terms, collapse = " + ")), env = environment)
  model <- td(formula, to = "monthly", method = "chow-lin-maxlog", conversion = "sum")
  prediction <- as.numeric(predict(model))
  data.frame(
    Year = inputs$Year,
    Month = inputs$Month,
    estimate = prediction,
    set = set_name,
    rho = as.numeric(model$rho),
    stringsAsFactors = FALSE
  )
}

results <- rbind(fit_set("old"), fit_set("current"))
write.csv(results, file.path(out_dir, "chowlin_old_vs_current.csv"), row.names = FALSE)

old_saved <- read.csv(file.path(
  dirname(dirname(base_dir)),
  "temporal_disaggregation",
  "classical_regression_tempdisagg",
  "Disaggregated_Monthly_RD_Expenditure_Sax.csv"
))
names(old_saved)[names(old_saved) == "Monthly_RD_Expenditure"] <- "saved_old"
old_rerun <- subset(results, set == "old")
comparison <- merge(old_rerun, old_saved, by = c("Year", "Month"))
comparison$difference <- comparison$estimate - comparison$saved_old
write.csv(comparison, file.path(out_dir, "chowlin_old_replication_check.csv"), row.names = FALSE)

cat("Current indicators:", paste(sets$topic[sets$set == "current"], collapse = ", "), "\n")
cat("Old replication max absolute difference:", max(abs(comparison$difference)), "\n")
