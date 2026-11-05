# Panel VAR by GMM with the R package panelvar (Sigmund & Ferstl 2021): first-order panel VAR in the tradition of
# Holtz-Eakin, Newey & Rosen (1988), estimated by two-step system GMM (Blundell & Bond 1998) on forward orthogonal
# deviations, collapsed instruments (lags 2-4 of the dependent variables).
# Usage: Rscript pvar.R <panel.csv> <out.csv> <dependent vars, comma-separated> <exogenous vars, comma-separated, or -> <cutoff> ...
# The panel has columns unit, time and the variables. One model per cutoff, estimated on the rows with time <= cutoff;
# exogenous variables without variation in that sample are dropped. Writes the coefficients in long form.
suppressMessages({library(panelvar); library(parallel)})
a <- commandArgs(TRUE)
d <- read.csv(a[1]); dep <- strsplit(a[3], ',')[[1]]
ex <- if (a[4] == '-') character(0) else strsplit(a[4], ',')[[1]]
cuts <- as.integer(a[-(1:4)])
fit <- function(k) {
  s <- d[d$time <= k, ]
  x <- ex[vapply(ex, function(v) var(s[[v]]) > 1e-12, logical(1))]
  args <- list(dependent_vars = dep, lags = 1, transformation = 'fod', data = s, panel_identifier = c('unit', 'time'),
               steps = 'twostep', system_instruments = TRUE, system_constant = TRUE, max_instr_dependent_vars = 4,
               min_instr_dependent_vars = 2L, collapse = TRUE)
  if (length(x)) args$exog_vars <- x                     # panelvar does not accept exog_vars = NULL
  m <- tryCatch(suppressWarnings(do.call(pvargmm, args)), error = function(e) conditionMessage(e))
  if (is.character(m)) { message('cutoff ', k, ': no estimate (', m, ')'); return(NULL) }
  b <- coef(m)
  data.frame(cutoff = k, equation = rep(rownames(b), ncol(b)), term = rep(colnames(b), each = nrow(b)), value = as.vector(b),
             hansen_p = tryCatch(hansen_j_test(m)$p.value, error = function(e) NA))
}
res <- do.call(rbind, mclapply(cuts, fit, mc.cores = max(1, min(4, detectCores()))))
write.csv(res, a[2], row.names = FALSE)
