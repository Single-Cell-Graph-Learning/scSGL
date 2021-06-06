#' Performs a genewise Kruskal Wallis test on a SingleCellExperiment object
#' 
#' @author Jack Dodson
#' @param sce SingleCellExperiment object with a logcounts assay 
#' and Dose column in the cell metadata
#' 
#' @return a vector of p values from the Kruskal Wallis test
#' @export
batchKW = function(sce){
  data = as.matrix(logcounts(sce))
  dose = colData(sce)[,"Dose"]
  kw.pvalues = apply(data, 1, function(x) runKW(x, dose))
  kw.out = data.frame(kw.pvalues)
  return(kw.out)
}

#' Performs a Kruskal Wallis test on a logcounts vactor for a given dose
#' 
#' @author Jack Dodson
#' @param data The logcounts vector
#' @param dose The dose to analyze
#' 
#' @return A p value from the Kruskal Wallis test
#' @export
runKW = function(data, dose){
  my_data <- data.frame(value = data, dose = dose)
  res.kw = kruskal.test(value ~ dose, data = my_data)
  kw.pvalue = res.kw[[3]]
  return(kw.pvalue)
}