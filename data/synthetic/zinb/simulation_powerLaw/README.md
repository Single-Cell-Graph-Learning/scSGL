All the simulation RDS files are lists having a) true covariance matrix, b)true partial correlation matrix (theta), c)true adjacency matrix,
d)true multivariate normal data e)zero inflated negative binomial count data. The multivariate normal and zero-inflated negative binomial datasets have 
a cell(rows) vs genes(column) structure. In all the RDS files; nc is the number of clusters, number of genes per cluster is 50, rho is the exponential decay parameter controlling the dropout 
rates in the datasets and true graph density is 0.01. The zeroinflated negative binomial data needs to be rowsum scaled and log1p transformed before 
applying the scSGL technique.
