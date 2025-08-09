# 17072025: Ali M, N.
# Step 0: Packages ------------
library(TCGAbiolinks)
library(dplyr)
library(pheatmap)
library(SummarizedExperiment)
library(sesameData)
library(sesame)
library(edgeR)
library(limma)
library(ggplot2)

# Step 1: Query Data --------------
prad.query.prot <- GDCquery(project = "TCGA-PRAD",
                            data.category = "Proteome Profiling",
                            data.type = "Protein Expression Quantification",
                            access = "open")
prad.out.prot <- getResults(prad.query.prot)
prad.out.prot %>% 
  select(sample_type) %>% 
  table()

GDCdownload(prad.query.prot, files.per.chunk = 20)


# Step 2: Process Data --------------
prad.prot.data <- GDCprepare(prad.query.prot, summarizedExperiment = TRUE)
prad.prot.data %>% 
  head() %>% 
  View()

prad.prot.data2 <- prad.prot.data[, -c(2:5)]
prad.prot.meta <- prad.out.prot
rownames(prad.prot.data2) <- prad.prot.data2[,1]
prad.prot.data2 <- prad.prot.data2[,-1]
prad.prot.data2 %>% 
  head() %>% View()

# Step 3: Imputation and filtering data ---------------
# Make sure it's a numeric matrix
prot_matrix <- as.matrix(prad.prot.data2)
mode(prot_matrix) <- "numeric"

# Remove rows with all NAs
non_empty_rows <- rowSums(is.na(prot_matrix)) < ncol(prot_matrix)
prot_matrix <- prot_matrix[non_empty_rows, ]
prot_matrix %>% head()

# Impute row means
prot_imputed <- t(apply(prot_matrix, 1, function(x) {
  if (all(is.na(x))) {
    return(rep(NA, length(x)))  # optionally skip or remove these
  } else {
    x[is.na(x)] <- mean(x, na.rm = TRUE)
    return(x)
  }
}))

# Check remaining NAs
sum(is.na(prot_imputed))  # Should be 0 or very few

# Step 4: PCA -------------------
prad.prot.scaled <- scale(prot_imputed)
prad.prot.scaled %>% head()
pca.results <- prcomp(t(prad.prot.scaled), center = TRUE, scale = TRUE)

# explained variances
explained.variance <- summary(pca.results)$importance[2,]
cum.sum.var <- summary(pca.results)$importance[3,]

num.components <- which(cum.sum.var >= 0.90)[1]
cat("Number of components to retain 90% variance: ", num.components, "\n")
### Number of component: 84
pca.data <- pca.results$x[, 1:num.components]
dim(pca.data)

# Step : Save the dataset
if(!dir.exists("data/TCGA_PRAD/prot"))
  dir.create("data/TCGA_PRAD/prot")
write.csv(pca.data, "data/TCGA_PRAD/prot/prad_prot_pca_data.csv")
write.csv(prad.out.prot, "data/TCGA_PRAD/prot/prad_prot_meta_data.csv")
