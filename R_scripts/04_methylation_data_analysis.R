# Step 0: Load Libraries ----------
library(TCGAbiolinks)
library(dplyr)
library(pheatmap)
library(SummarizedExperiment)
library(sesameData)
library(sesame)
library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
library(ChAMP)
library(minfi)
library(limma)


# Step 1: Methylation Data Query and Download --------------- 
prad.query.meth <- GDCquery(project = "TCGA-PRAD",
                            data.category = "DNA Methylation",
                            data.type = "Methylation Beta Value",
                            platform = "Illumina Human Methylation 450",
                            access = "Open")
prad.out.meth <- getResults(prad.query.meth)
prad.out.meth %>% 
  pull(sample_type) %>% 
  table()
### Sample Type:
### Metastatic 1
### Primary Tumor 502
### Solid Tissue Normal 50

GDCdownload(prad.query.meth, files.per.chunk = 10)

# Data Preparation
prad.meth.data <- GDCprepare(prad.query.meth, summarizedExperiment = TRUE)
prad.meth.data.mat <- assay(prad.meth.data)
prad.meth.row <- as.data.frame(SummarizedExperiment::rowData(prad.meth.data))
prad.meth.col <- as.data.frame(SummarizedExperiment::colData(prad.meth.data))

### Visualize some of the data
prad.meth.data.mat %>% head() %>% View()
# class(prad.meth.data.mat)
# View(prad.meth.data.mat)
prad.meth.row %>% head() %>% View()
prad.meth.col %>% head() %>% View()

# Step 2: Preprocess the datasets
### Filter data
beta <- prad.meth.data.mat
### Remvoe probes with NA
beta <- beta[complete.cases(beta),]

### Filter: remvoe sex chromosome probes
annotation <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
### Sex probes
sex_probes <- annotation$char %in% c("chrX", "chrY")
### 0 sex probes
group <- as.factor(prad.meth.col$sample_type)
design <- model.matrix(~0 + group)
colnames(design) <- make.names(levels(group)) 

colnames(design)

# Step 3: Differential Methlylation ----------------
fit <- lmFit(beta, design)

contrast.matrix <- makeContrasts(
  Primary.Tumor - Solid.Tissue.Normal,
  levels = design
)

fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)
top_dmcs <- topTable(fit2, number = Inf, adjust = "fdr")
# Step 4: Subset beta matrix -----------------
top_beta <- beta[rownames(top_dmcs),]
dim(top_beta)
dim(beta)
# Step 5: Apply PCA ---------------
top.beta.scaled <- scale(top_beta)
pca.results <- prcomp(t(top.beta.scaled), center = TRUE, scale = TRUE)
# Step 6: Explained Variance -------------
explained.variance <- summary(pca.results)$importance[2,]
cum.sum.var <- summary(pca.results)$importance[3,]

num.components <- which(cum.sum.var >= 0.90)[1]
cat("Number of components to retain 90% variance: ", num.components, "\n")

pca.data <- pca.results$x[, 1:num.components]
dim(pca.data)
# Step 7: Save the results ----------------
if(!dir.exists("data/TCGA_PRAD/methylation"))
  dir.create("data/TCGA_PRAD/methylation")
write.csv(pca.data, "data/TCGA_PRAD/methylation/prad_meth_pca_result.csv")
write.csv(prad.meth.row, "data/TCGA_PRAD/methylation/prad_row_data.csv")
### concat list in 
for(col in names(prad.meth.col)){
  if(is.list(prad.meth.col[[col]])){
    prad.meth.col[[col]] <- sapply(prad.meth.col[[col]], function(x) paste(x, collapse = ";"))
  }
}
write.csv(prad.meth.col, "data/TCGA_PRAD/methylation/prad_col_data.csv")

