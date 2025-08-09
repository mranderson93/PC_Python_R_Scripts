# 17072025: Ali M, N.
# Step 0: libraries
library(TCGAbiolinks)
library(dplyr)
library(pheatmap)
library(SummarizedExperiment)
library(sesameData)
library(sesame)
# library(edgeR)
# library(limma)
library(ggplot2)
library(DESeq2)

# Step 1: Query Data
prad.query.mirna <- GDCquery(project = "TCGA-PRAD",
                             data.category = "Transcriptome Profiling",
                             data.type = "miRNA Expression Quantification",
                             access = "Open")
prad.out.mirna <- getResults(prad.query.mirna)
# sink("prad.out.mirna.txt")
prad.out.mirna %>% 
  View()
# sink()

prad.out.mirna %>% 
  select(sample_type) %>% 
  table()
### Sample Typle
# Metastatic: 1
# Primary Tumor: 498
# Solid Tissue Normal: 52

GDCdownload(prad.query.mirna, files.per.chunk = 20)
### Data Preparation
prad.mirna.data <- GDCprepare(prad.query.mirna, summarizedExperiment = TRUE)

prad.mirna.readcount <- prad.mirna.data[, grep("read_count", colnames(prad.mirna.data))]
colnames(prad.mirna.readcount) <- gsub("read_count_", "", colnames(prad.mirna.readcount))
# prad.out.mirna$cases
prad.mirna.data.filtered <- prad.mirna.readcount[, prad.out.mirna$cases]
prad.mirna.data.filtered %>% dim()


### Row Data
prad.mirna.row <- prad.out.mirna
prad.mirna.col <- data.frame(
  row.names <- colnames(prad.mirna.data.filtered),
  condition <- factor(prad.mirna.row$sample_type)
)
colnames(prad.mirna.col) <- c("Cases", "Sample_type")
rownames(prad.mirna.col) <- prad.mirna.col$Cases

all(colnames(prad.mirna.data.filtered) %in% rownames(prad.mirna.col))
all(colnames(prad.mirna.data.filtered) == rownames(prad.mirna.col))

dds <- DESeqDataSetFromMatrix(countData = prad.mirna.data.filtered,
                              colData = prad.mirna.col,
                              design = ~ Sample_type)

dds
# Filter Low Counts
dds <- dds[rowSums(counts(dds)) > 10,]

# Run Differential Expression Analysis
dds <- DESeq(dds)
res <- results(dds)
View(res)

# sink("mirnaDifferential_exprn.txt")
# res@listData
# sink()
res.df <- as.data.frame(res@listData)
res.df %>% head() %>% View()


## Filter res.df
alpha <- 0.05
lfc <- 1
res.df.sig <- res.df %>% 
  filter(padj < alpha & abs(log2FoldChange) > lfc)
dim(res.df)
dim(res.df.sig)
deg.indices <- which(res.df$padj < alpha & abs(res.df$log2FoldChange) > lfc)

#### START FROM Here: 16.07.2025 ##############
prad.mirna.deg.data <- prad.mirna.data.filtered[deg.indices,]
dim(prad.mirna.data.filtered)
dim(prad.mirna.deg.data)
# Filter Significant differentially expressed ones
prad.mirna.deg.data %>% head() %>% View()

## Apply PCA --------------------
prad.mirna.scaled <- scale(prad.mirna.deg.data)
pca.results <- prcomp(t(prad.mirna.scaled), center = TRUE, scale = TRUE)

explained.var <- summary(pca.results)$importance[2,]
cum.sum.var <- summary(pca.results)$importance[3,]
cum.sum.sorted <- sort(cum.sum.var, decreasing = TRUE)
plot(cum.sum.var, type = "l")

### Selecting number of components
num.components <- which(cum.sum.var >= 0.9)[1]
pca.data <- pca.results$x[, 1:80]

### Save the filtered pca data
write.csv(pca.data, "./data/TCGA_PRAD/mirna/mirna_pca_filtered_data.csv")
write.csv(res.df.sig, "./data/TCGA_PRAD/mirna/differential_expresion_result.csv")
# write.csv(prad.mirna.col, "")
######
deg.indices <- which(res$padj < alpha & abs(res$log2FoldChange) > lfc_threshold)
deg.genes <- rownames(res)
# Step 2: Differential Expression -------------

if(!dir.exists("data/TCGA_PRAD/mirna"))
  dir.create("data/TCGA_PRAD/mirna")
write.csv(prad.mirna.data.filtered, "data/TCGA_PRAD/mirna/mirna_data_matrix.csv")
write.csv(prad.out.mirna, "data/TCGA_PRAD/mirna/mirna_meta_data.csv")