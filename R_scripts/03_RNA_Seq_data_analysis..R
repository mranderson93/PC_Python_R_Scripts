#
# Md. Neaz Ali
# M.Sc in Statistics
# Islamic University, Kushtia - 7003,
# Bangladesh
# Date: 14072025
#

# Step 0: Load Libraries ----------
library(TCGAbiolinks)
library(DESeq2)
library(edgeR)
library(tximport)
library(edgeR)
library(org.Hs.eg.db)
library(ggplot2)
library(pheatmap)
library(dplyr)
library(SummarizedExperiment)
library(sesame)
library(sesameData)

# Step 1: Query All the RNA-Seq data ----------
## CMI-MPC project --------------- 
### Total Cases: 38 Primary Tumor
cmi.proj.rna <- GDCquery(project = "CMI-MPC",
                         data.category = "Transcriptome Profiling",
                         data.type = "Gene Expression Quantification",
                         experimental.strategy = "RNA-Seq",
                         workflow.type = "STAR - Counts",
                         access = "Open")

out.cmi.proj.rna <- getResults(cmi.proj.rna)
View(out.cmi.proj.rna)

out.cmi.proj.rna %>% 
  select(sample_type) %>% 
  table()
### Download The Dataset
GDCdownload(cmi.proj.rna, files.per.chunk = 20)
cmi.rna <- GDCprepare(cmi.proj.rna, summarizedExperiment = TRUE)
cmi.rna.data <- assay(cmi.rna)
dim(cmi.rna.data) # Need to trnaspose the dataset
# cmi.rna.data <- cmi.rna.data %>% t()
cmi.rna.row.df <- as.data.frame(SummarizedExperiment::rowData(cmi.rna))
cmi.rna.col.df <- as.data.frame(SummarizedExperiment::colData(cmi.rna))
cmi.rna.col.filt <- cmi.rna.col.df %>% 
  select("sample", "sample_type", "primary_diagnosis", "vital_status", "disease_type", "primary_site" )


## TCGA-PRAD Project --------------
### Total Cases: 554
### Primary Tumor: 501
### Solid Tissue Normal: 52
### Metastatic: 1
prad.proj.rna <- GDCquery(project = "TCGA-PRAD",
                          data.category = "Transcriptome Profiling",
                          data.type = "Gene Expression Quantification",
                          experimental.strategy = "RNA-Seq",
                          workflow.type = "STAR - Counts",
                          access = "Open" )
out.prad.rna <- getResults(prad.proj.rna)
View(out.prad.rna)

barcodes <- out.prad.rna$cases
# barcodes[1] %>% class()
# substr(barcodes[1], 1, 12)
# #### Taking out the substr barcodes to match with the tcia images
# sink("TCGA-PRAD-barcodes.txt")
# barcodes.short <- substr(barcodes, 1, 12)
# print(barcodes.short)
# sink()

### Combine all the short barcodes in out file
out.prad.rna2 <- cbind(out.prad.rna, barcodes.short)
View(out.prad.rna2)
### Filter only main barcode, short barcode and sample type
# indices <- which(colnames(out.prad.rna2) %in% c("cases", "sample_type", "barcodes.short"))
# out.prad.rna.filt <- out.prad.rna2[,indices]
# if(!dir.exists("data/images"))
#   dir.create("data/images")
# write.csv(out.prad.rna.filt, "data/images/prad_barcodes.csv")

out.prad.rna %>% select(sample_type) %>% table()
### Data Download
GDCdownload(prad.proj.rna, files.per.chunk = 20)
prad.rna.data <- GDCprepare(prad.proj.rna, summarizedExperiment = TRUE)

prad.rna.data.matrix <- assay(prad.rna.data)
prad.rna.data.matrix %>% dim()
# prad.rna.data.matrix <- prad.rna.data.matrix %>% t()
prad.rna.row.df <- as.data.frame(SummarizedExperiment::rowData(prad.rna.data))
prad.rna.col.data2 <- as.data.frame(SummarizedExperiment::colData(prad.rna.data))
prad.rna.col.filt <- prad.rna.col.data2 %>% 
  select("sample", "sample_type", "primary_diagnosis", "vital_status", "disease_type", "primary_site" )

dim(prad.rna.row.df)
## WCDT-MCRPC Project -----------------
### Total Cases: 99
### Metastatic: 99
wcdt.proj.rna <- GDCquery(project = "WCDT-MCRPC",
                          data.category = "Transcriptome Profiling",
                          data.type = "Gene Expression Quantification",
                          experimental.strategy = "RNA-Seq",
                          workflow.type = "STAR - Counts")

out.wcdt.rna <- getResults(wcdt.proj.rna)
View(out.wcdt.rna)

out.wcdt.rna %>% 
  select(sample_type) %>% 
  table()
## Download data
GDCdownload(wcdt.proj.rna, files.per.chunk = 20)
wcdt.rna <- GDCprepare(wcdt.proj.rna, summarizedExperiment = TRUE)
# data matrix
wcdt.rna.data <- assay(wcdt.rna)
wcdt.rna.data %>% dim()
wcdt.rna.col <- as.data.frame(SummarizedExperiment::colData(wcdt.rna))
wcdt.rna.row.df <- as.data.frame(SummarizedExperiment::rowData(wcdt.rna))
wcdt.rna.col.filt <-  wcdt.rna.col %>%
  select("sample", "sample_type", "primary_diagnosis", "vital_status", "disease_type", "primary_site" )

# Step 2: Process the data -----------
#Done
# Step 3: Integrate all the RNA-Seq data -----------
dim(cmi.rna.data)
dim(prad.rna.data.matrix)
dim(wcdt.rna.data)
rna.seq.data <- cbind(cmi.rna.data,
                      prad.rna.data.matrix,
                      wcdt.rna.data)
rna.seq.data %>% dim()
### Integrate Row Data
cmi.rna.row.df %>% dim()
prad.rna.row.df %>% dim()
wcdt.rna.row.df %>% dim()

rna.row.data <- rbind(cmi.rna.row.df, 
                      prad.rna.row.df, 
                      wcdt.rna.row.df)

rna.row.data %>% dim()
rna.row.data %>% head() %>% View()
### Integrate Col data
cmi.rna.col %>% as.data.frame(cmi.rna.col)

cmi.rna.col %>% View()
prad.rna.col.data %>% View()
wcdt.rna.col %>% View()

cmi.rna.col@listData %>% View()

cmi.rna.col.filt %>% dim()
prad.rna.col.filt %>% dim()
wcdt.rna.col.filt %>% dim()
# Integrate
rna.col.data <- rbind(cmi.rna.col.filt,
                      prad.rna.col.filt,
                      wcdt.rna.col.filt)
rna.col.data %>% head() %>% View()
### Major Important Variables To Capture From Col Data
### 1. Barcodes
### 2. Sample Type
rna.col.data %>% select("sample_type") %>% table()
# names(cmi.rna.col)
# any(c("sample", "sample_type", "primary_diagnosis",  "vital_status", "disease_type", "primary_site" ) %in% colnames(cmi.rna.col))
# which(c("sample", "sample_type", "primary_diagnosis", "vital_status", "disease_type", "primary_site" ) %in% colnames(cmi.rna.col))
# 
# cmi.rna.col %>% View()
# cmi.rna.col.filt <- cmi.rna.row.df %>% 
#   select("sample", "sample_type", "primary_diagnosis", "vital_status", "disease_type", "primary_site" )
# 
# cmi.rna.row.df %>% View()
# cmi.rna.col.filt %>% View()
# cmi.rna.col.filt$disease_type[1]
# cmi.rna.col$primary_site



# Step 4: Select only protein coding genes -------------
# Protein Coding Genes are in the Row Data
rna.row.df <- rna.row.data@listData %>% 
  as.data.frame()

rna.row.data %>% 
  head() %>% 
  View()
rna.row.data$gene_type %>% 
  table()
### Select only protein coding genes: 19962
# protein.coding <- rna.row.data %>% 
#   select(gene_type, gene_id) %>% 
#   filter(gene_type == "protein_coding") %>% 
#   select(gene_id) %>% 
#   as.vector()

protein.coding <- rna.row.data %>% 
  filter(gene_type == "protein_coding") %>% 
  pull(gene_id)  # pulls gene_id as a vector

length(protein.coding)
duplicated(protein.coding) %>% sum()
### Remove the duplicated values only keep the unique gene id
protein.coding.unique <- unique(protein.coding)
length(protein.coding.unique) # Lenght Now: 19962



### Filter Main data selecting protein coding genes
### rna seq data dim: 60660 x 691
rna.seq.data %>% head() %>% View()
rna.seq.data.filtered <- rna.seq.data[rownames(rna.seq.data) %in% protein.coding.unique,]
rna.seq.data.filtered %>% dim() 
### After Filtering dimension: 19962 x 619
### RNA expr filtered ---------------
rna.seq.data.filtered %>% head() %>% View()
# Step 5: Quality Control -------------
### Transpose the expression matrix: The row should represen genes and column: samples/cases
# rna.seq.data.filtered.T <- t(rna.seq.data.filtered)
keep <- rowSums(rna.seq.data.filtered > 10) > 5
length(keep) # 691
rna.seq.filt.keep <- rna.seq.data.filtered[keep,]
rna.seq.filt.keep %>% dim()

# Library Size
sample_sums <- colSums(rna.seq.filt.keep)
hist(sample_sums, main = "Library Sizes", xlab = "Total counts per sample")
# Step 6: Differential Expression Analysis -------------
rna.seq.filt.keep %>% head() %>% View()
rna.col.data %>% dim()

all(colnames(rna.seq.filt.keep) %in% rownames(rna.col.data))
all(colnames(rna.seq.filt.keep) == rownames(rna.col.data))
### all samples are in the dataset
rna.col.data$sample_type <- as.factor(rna.col.data$sample_type)
levels(rna.col.data$sample_type)


dds <- DESeqDataSetFromMatrix(countData = rna.seq.filt.keep,
                              colData = rna.col.data,
                              design = ~sample_type)
dds
dds <- DESeq(dds)
res <- results(dds)

### Contrast between: primary tumor vs solid tissue normal
res_tumor_vs_normal <- results(dds, contrast = c("sample_type", "Primary Tumor", "Solid Tissue Normal"))
### contrast between: metastatic vs solid tissue normal
res_met_vs_normal <- results(dds, contrast = c("sample_type", "Metastatic", "Solid Tissue Normal"))
### Metastatic vs Primary Tumor
res_met_vs_tumor <- results(dds, contrast = c("sample_type", "Metastatic", "Primary Tumor"))

# Step 7: Filter significant genes --------------
sig.genes <- res %>%
  as.data.frame() %>%
  filter(padj < 0.05) %>%
  rownames()

rna.seq.data.filt.sig <- rna.seq.data.filtered[sig.genes, ]
# Step 8: Check for upregulated and downregulated genes -------------
rna.seq.data.filt.sig %>% head() %>% View()
rna.seq.scaled <- scale(rna.seq.data.filt.sig)
pca.results <- prcomp(t(rna.seq.scaled), center = TRUE, scale = TRUE)
### Explained Variance
explained.var <- summary(pca.results)$importance[2,]
cum.sum.var <- summary(pca.results)$importance[3,]
cumsumvar.sorted <- sort(cum.sum.var, decreasing = TRUE)
plot(cumsumvar.sorted, type = "l")
### Number of components
num.components <- which(cum.sum.var >= 0.9)[1]
# 80 components
pca.data <- pca.results$x[, 1:num.components]
dim(pca.data)
pca.data %>% dim()
plot(cum.sum.var)
# Step 9: Save the filtered merged data set and column datasets -----------
rna.row.data %>% dim()
rna.col.data %>% dim()
write.csv(pca.data, "./data/TCGA_PRAD/rna/integrated_rna_seq_pca_result.csv")
write.csv(rna.row.data, "./data/TCGA_PRAD/rna/integrated_rna_row_data.csv")
### Convert list into strings of column data
for(col in names(rna.col.data)){
  if(is.list(rna.col.data[[col]])){
    rna.col.data[[col]] <- sapply(rna.col.data[[col]], function(x) paste(x, collapse = ";"))
  }
}
write.csv(rna.col.data, "./data/TCGA_PRAD/rna/integrated_rna_col_data.csv")
