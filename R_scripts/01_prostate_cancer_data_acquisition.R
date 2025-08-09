# Step: 0 Load Required Libraries
# if(!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# 
# BiocManager::install("TCGAbiolinks")

library(TCGAbiolinks)
library(dplyr)
library(pheatmap)
library(SummarizedExperiment)
library(sesameData)
library(sesame)

?library()
## Step 1: Check For Projects
projects <- getGDCprojects()
projects %>% 
  dim()
View(projects)


# Step 1.2: Projects to include: 1. CMI-MPC, 2. TCGA-PRAD, 3. WCDT-MCRPC ------------------

# Step 2: Download CMI-MPC Related Datasets: Available: 1. Simple Nucleotide Variation, 
# 2. Sequencing Reads, 3. Transcriptome Profiling , 4. Structural Variation ------------------
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
# Cancer Type: 38 cases of Primary Tumor
# Select Bar Code
cmi.proj.rna.cases <- out.cmi.proj.rna %>% 
  pull(cases)

# Download RNA-Seq data
cmi.proj.rna2 <- GDCquery(project = "CMI-MPC",
                          data.category = "Transcriptome Profiling",
                          data.type = "Gene Expression Quantification",
                          experimental.strategy = "RNA-Seq",
                          workflow.type = "STAR - Counts",
                          access = "Open",
                          barcode = cmi.proj.rna.cases)
GDCdownload(cmi.proj.rna2, files.per.chunk = 20)
# Data preprocessing
cmi.rna <- GDCprepare(cmi.proj.rna2, summarizedExperiment = TRUE)
# Data Matrix
cmi.rna.data <- assay(cmi.rna)
# View(cmi.rna.data)
cmi.rna.row <- rowData(cmi.rna)
cmi.rna.col <- colData(cmi.rna)

# Save data
if(!dir.exists("data/CMI_MPC/rna"))
  dir.create("data/CMI_MPC/rna")

write.csv(cmi.rna.data, "data/CMI_MPC/rna/cmi_rna_seq_data_mat.csv", row.names = F)
write.csv(cmi.rna.row, "data/CMI_MPC/rna/cmi_rna_seq_data_row.csv", row.names = F)
for(col in names(cmi.rna.col)){
  if(is.list(cmi.rna.col[[col]])){
    cmi.rna.col[[col]] <- sapply(cmi.rna.col[[col]], function(x) paste(x, collapse = ";"))
  }
}
write.csv(cmi.rna.col, "data/CMI_MPC/rna/cmi_rna_seq_data_col.csv")



## Step 1.2: Single Nucleotide Variation data ---------------
cmi.snv.query <- GDCquery
  project = "CMI-MPC",
  data.category = "Simple Nucleotide Variation",
  data.type = "Masked Somatic Mutation",
  access = "controlled"  # Most SNV data is controlled
)

out.cmi.snv.query <- getResults(cmi.snv.query)
View(out.cmi.snv.query)
GDCdownload(out.cmi.proj.rna, files.per.chunk = 20)


## Step 1.3: miRNA data *** No data formiRNA data
# Download RNA-Seq data
cmi.proj.mirna <- GDCquery(
  project = "CMI-MPC",
  data.category = "Transcriptome Profiling",
  data.type = "miRNA Expression Quantification",
  access = "controlled"
)

out.cmi.proj.mirna <- getResults(cmi.proj.mirna)
View(out.cmi.proj.mirna)

cmi.proj.mirna2 <- GDCquery(project = "CMI-MPC",
                          data.category = "Transcriptome Profiling",
                          data.type = "Gene Expression Quantification",
                          experimental.strategy = "miRNA-Seq",
                          workflow.type = "STAR - Counts",
                          access = "Open",
                          barcode = cmi.proj.rna.cases)
GDCdownload(cmi.proj.mirna2, files.per.chunk = 20)
# Data preprocessing
cmi.mirna <- GDCprepare(cmi.proj.mirna2, summarizedExperiment = TRUE)
# Data Matrix
cmi.mirna.data <- assay(cmi.mirna)
# View(cmi.rna.data)
cmi.mirna.row <- rowData(cmi.mirna)
cmi.mirna.col <- colData(cmi.mirna)

# Save data
if(!dir.exists("data/CMI_MPC/mirna"))
  dir.create("data/CMI_MPC/mirna")

write.csv(cmi.rna.data, "data/CMI_MPC/mirna/cmi_mirna_seq_data_mat.csv", row.names = F)
write.csv(cmi.rna.row, "data/CMI_MPC/mirna/cmi_mirna_seq_data_row.csv", row.names = F)
for(col in names(cmi.mirna.col)){
  if(is.list(cmi.mirna.col[[col]])){
    cmi.mirna.col[[col]] <- sapply(cmi.mirna.col[[col]], function(x) paste(x, collapse = ";"))
  }
}
write.csv(cmi.mirna.col, "data/CMI_MPC/mirna/cmi_mirna_seq_data_col.csv")




# Step 3: Project TCGA_PRAD ----------------
## Step 3.1: RNA Seq
prad.proj.rna <- GDCquery(project = "TCGA-PRAD",
                          data.category = "Transcriptome Profiling",
                          data.type = "Gene Expression Quantification",
                          experimental.strategy = "RNA-Seq",
                          workflow.type = "STAR - Counts",
                          access = "Open" )
out.prad.rna <- getResults(prad.proj.rna)
View(out.prad.rna)


out.prad.rna %>% select(sample_type) %>% table()
GDCdownload(prad.proj.rna, files.per.chunk = 20)

### Data Processing
prad.rna.data <- GDCprepare(prad.proj.rna, summarizedExperiment = TRUE)
prad.rna.data.matrix <- assay(prad.rna.data) %>% t()

prad.rna.data.matrix %>% head() %>% View()

prad.rna.row.data <- rowData(prad.rna.data)
prad.rna.col.data <- colData(prad.rna.data)
# Save rna.data and row.data
if(!dir.exists("data/TCGA_PRAD/rna/"))
  dir.create("data/TCGA_PRAD/rna/")
write.csv(prad.rna.data.matrix, "data/TCGA_PRAD/rna/rna_data_matrix.csv")
write.csv(prad.rna.row.data, "data/TCGA_PRAD/rna//rna_data_row_data.csv")

for(col_name in names(prad.rna.col.data)){
  if(is.list(prad.rna.col.data[[col_name]])){
    prad.rna.col.data[[col_name]] <- sapply(prad.rna.col.data[[col_name]], function(x) paste(x, collapse = ";"))
  }
}
write.csv(prad.rna.col.data, "data/TCGA_PRAD/rna/rna_data_col_data.csv")

## Step 3.2: miRNA Seq --------------
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
GDCdownload(prad.query.mirna, files.per.chunk = 20)
### Data Preparation
prad.mirna.data <- GDCprepare(prad.query.mirna, summarizedExperiment = TRUE)

prad.mirna.readcount <- prad.mirna.data[, grep("read_count", colnames(prad.mirna.data))]
colnames(prad.mirna.readcount) <- gsub("read_count_", "", colnames(prad.mirna.readcount))
# prad.out.mirna$cases
prad.mirna.data.filtered <- prad.mirna.readcount[, prad.out.mirna$cases]
prad.mirna.data.filtered %>% dim()


if(!dir.exists("data/TCGA_PRAD/mirna"))
  dir.create("data/TCGA_PRAD/mirna")
write.csv(prad.mirna.data.filtered, "data/TCGA_PRAD/mirna/mirna_data_matrix.csv")
write.csv(prad.out.mirna, "data/TCGA_PRAD/mirna/mirna_meta_data.csv")


#################################
# START - FROM PROTEOMICS DATA
# 25062025: Ali M, N.
prad.query.prot <- GDCquery(project = "TCGA-PRAD",
                            data.category = "Proteome Profiling",
                            data.type = "Protein Expression Quantification",
                            access = "open")
prad.out.prot <- getResults(prad.query.prot)
prad.out.prot %>% 
  select(sample_type) %>% 
  table()

GDCdownload(prad.query.prot, files.per.chunk = 20)


### Data preparing 
prad.prot.data <- GDCprepare(prad.query.prot, summarizedExperiment = TRUE)
prad.prot.data %>% 
  head() %>% 
  View()

prad.prot.data2 <- prad.prot.data[, -c(2:5)]
prad.prot.meta <- prad.out.prot

if(!dir.exists("data/TCGA_PRAD/prot"))
  dir.create("data/TCGA_PRAD/prot")
write.csv(prad.prot.data2, "data/TCGA_PRAD/prot/prad_prot_data_matrix.csv")
write.csv(prad.prot.meta, "data/TCGA_PRAD/prot/prad_prot_meta_data.csv")


# Methylation Data 
prad.query.meth <- GDCquery(project = "TCGA-PRAD",
                            data.category = "DNA Methylation",
                            data.type = "Methylation Beta Value",
                            platform = "Illumina Human Methylation 450",
                            access = "Open")
prad.out.meth <- getResults(prad.query.meth)
prad.out.meth %>% 
  select(sample_type) %>% 
  table()
### Sample Type:
### Metastatic 1
### Primary Tumor 502
### Solid Tissue Normal 50

GDCdownload(prad.query.meth, files.per.chunk = 10)

### Data Preparation
prad.meth.data <- GDCprepare(prad.query.meth, summarizedExperiment = TRUE)
prad.meth.data.mat <- assay(prad.meth.data)
prad.meth.row <- as.data.frame(SummarizedExperiment::rowData(prad.meth.data))
prad.meth.col <- as.data.frame(SummarizedExperiment::colData(prad.meth.data))


# cmi.rna.row.df <- as.data.frame(SummarizedExperiment::rowData(cmi.rna))
# cmi.rna.col.df <- as.data.frame(SummarizedExperiment::colData(cmi.rna))
# cmi.rna.col.filt <- cmi.rna.col.df %>% 
#   select("sample", "sample_type", "primary_diagnosis", "vital_status", "disease_type", "primary_site" )

# Project: WCDT-MCRPC -----
# proj summary: 
wcdt.summary <- getProjectSummary("WCDT-MCRPC")
View(wcdt.summary)
wcdt.summary$data_categories


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

# Cancer Type: 99 Metastatic Cancer
# Download wcdt-rna data

GDCdownload(wcdt.proj.rna, files.per.chunk = 20)

# preprocessing
wcdt.rna <- GDCprepare(wcdt.proj.rna, summarizedExperiment = TRUE)
# data matrix
wcdt.rna.data <- assay(wcdt.rna)
wcdt.rna.col <- colData(wcdt.rna)
wcdt.rna.row <- rowData(wcdt.rna)

if(!dir.exists("data/WCDT")){
  dir.create("data/WCDT")
}

if(!dir.exists("data/WCDT/rna")){
  dir.create("data/WCDT/rna")
}

write.csv(wcdt.rna.data, "data/WCDT/rna/wcdt_rna_data.csv")
write.csv(wcdt.rna.row, "data/WCDT/rna/wcdt_rna_row_data.csv")

for(col in names(wcdt.rna.col)){
  if(is.list(wcdt.rna.col[[col]])){
    wcdt.rna.col[[col]] <- sapply(wcdt.rna.col[[col]], function(x) paste(x, collapse = ";"))
  }
}
write.csv(wcdt.rna.col, "data/WCTD/wcdt_rna_col_data.csv")




### WCDT-CNV: Data NA
# wcdt.snv.query <- GDCquery(
#   project = "WCDT-MCRPC",
#   data.category = "Simple Nucleotide Variation",
#   data.type = "Masked Somatic Mutation",
#   access = "controlled"  # Most SNV data is controlled
# )
# out.wcdt.snv <- getResults(wcdt.snv.query)
# View(out.wcdt.snv)

### miRNA: NA
 
# wcdt.proj.mirna <- GDCquery(
#   project = "WCDT-MCRPC",
#   data.category = "Transcriptome Profiling",
#   data.type = "miRNA Expression Quantification",
#   access = "controlled"
# )
# out.wcdt.mirna <- getResults(wcdt.proj.mirna)
# View(out.wcdt.mirna)



###  Methylation Data: Methylation Data NA 
wcdt.query.methylation <- GDCquery(project = "WCDT-MCRPC",
                            data.category = "DNA Methylation",
                            data.type = "Methylation Beta Value",
                            platform = "Illumina Human Methylation 450",
                            access = "Open")
out.wcdt.meth <- getResults(wcdt.query.methylation)



### Somatic Mutation: NA
wcdt.query.snv <- GDCquery(project = "WCDT-MCRPC",
                            data.category = "Simple Nucleotide Variation",
                            data.type = "Masked Somatic Mutation",
                            workflow.type = "MuTect2",
                            access = "controlled")
