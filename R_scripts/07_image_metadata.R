# Ali M, N.

# Step 0: Load Libraries ----------------
library(httr)
library(jsonlite)
library(utils)
library(oro.dicom)
library(dplyr)
# Step 1: Read metadata --------------------
df <- read.csv("./softwares/images/manifest-LKlXeErz8879128910887416605/metadata.csv", header = TRUE)
View(df)
# Step 2: View Images
first.images <- list.files("softwares/images/manifest-LKlXeErz8879128910887416605/TCGA-PRAD/TCGA-EJ-5495/01-30-1992-NA-MRI PELVIS-73360/9.000000-SAG SPGR POST-72857/")

first.images %>% 
  length()
counter <- 1
for(i in 1:length(first.images)){
  path <- "softwares/images/manifest-LKlXeErz8879128910887416605/TCGA-PRAD/TCGA-EJ-5495/01-30-1992-NA-MRI PELVIS-73360/9.000000-SAG SPGR POST-72857/"
  path.extended <- paste0(path, first.images[i])
  print(path.extended)
  # print(path.extended
  dcm.file <- readDICOMFile(path.extended)
  # print(dcm.file$hdr)
  # print(dcm.file$img)
  ## Plot the image
  image(dcm.file$img, main = paste0("Image no: ",i))
  counter <- counter + 1
  if(counter == 10)
    break
}

list.files("softwares/images/manifest-LKlXeErz8879128910887416605/TCGA-PRAD/TCGA-EJ-5495/01-30-1992-NA-MRI PELVIS-73360/1.000000-LOCALIZER-28104/")


paste0(test, first.images[1])
