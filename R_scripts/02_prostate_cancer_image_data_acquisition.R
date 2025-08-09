### 1. Install Required Packages
install.packages(c("httr", "jsonlite", "utils"))
install.packages("oro.dicom")


### 2. Load Libraries
library(httr)
library(jsonlite)
library(utils)
library(oro.dicom)
library(dplyr)


### 3. Set Base URL for TCIA API

base_url <- "https://services.cancerimagingarchive.net/services/v4/TCIA/query"

# #### 4. List Available Collections
res <- GET(paste0(base_url, "/getCollectionValues"))
collections <- fromJSON(content(res, "text", encoding = "UTF-8"))
print(collections)
# for(col in collections){
#   print(col)
#   if(col == "TCGA-PRAD"){
#     print("TCGA-PRAD is present")
#     break
#   }
# }
# 

### 5. Find Available Series for a Collection
collection <- "TCGA-PRAD"
res <- GET(paste0(base_url, "/getSeries?Collection=", collection))
series_info <- fromJSON(content(res, "text", encoding = "UTF-8"))
head(series_info)
View(series_info)

### 6. Download Image Series
series_uid <- series_info$SeriesInstanceUID
series_info$SeriesInstanceUID %>% length()

### Download in chunks
### First 30

first.30.uid <- series_uid[1:30]

for(i in first.30.uid){
  print(paste("Printing series uid: ", i))
  options(timeout = 300)
  image.url <- paste0(base_url,  "/getImage?SeriesInstanceUID=", i)
  download.file(image.url, destfile = "TCIA_seriesfirst30.zip", mode = "wb")
  print("Image Downloaded")
}

### Second 30
second.30.uid <- series_uid[31:60]
count.second <- 1
for(i in second.30.uid){
  print(paste0("Count: ", count.second))
  print(paste("Printing series uid: ", i))
  options(timeout = 300)
  image.url <- paste0(base_url,  "/getImage?SeriesInstanceUID=", i)
  download.file(image.url, destfile = "TCIA_seriessecond30.zip", mode = "wb")
  print("Image Downloaded")
  count.second <- count.second+1
}
 ### third 30
third.30.uid <- series_uid[60:90]
count.third <- 1
for(i in third.30.uid){
  print(paste0("Count: ", count.third))
  print(paste("Printing series uid: ", i))
  options(timeout = 300)
  image.url <- paste0(base_url,  "/getImage?SeriesInstanceUID=", i)
  download.file(image.url, destfile = "TCIA_seriesThird30.zip", mode = "wb")
  print("Image Downloaded")
  count.third <- count.third + 1
}

### fourth 30
fourth.30.uid <- series_uid[91:120]

for(i in second.30.uid){
  print(paste("Printing series uid: ", i))
  options(timeout = 300)
  image.url <- paste0(base_url,  "/getImage?SeriesInstanceUID=", i)
  download.file(image.url, destfile = "TCIA_seriessecond30.zip", mode = "wb")
  print("Image Downloaded")
}
### fifth 30

### sixth 30

### Rest


count <- 1
for(i in series_uid){
  print(paste0("Series: ",count))
  if(count == 36){
    count <- count + 1
    next
  }
  print(paste("Printing series uid: ", i))
  options(timeout = 300)
  image.url <- paste0(base_url,  "/getImage?SeriesInstanceUID=", i)
  download.file(image.url, destfile = "TCIA_series.zip", mode = "wb")
  print("Image Downloaded")
  count = count + 1
}

image.url <- paste0(base_url,  "/getImage?SeriesInstanceUID=", 37)
download.file(image.url, destfile = "TCIA_series.zip", mode = "wb")

image_url <- paste0(base_url, "/getImage?SeriesInstanceUID=", series_uid)


for (uid in series_uid) {
  image_url <- paste0(base_url, "?SeriesInstanceUID=", uid)
  destfile <- paste0("Series_", gsub("\\.", "_", uid), ".zip")  # Clean filename
  cat("Downloading:", uid, "\n")
  tryCatch({
    download.file(url = image_url, destfile = destfile, mode = "wb")
  }, error = function(e) {
    cat("Failed to download:", uid, "\n")
  })
}
# Download the ZIP file containing the DICOM images
download.file(image_url, destfile = "TCIA_series.zip", mode = "wb")


### 7. Unzip DICOM Files
unzip("TCIA_series.zip", exdir = "TCIA_images")

### Reading Images
dcm.files <- readDICOM("./TCIA_images/1-107.dcm")
str(dcm.files)

### Meta Data
dcm.files$hdr
dcm.files$img

### Plot the image data
image(dcm.files$img[[0]], col = gray(0:64/64), main = "PC Image")
