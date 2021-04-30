library(tseries)
source("acquirePrices.R")
source("acquireCoalescedPrices.R")
source("findAllPrices.R")
#required: ticker symbol files
if(!file.exists("NYSEclean.txt") || !file.exists("NASDAQclean.txt"))
  error("Need to download both NYSEclean.txt and NASDAQclean.txt from repository.")
  
split_path <- function(path) {
  setdiff(strsplit(path,"/|\\\\")[[1]], "")
}
#required: homeuser user path variable and .../FinAnalytics subdir
homeuser <- getwd()
subDir <- "FinAnalytics"
dataPath <- "data"

#check for and create .../FinAnalytics if not present
if (rev(strsplit(homeuser,"/")[[1]])[1] == subDir) { #Need to create it below c.w.dir
  strippedPath = paste0("/",paste(rev(rev(strsplit(homeuser,"/")[[1]][-1])[-1]),collapse="/"))
  homeuser <- strippedPath
} else {
  dir.create(file.path(homeuser, subDir))
  setwd(file.path(homeuser, subDir))
  current <- homeuser
  new <- getwd()
  filelist <- list.files(current, "*clean.txt")
  # copy the 2 files to the new folder
  file.copy(paste0(current,"/",filelist[1]), paste0(new,"/",filelist[1]))
  file.copy(paste0(current,"/",filelist[2]), paste0(new,"/",filelist[2]))
  filelist <- list.files(current, "coalescePrices.py")
  file.copy(paste0(current,"/",filelist[1]), paste0(new,"/",filelist[1]))
  dataPath = "../data"
}

#create symblink file to make FinAnalytics and data both usable
### Done inn docker build now
#if(file.exists(dataPath)) {
#  print("Conflict cannot crate linked called ../data: file exists.")
#} else {
#  system(paste0("ln -s ",file.path(homeuser, subDir)," ",dataPath))
#}

#obtain entire market from start to end dates: daily adjClose
dir <- "MVO3.2020.05" #EDIT HERE
res <- findAllPrices(dir = dir,
                     start = "2017-05-28", #EDIT HERE
                     end   = "2020-05-27",needToAcquire=T) #EDIT HERE
#make 2 very wide data frame .csv files using python script
setwd("../NYSE")
system("python ../../coalescePrices.py .")
setwd("../NASDAQ")
system("python ../../coalescePrices.py .")

#obtain entire market from start to end dates: daily adjClose
dir <- "MVO3.2020.08" #EDIT HERE
res <- findAllPrices(dir = dir,
                     start = "2017-08-28", #EDIT HERE
                     end   = "2020-08-27",needToAcquire=T) #EDIT HERE
#make 2 very wide data frame .csv files using python script
setwd("../NYSE")
system("python ../../coalescePrices.py .")
setwd("../NASDAQ")
system("python ../../coalescePrices.py .")