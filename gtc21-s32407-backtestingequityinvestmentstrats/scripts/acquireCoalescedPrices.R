acquireCoalescedPrices <- function(dir,isSubDir=TRUE) {
  utilName <- paste0("python3 ",homeuser,
          "/FinAnalytics/misc/coalescePrices.py ")
  
  if(isSubDir) { path <- paste0(homeuser,"/FinAnalytics/",dir,"/NYSE/") }
  else { path <- paste0(homeuser,"/FinAnalytics/",dir)  }
  print(paste("acquireCoalescedPrices path =",path))
  
  if(!file.exists(paste0(path,"prices.csv")))
    system(paste0(utilName,path))
  setwd(path)
  mat1 <- read.csv("prices.csv",
                   sep=',',header=F,stringsAsFactors=F)
  lab1 <- as.character(mat1[1,])
  prices1 <- apply(
    as.matrix(mat1[2:nrow(mat1),],nrow=(nrow(mat1)-1),ncol=ncol(mat1)),
    c(1,2),as.numeric)
  if(isSubDir) {
    path <- paste0(homeuser,"/FinAnalytics/",dir,"/NASDAQ/")
    if(!file.exists(paste0(path,"prices.csv")))
      system(paste0(utilName,path))
    setwd(path)
    mat2 <- read.csv("prices.csv",
                     sep=',',header=F,stringsAsFactors=F)
    lab2 <- as.character(mat2[1,])
    prices2 <- apply(
      as.matrix(mat2[2:nrow(mat2),],nrow=(nrow(mat2)-1),ncol=ncol(mat2)),
      c(1,2),as.numeric)
    lab <- c(lab1,lab2)
    prices <- cbind(prices1,prices2)
  } else {
    lab <- lab1
    prices <- prices1
  }
  list(prices,lab)
}
