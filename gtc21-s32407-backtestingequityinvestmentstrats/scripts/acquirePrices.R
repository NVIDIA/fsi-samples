isPlotInAdjCloses <- FALSE
isCacheEnabled    <- TRUE

readExchSymbols <- function(fileName) {
  frame <- read.csv(fileName,header=TRUE,sep="\t")
  return(as.character(frame[,1]))
}

createDirs <- function(dir,isSubDir=TRUE) {
  #check for the two subdirs if isSubDir TRUE
  mainDir <- paste(homeuser,"/FinAnalytics/",sep="")
  destDir <- paste(mainDir,dir,sep="")
  if (!file.exists(destDir))
    dir.create(file.path(destDir))
  setwd(file.path(destDir))
  if(isSubDir) {
    f1 <- "NYSEclean.txt"
    f2 <- "NASDAQclean.txt"
  
    NYSEsubDir <- paste(destDir,"/NYSE",sep="")
    if (!file.exists(NYSEsubDir))
      dir.create(file.path(NYSEsubDir))
    if(!file.exists(paste(NYSEsubDir,"/NYSEclean.txt",sep="")))
      file.copy(paste0(homeuser,"/FinAnalytics/",f1),
             NYSEsubDir)
    
    NASDAQsubDir <- paste(destDir,"/NASDAQ",sep="")
    if (!file.exists(NASDAQsubDir))
      dir.create(file.path(NASDAQsubDir))
    if(!file.exists(paste(NASDAQsubDir,"/NASDAQclean.txt",sep="")))
      file.copy(paste0(homeuser,"/FinAnalytics/",f2),
             NASDAQsubDir)
  } else {
    f <- paste(dir,"clean.txt",sep="")
    if(!file.exists(paste(destDir,"/",f,sep="")))
      if(file.exists(paste(mainDir,"/",f,sep="")))
        file.copy(paste0(homeuser,"/FinAnalytics/",f),".")
  }
}

readSubDirs <- function(dir,isSubDir=TRUE) {
  if(isSubDir) {
    #Case: 2 sub-dirs: NYSE and NASDAQ
    #Return 3 results, the last being a large vec
    setwd(paste(homeuser,"/FinAnalytics/",dir,"/NYSE",sep=""))
    lab <- readExchSymbols("NYSEclean.txt")
    D1 <- length(lab)
    print(D1)
    setwd(paste(homeuser,"/FinAnalytics/",dir,"/NASDAQ",sep=""))
    lab2 <- readExchSymbols("NASDAQclean.txt")
    lab <- append(lab,lab2)
    D2 <- length(lab2)
    print(D2)
    list(D1,D2,as.character(lab))
  } else {
    setwd(paste(homeuser,"/FinAnalytics/",dir,sep=""))
    lab <- readExchSymbols(paste(dir,"clean.txt",sep=""))
    D <- length(lab)
    print(D)
    list(D,as.character(lab))
  }
}

acquirePrices <- function(prices,lab,len,D,D1,D2,dir,
                 start,end,isSubDir=TRUE,verbose=TRUE) {
  isSuccessfulQuote <- FALSE
  for(d in 1:D) {
    if(d == 1 || (isSubDir && d == (D1+1)))
      if(d == 1 && isSubDir) {
        setwd(paste(homeuser,"/FinAnalytics/",dir,"/NYSE",sep=""))
        unlink('bad*')
        print(paste("NYSE=======:",d))
      } else if(d == (D1+1) && isSubDir) {
        setwd(paste(homeuser,"/FinAnalytics/",dir,"/NASDAQ",sep=""))
        unlink('bad*')
        print(paste("NASDAQ=======:",d))
      } else {
        setwd(paste(homeuser,"/FinAnalytics/",dir,sep=""))
        unlink('bad*')
        print(paste("ETF==========:",d))
      }
    if(verbose) print(paste(d,lab[d]))
    fileName = paste("cached",lab[d],".csv",sep="")
    usingCacheThisFileName <- FALSE
    if(file.exists(fileName)) {
      usingCacheThisFileName <- TRUE
      pricesForStock <- read.csv(fileName,header=TRUE,sep="")[,1]
      if(!is.na(pricesForStock[1]))
        isSuccessfulQuote <- TRUE
    }
    if(!usingCacheThisFileName ||
         (usingCacheThisFileName && length(pricesForStock) != len)) {
      usingCacheThisFileName <- FALSE
      tryCatch( {
        #print(start);print(end)
        Sys.sleep(1)
        pricesForStock <- get.hist.quote(lab[d],quote="Adj",
                                         start=start,end=end)
        if(!is.na(pricesForStock[1]))
          isSuccessfulQuote <- TRUE
      }, error = function(err) {
        print(err);cat(lab[d],file="badsyms.txt",
                       append=TRUE,sep="\n")
        isSuccessfulQuote <- FALSE
      } )
    }
    if(length(pricesForStock) == len) {
      prices[,d] <- pricesForStock
      if(sum(is.na(prices[,d])) > 0 || (sum(is.na(prices[,d-1])) == 0 &&
            d > 1 && prices[1,d] == prices[1,d-1])) {
        print(paste(lab[d],"has NA prices"))
        cat(lab[d],file="badsyms.txt",
            append=TRUE,sep="\n")
        isSuccessfulQuote <- FALSE
      }
    } else {
      cat(lab[d],file="badsyms.txt",append=TRUE,sep="\n")
    }
    if(!isSuccessfulQuote)
      cat(lab[d],file="badsyms.txt",append=TRUE,sep="\n")
    if(isPlotInAdjCloses) {
      if(d == 1)
        plot(prices[,d]/prices[1,d],type="l",col="blue",ylim=c(.2,6))
      else
        lines(prices[,d]/prices[1,d],type="l",col="blue")
      text(len,(prices[len,d]/prices[1,d]),lab[d],cex=.6)
    }
    if(isCacheEnabled && !usingCacheThisFileName &&
         isSuccessfulQuote) {
      #save redundant re-write
      fileName = paste("cached",lab[d],".csv",sep="")
      print(fileName)
      write.csv(prices[,d],file=fileName,row.names = FALSE)
    }
    isSplitAdjusted = TRUE
  }
  prices
}
