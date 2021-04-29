#////////////////////////////////////////////////////////////////////////////
#//
#// Copyright (C) NVIDIA Corporation.  All rights reserved.
#//
#// NVIDIA Sample Code
#//
#// Please refer to the NVIDIA end user license agreement (EULA) associated
#// with this source code for terms and conditions that govern your use of
#// this software. Any use, reproduction, disclosure, or distribution of
#// this software and related documentation outside the terms of the EULA
#// is strictly prohibited.
#//
#////////////////////////////////////////////////////////////////////////////

findAllPrices <- function(dir,start=NA,end=NA,
            havePrices=F,needToAcquire=F,isSubDir=T) {
  isPlotInAdjCloses = FALSE
  createDirs(dir,isSubDir=isSubDir)
  res <- readSubDirs(dir,isSubDir=isSubDir)
  isCacheEnabled <- TRUE
  if(isSubDir) {
    D1  <- res[[1]]
    D2  <- res[[2]]
    lab <- res[[3]]
    D <- D1 + D2
  } else {
    D <- res[[1]]; lab <- res[[2]]
  }
  print(paste("findAllPrices",dir,getwd(),start,end,D1,D2,D))
  if(havePrices) { #was: havePrices
    print("have prices.")
    return(list(saveNPrices,saveNLab))
  } else {
    if(needToAcquire) {
      len <- length(get.hist.quote("A",quote="AdjClose",start,end))
      prices <- matrix(rep(NA,len*D),nrow=len,ncol=D)
      prices <- acquirePrices(prices,lab,len,D,D1,D2,
               start=start,end=end,dir,isSubDir=isSubDir)
    } else {
      print(dir)
      return(acquireCoalescedPrices(dir,isSubDir=isSubDir))
    }
  }
}
