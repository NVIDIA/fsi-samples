#/bin/bash

# download the mortgage dataset, untar and remove some data.
mkdir data
cd data/
wget http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_2000-2016.tgz
tar xzvf mortgage_2000-2016.tgz
cd acq
rm *2000* *2001* *2002* *2003* *2004* *2005* *2006* *2013* *2014* *2015* *2016*
cd ../perf
rm *2000* *2001* *2002* *2003* *2004* *2005* *2006* *2013* *2014* *2015* *2016*
