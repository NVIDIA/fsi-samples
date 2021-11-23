S32407 is a repo with necesssary files for Part1.2021.ipynb and Part2.2021.ipynb Python Jupyter notebooks for the
course: How to Use GPUs for More Accurate Backtesting Equity Investment Strategies.

Best to clone this repository to obtain the R, Python, and .sh files into a working directory.
The majority of the files here are concerned with creating a FinAnalytics directory below the
working one if the working one is not already called that and populating it with 3 years of
daily prices quotes using get.hist.quote() from R to dynamically download using two tickers files
known as NYSEclean.txt and NASDAQ.txt.

Once there is a FinAnalytics directory with a subdir MVO3.yyyy.mm with NYSE and NASDAQ subdirs
and prices.csv files in each of these 2 subdirs, created by the Python script coalescePrices.py,
the Part 1 and Part 2 Python Jupyter notebooks should run correctly.
Note: there should be a symbolic link created at the equivalent level to FinAnalytics called data.

Note that for Part 2 the strategy is to read a back-in-time directory, say, 2020.08 for the
3 year historical Sharpe Ratio analysis then roll forward to, say, 2020.11 in order to apply
the strategy for 3 months forward from 2020.08, namely going long a small list of stocks.
This then requires two subdirs MVO3.2020.08 and MVO3.2020.11 in order for Part2.2021.ipynb
to run properly.

The book Financial Analytics with R explains the acquirePrices() logic in Chapter 4,
available at http://cambridge.org/FAR

NOTE: the data provider does limit download speeds by ticker.

Use the command:
$ sh run.sh
to run the process, controlled by the R file createDirOfPrices.R.
Edit this R file at the 3 points marked #EDIT HERE to select other date ranges of interest
and, once saved, use sh run.sh again.
