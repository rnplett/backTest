from tradeSystem import *

# create a trade system object
tSys = tradeSystem()

tSys.ticList = ["XLE"]

# get last 10 years of daily stock data
#tSys.daGoogleCSV()

# run system to build a tSys.trades data frame
#tSys.tsRandomTrades()
#tSys.tsAssetAllocationDalio(period=62)
#tSys.tsOpeningGap(threshold=0.1,testLen=1000)
tSys.tsMATrendHold()
#tSys.tsMATrendTrail(testLen=200)
tSys.tsMATrendSqeezeTrail(testLen=2000)

# calculate the trade returns and add them to the trades data frame
tSys.anReturns(period=21)

#tSys.pltTradeGraph()
tSys.pltTradeHistogram()

