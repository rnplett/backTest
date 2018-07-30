import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import datetime
from pandas import DataFrame
from returnAnalysis import *

# Get data from 'yahoo' API
#***************************
start = datetime.datetime(2005, 1, 1)
# end = datetime.datetime(2010, 1, 1)
end = datetime.datetime.now()

# ticList = ['ALXN','CSCO','AAL','DFS','PCLN','GS','GD','WYN','ZLTQ','MUR','COF','TROW','DIS','AAPL']
# ticList = ['SPY','GLD','TLT']
weight = [0.30, 0.15, 0.55]
ticList = ['SSO', 'UGL','TMF']

# Run System against the Data
#*****************************
period = 2000
lookback = 62
rtnTotal = 0
tradesTotal = 0
bnhTotal = 0
rtnOptTotal = 0
errorCount = 0
trades = DataFrame(columns=["sym","buyPtX","sellPtX","buyPtY","sellPtY"])

fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(1,1,1)

for sym in ticList:
    try:
        p = pd.read_csv('https://finance.google.com/finance/historical?q=' + sym + '&startdate=Dec+27%2C+2008&output=csv')
    except:
        try:
            p = pd.read_csv('https://finance.google.com/finance/historical?q=NYSE:' + sym + '&startdate=Dec+27%2C+2008&output=csv')
        except:
            print(sym + " - Google lookup error")

    p = p[::-1]
    p = p.reset_index(drop=True)

    o = p.loc[-period:,'Open']
    h = p.loc[-period:,'High']
    l = p.loc[-period:,'Low']
    c = p.loc[-period:,'Close']

    buyPt = {}
    sellPt = {}
    Position = 0

    #
    # System starts here to assign buy and sell points
    for i in np.arange(0,period,lookback):
        buyPt[i] = o.iloc[i]
        sellPt[i+lookback] = o.iloc[i+lookback]
    # system ends here

    profit = 0
    optProfit = 0
    for i in np.arange(len(buyPt)):
        textX = sorted(buyPt)[i]
        textY = float(o.iloc[i])
        try:
            trades = trades.append([{"sym": sym, "buyPtX": sorted(buyPt)[i],
                                     "sellPtX": sorted(sellPt)[i],
                                     "buyPtY": float(buyPt[sorted(buyPt)[i]]),
                                     "sellPtY": float(sellPt[sorted(sellPt)[i]])}], ignore_index=True)
            ax.plot([sorted(buyPt)[i], sorted(sellPt)[i]], [buyPt[sorted(buyPt)[i]], sellPt[sorted(sellPt)[i]]],
                    marker='o', color='g', linewidth=2.0)
            ptProfit = (float(sellPt[sorted(sellPt)[i]]) - float(buyPt[sorted(buyPt)[i]]))/float(buyPt[sorted(buyPt)[i]])
            profit = profit + ptProfit
            optProfit = optProfit - textY * 0.01 * 0.03 * (sorted(sellPt)[i] - sorted(buyPt)[i]) # Theta = 0.01
            if ptProfit > 0:
                optProfit = optProfit + 0.75*ptProfit
            else:
                optProfit = optProfit + 0.25*ptProfit
        except:
            errorCount += 1
            continue

    #     #ax.text(textX,textY,"{:10.2f}".format(ptProfit),size='small')

    annualFactor = 252.01/period
    rtnProfit = 100*profit/float(o.iloc[0])*annualFactor # should textY this be the starting value?
    tradesPerYear = len(buyPt)*annualFactor
    rtnBnH = 100 * (float(c.iloc[period]) - float(o.iloc[0])) / float(o.iloc[0]) * annualFactor
    rtnOptProfit = 100*optProfit/(float(o.iloc[0])/10)*annualFactor
    print sym + ":{:10.2f}% average annual return".format(rtnProfit) + \
          "   {:3.1f} round trip trades per year".format(tradesPerYear) + \
          "   {:10.2f}% BnH Return".format(rtnBnH) + \
          "   {:10.2f}% Option Return".format(rtnOptProfit)
    rtnTotal = rtnTotal + rtnProfit
    tradesTotal = tradesTotal + tradesPerYear
    if not math.isnan(rtnBnH): bnhTotal = bnhTotal + rtnBnH

print(" ")
print("Aggregate:{:10.2f}% average annual return".format(rtnTotal/len(ticList)) + \
      "   {:3.1f} round trip trades per year".format(tradesTotal/len(ticList)) + \
      "   {:3.1f}% BnH annual return".format(bnhTotal/len(ticList)))
print(" ")

trades["return"] = 100*(trades.sellPtY - trades.buyPtY)/trades.buyPtY

comboProfit = pd.Series([])
for i in trades.index[trades.sym == ticList[0]]:
    try:
        s = trades[trades.buyPtX == trades.buyPtX[i]]["return"].values
        #print(s)
        s = s*weight
        #print(s)
        s = sum(s)
        #print(s)
        comboProfit[i] = s
    except:
        continue

print(trades.describe())
print(" ")
print(trades.head())
print(" ")
print(comboProfit.describe())
print(" ")
print(comboProfit.head())
plt.show()

print "*************************************************************** "
print "*** MONTE CARLO SIMULATION - IF TRADE ORDER DOESN'T MATTER  *** "
print "*************************************************************** "


# Monte Carlo Simulation - Long Stock
#**************************************
mcSim = DataFrame(columns=[])
print " "
print("*** MONTE CARLO SIMULATION - ComboProfit ({} Trades per Sim) ***".format(int(tradesPerYear)))

no_sims = 1000
sample_size = int(tradesPerYear)
profit=pd.Series([])
for i in np.arange(no_sims):
    #s = random.sample(comboProfit.index,sample_size)
    r = random.sample(comboProfit.index,1)
    s = r + [r[0] + 1] + [r[0] + 2] + [r[0] + 3]
    profit[i] = 1.0
    #print(comboProfit[s])
    for j in comboProfit[s]:
        #print(j)
        #print(profit[i])
        #profit[i] = profit[i] + j
        profit[i] = profit[i]*(1+j/100)
    #print(profit[i])
    profit[i] = (profit[i] - 1)*100
mcSim[sym]=profit

print mcSim.describe()
print " "
print mcSim.head()

# Monte Carlo Simulation - Long Stock
#**************************************

mcSim = DataFrame(columns=[])
print " "
print("*** MONTE CARLO SIMULATION - LONG STOCK ({} Trades per Sim) ***".format(int(tradesPerYear)))
for sym in ticList:
    no_sims = 1000
    sample_size = int(tradesPerYear)
    profit=pd.Series([])
    for i in np.arange(no_sims):
        s = random.sample(trades[trades["sym"]==sym].index,sample_size)
        profit[i] = sum(trades.iloc[s]['return'])
    mcSim[sym]=profit

print mcSim.describe()
print " "
print mcSim.head()

# Monte Carlo Simulation - Options
#************************************
print " "
print "*** MONTE CARLO SIMULATION - LONG CALLS ***"
mcSim = DataFrame(columns=[])
for sym in ticList:
    no_sims = 1000
    sample_size = 30
    profit=pd.Series([])
    for i in np.arange(no_sims):
        s = random.sample(trades.index,sample_size)
        # profit = [long stock profit]*[0.3|0.7]-[Trade Length]*[Theta]*[stock price]*[0.03]
        p = trades.iloc[s]["sellPtY"]-trades.iloc[s]["buyPtY"]
        d = trades.iloc[s]["sellPtX"]-trades.iloc[s]["buyPtX"]
        optFactor = np.repeat(0.7,sample_size)
        optFactor[p < 0] = 0.3
        thetaAmt = d*0.01*trades.ix[s]["buyPtY"]*0.03
        profit[i] = sum((p*optFactor-thetaAmt)/max(trades.ix[s]["buyPtY"]*0.03))*100
    mcSim[sym]=profit

print mcSim.describe()
print " "
print mcSim.head()