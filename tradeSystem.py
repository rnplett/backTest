import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import datetime
from pandas import DataFrame
import sys
from inputs.settings import *
import quandl


NO_ATTRIBUTES_SET=object()
quandl.ApiConfig.api_key = QUANDL_API_KEY

class tradeSystem(object):
    """
    object to standardize the definition and processing of a trading system.
    """

    def __init__(self, **kwargs):
        """

        :param id: master reference, has to be an immutable type
        :param kwargs: other attributes which will appear in list returned by attributes() method
        """

        self.ticList = ["SPY","TLT","GLD"]

        attr_to_use=self.attributes()

        for argname in kwargs:
            if argname in attr_to_use:
                setattr(self, argname, kwargs[argname])
            else:
                print("Ignoring argument passed %s: is this the right kind of object? If so, add to .attributes() method" % argname)

    #other standard functions

    def attributes(self):
        ## should return a list of str here
        ## eg return ["thingone", "thingtwo"]
        return ["ticList","data","trades"]

    def _name(self):
        return "Trading System object - "

    def __repr__(self):

        attr_list = self.attributes()
        if attr_list is NO_ATTRIBUTES_SET:
            return self._name()

        return self._name() + " ".join(
            ["%s: %s" % (attrname, str(getattr(self, attrname))) for attrname in attr_list
             if getattr(self, attrname, None) is not None])

    def daGoogleCSV(self,sym='SPY',start='Jan+1,+2008', end=None):
        try:
            p = pd.read_csv('https://finance.google.com/finance/historical?q=' + sym + '&startdate=' + start + '&output=csv')
        except:
            try:
                p = pd.read_csv('https://finance.google.com/finance/historical?q=NYSE:' + sym + '&startdate=' + start + '&output=csv')
            except:
                print(sym + " - Google lookup error")

        p = p[::-1]
        p = p.reset_index(drop=True)
        self.data = p

    def daQuandl(self,sym='SPY'):
        try:
            p = pd.read_csv('https://www.quandl.com/api/v3/datasets/WIKI/' + sym + '/data.csv?api_key=' + QUANDL_API_KEY)
        except:
            print(sym + " - Quandl lookup error")

        p = p[::-1]
        p = p.reset_index(drop=True)
        self.data = p


    def tsRandomTrades(self, size=200, xLow=0, xHigh=500, xRange=20, yLow=50, yHigh=70, yRange=3):
        t = DataFrame(np.random.choice(self.ticList, size, replace=True), columns=['sym'])
        t['x1'] = DataFrame(np.random.choice(np.arange(xLow,xHigh-xRange), size=size, replace=True))
        t['x2'] = t['x1'] + np.random.choice(np.arange(xRange), size=size, replace=True)
        t['y1'] = DataFrame(np.random.uniform(low=yLow+yRange, high=yHigh-yRange, size=size))
        t['y2'] = t['y1'] + np.random.choice(np.arange(-yRange,yRange),size=size, replace=True)
        self.trades = t

    def tsAssetAllocationDalio(self, weightList=[0.3, 0.15, 0.55], testLen=2000, period=5  ):

        self.trades = DataFrame(columns=["sym", "x1", "x2", "y1", "y2"])
        w = {}
        for i in np.arange(0,len(self.ticList)):
            w[self.ticList[i]]=weightList[i]

        for sym in self.ticList:

            self.daGoogleCSV(sym)
            p = self.data

            o = p.loc[-testLen:,'Open']
            h = p.loc[-testLen:,'High']
            l = p.loc[-testLen:,'Low']
            c = p.loc[-testLen:,'Close']

            buyPt = {}
            sellPt = {}

            for i in np.arange(0,testLen,period):
                buyPt[i] = o.iloc[i]
                sellPt[i+period] = o.iloc[i+period]

            for i in np.arange(len(buyPt)):
                try:
                    y1 = float(buyPt[sorted(buyPt)[i]])
                    y2 = float(sellPt[sorted(sellPt)[i]])
                    self.trades = self.trades.append([{"sym": sym, "x1": sorted(buyPt)[i],
                                             "x2": sorted(sellPt)[i],
                                             "y1": y1,
                                             "y2": y2}], ignore_index=True)
                    self.trades = self.trades.append([{"sym": 'comboDalio3', "x1": sorted(buyPt)[i],
                                                       "x2": sorted(sellPt)[i],
                                                       "y1": y1,
                                                       "y2": y1 + (y2-y1)*w[sym]}], ignore_index=True)
                except:
                    continue

    def tsOpeningGap(self, threshold=0.1, commission=0.01, testLen=2000):

        # there is error in the assumptions made about hitting target first vs hitting stop first
        # when both occur in the same period.

        self.trades = DataFrame(columns=["sym", "x1", "x2", "y1", "y2"])

        for sym in self.ticList:
            self.daGoogleCSV(sym,start='Jan+1,+2000')
            p = self.data

            d = p.loc[-testLen:,'Date']
            o = p.loc[-testLen:,'Open']
            h = p.loc[-testLen:,'High']
            l = p.loc[-testLen:,'Low']
            c = p.loc[-testLen:,'Close']

            #print(p.head())

            for i in np.arange(1,testLen):
                try:
                    d1 = float(c.iloc[i - 1])
                    c0 = float(c.iloc[i-1])
                    o1 = float(o.iloc[i])
                    L1 = float(l.iloc[i])
                    h1 = float(h.iloc[i])
                    c1 = float(c.iloc[i])
                    #t = (o1+c0)/2
                    t = c0
                    s = o1 + (o1-t)
                    x1 = i
                    x2 = i + 1

                    # short play when the gap is up more than the threshold
                    if (100*(o1-c0)/c0 > threshold):
                        y2=o1
                        if L1 < t:
                            y1 = t
                        else:
                            if h1 > s:
                                y1 = s
                            else:
                                y1 = c1

                        self.trades = self.trades.append([{"sym": sym, "x1": x1, "x2": x2, "y1": y1, "y2": y2}],
                                                             ignore_index=True)

                    # long play when the gap is down more than the threshold
                    if (100*(c0-o1)/c0 > threshold):
                        y1 = o1
                        if h1 > t:
                            y2 = t
                        else:
                            if h1 < s:
                                y2 = s
                            else:
                                y2 = c1


                        self.trades = self.trades.append([{"sym": sym,"x1": x1,"x2": x2,"y1": y1,"y2": y2}],
                                                         ignore_index=True)
                except:
                    continue

    def tsMATrendHold(self, commission=0.01, testLen=1000):

        # there is error in the assumptions made about hitting target first vs hitting stop first
        # when both occur in the same period.

        self.trades = DataFrame(columns=["sym", "x1", "x2", "y1", "y2"])

        for sym in self.ticList:
            self.daQuandl(sym)
            p = self.data

            d = p.loc[:, 'Date']
            o = p.loc[:, 'Open']
            h = p.loc[:, 'High']
            l = p.loc[:, 'Low']
            c = p.loc[:, 'Close']

            holdTime = 10
            atrFactor = 1

            # print(p.head())

            for i in np.arange(50, testLen):
                try:
                    c0 = float(c.iloc[i])
                    m8 = c.iloc[(i-8):i].mean()
                    m21 = c.iloc[(i-21):i].mean()
                    m50 = c.iloc[(i-50):i].mean()
                    atr14 = (h.iloc[(i-14):i].astype(float)-l.iloc[(i-14):i].astype(float)).mean()
                    #print(atr14)

                    # long play when MA's stacked for up trend and closing price has pulled back below m8
                    if (m8 > c0 > m21 > m50):
                        x1 = i + 1
                        y1 = float(o.iloc[x1])
                        x2 = x1 + holdTime
                        y2 = float(o.iloc[x2])
                        L0 = l.iloc[x1:x2].astype(float).min()
                        if L0 < y1 - atr14*atrFactor:
                            y2 = y1 - atr14*atrFactor
                        self.trades = self.trades.append([{"sym": sym, "x1": x1, "x2": x2, "y1": y1, "y2": y2}],
                                                         ignore_index=True)

                    # short play when MA's stacked for down trend and closing price has pulled up to above m8
                    if (m8 < c0 < m21 < m50):
                        x2 = i + 1
                        y2 = float(o.iloc[x2])
                        x1 = x2 + holdTime
                        y1 = float(o.iloc[x1])
                        h0 = h.iloc[x2:x1].astype(float).max()
                        if h0 > y2 + atr14*atrFactor:
                            y1 = y2 + atr14*atrFactor
                        self.trades = self.trades.append([{"sym": sym, "x1": x1, "x2": x2, "y1": y1, "y2": y2}],
                                                 ignore_index=True)

                    self.trades = self.trades.append([{"sym": "price"+sym, "x1": i, "x2": i+1, "y1": float(o.iloc[i]),
                                                       "y2": float(o.iloc[i+1])}], ignore_index=True)

                except:
                    continue

    def tsMATrendTrail(self, commission=0.01, testLen=2000):

        # there is error in the assumptions made about hitting target first vs hitting stop first
        # when both occur in the same period.

        self.trades = DataFrame(columns=["sym", "x1", "x2", "y1", "y2"])

        for sym in self.ticList:
            self.daGoogleCSV(sym, start='Jan+1,+2008')
            p = self.data

            d = p.loc[-testLen:, 'Date']
            o = p.loc[-testLen:, 'Open']
            h = p.loc[-testLen:, 'High']
            l = p.loc[-testLen:, 'Low']
            c = p.loc[-testLen:, 'Close']

            holdTime = 10
            atrFactor = 1.5

            # print(p.head())

            for i in np.arange(50, testLen):
                try:
                    c0 = float(c.iloc[i])
                    m8 = c.iloc[(i-8):i].mean()
                    m21 = c.iloc[(i-21):i].mean()
                    m50 = c.iloc[(i-50):i].mean()
                    atr14 = (h.iloc[(i-14):i].astype(float)-l.iloc[(i-14):i].astype(float)).mean()
                    #print(atr14)

                    # long play when MA's stacked for up trend and closing price has pulled back below m8
                    if (m8 > c0 > m21 > m50):
                        x1 = i + 1
                        y1 = float(o.iloc[x1])
                        j = x1
                        while c0 > m21:
                            j += 1
                            c0 = float(c.iloc[j])
                            m21 = c.iloc[(j - 21):j].mean()
                        x2 = j
                        y2 = float(o.iloc[x2])
                        self.trades = self.trades.append([{"sym": sym, "x1": x1, "x2": x2, "y1": y1, "y2": y2}],
                                                         ignore_index=True)

                    # short play when MA's stacked for down trend and closing price has pulled up to above m8
                    if (m8 < c0 < m21 < m50):
                        x2 = i + 1
                        y2 = float(o.iloc[x2])
                        j = x2
                        while c0 < m21:
                            j += 1
                            c0 = float(c.iloc[j])
                            m21 = c.iloc[(j - 21):j].mean()
                        x1 = j
                        y1 = float(o.iloc[x1])
                        self.trades = self.trades.append([{"sym": sym, "x1": x1, "x2": x2, "y1": y1, "y2": y2}],
                                                 ignore_index=True)

                    self.trades = self.trades.append([{"sym": "price"+sym, "x1": i, "x2": i+1, "y1": float(o.iloc[i]),
                                                       "y2": float(o.iloc[i+1])}], ignore_index=True)

                except:
                    continue

    def tsMATrendSqeezeTrail(self, commission=0.01, testLen=2000):

        # there is error in the assumptions made about hitting target first vs hitting stop first
        # when both occur in the same period.

        self.trades = DataFrame(columns=["sym", "x1", "x2", "y1", "y2"])

        for sym in self.ticList:
            self.daQuandl(sym)
            p = self.data

            d = p.loc[-testLen:, 'Date']
            o = p.loc[-testLen:, 'Open']
            h = p.loc[-testLen:, 'High']
            l = p.loc[-testLen:, 'Low']
            c = p.loc[-testLen:, 'Close']

            atrFactor = 1.5

            # print(p.head())

            for i in np.arange(50, testLen):
                try:
                    c0 = float(c.iloc[i])
                    m8 = c.iloc[(i-8):i].mean()
                    m21 = c.iloc[(i-21):i].mean()
                    m50 = c.iloc[(i-50):i].mean()
                    atr14 = (h.iloc[(i-14):i].astype(float)-l.iloc[(i-14):i].astype(float)).mean()
                    std21 = c.iloc[(i-21):i].std()
                    squeeze = ( std21 < atr14 )
                    #print(std21, squeeze)

                    # long play when MA's stacked for up trend and closing price has pulled back below m8
                    if (m8 > c0 > m21 > m50):
                        x1 = i + 1
                        y1 = float(o.iloc[x1])
                        j = x1
                        if squeeze:
                            while c0 > m21:
                                j += 1
                                c0 = float(c.iloc[j])
                                m21 = c.iloc[(j - 21):j].mean()
                            x2 = j
                            y2 = float(o.iloc[x2])
                        else:
                            t = y1 + atr14
                            s = y1 - atr14
                            j = x1
                            while s < c0 < t:
                                j += 1
                                c0 = float(c.iloc[j])
                                s = max([s,c0 - atr14])
                            x2 = j
                            y2 = float(o.iloc[x2])
                        self.trades = self.trades.append([{"sym": sym, "x1": x1, "x2": x2, "y1": y1, "y2": y2}],
                                                         ignore_index=True)

                    # short play when MA's stacked for down trend and closing price has pulled up to above m8
                    if (m8 < c0 < m21 < m50):
                        x2 = i + 1
                        y2 = float(o.iloc[x2])
                        j = x2
                        if squeeze:
                            while c0 < m21:
                                j += 1
                                c0 = float(c.iloc[j])
                                m21 = c.iloc[(j - 21):j].mean()
                            x1 = j
                            y1 = float(o.iloc[x1])
                        else:
                            t = y2 - atr14
                            s = y2 + atr14
                            j = x2
                            while t < c0 < s:
                                j += 1
                                c0 = float(c.iloc[j])
                                s = min(s,c0 + atr14)
                            x1 = j
                            y1 = float(o.iloc[x1])
                        self.trades = self.trades.append([{"sym": sym, "x1": x1, "x2": x2, "y1": y1, "y2": y2}],
                                                 ignore_index=True)

                    self.trades = self.trades.append([{"sym": "price"+sym, "x1": i, "x2": i+1, "y1": float(o.iloc[i]),
                                                       "y2": float(o.iloc[i+1])}], ignore_index=True)

                except:
                    continue

    def anReturns(self, period=50):

        # calculate profit and percentage returns, then add them to the trades data frame
        self.trades['profit'] = self.trades['y2'] - self.trades['y1']
        self.trades['profPct'] = 100*self.trades['profit']/self.trades['y1']
        print("\nSummary of Trades by Percent Return for full test\n============================================\n")
        #print(self.trades[s.startswith('price') for s in self.trades.sym].pivot(columns='sym',values='profPct').describe())

        #summarize the profits by symbol and period
        self.trades['periodBin'] = self.trades['x2']/period
        self.trades['periodBin'] = self.trades['periodBin'].apply(np.int)*period
        g = self.trades.groupby(['periodBin','sym'])[['profPct']].sum().round(2).unstack()
        print("\n\nAggregate Percentage Returns per Period\n=======================================\n")
        print(g)
        print()
        print(g.describe())

    def pltTradeGraph(self):
        # assign a color to each symbol in the trade data
        c = {}
        colorList = "bgrcmykbgrcmykbgrcmykbgrcmyk"
        ticList = self.trades.sym.unique()
        for i in np.arange(0,len(ticList)):
            c[ticList[i]]=colorList[i]

        print(c)

        # plot each trade in the trades DataFrame with a line segment
        fig = plt.figure(figsize=(20,12))
        ax = fig.add_subplot(1,1,1)
        ax.text(15,15,c)
        for i in self.trades.index:
            if not self.trades.sym[i].startswith('combo'):
                if self.trades.sym[i].startswith('price'):
                    ax.plot([self.trades.x1[i], self.trades.x2[i]], [self.trades.y1[i], self.trades.y2[i]],
                            marker='.', color=c[self.trades.sym[i]], linewidth=2.0)
                else:
                    if self.trades.x1[i] > self.trades.x2[i]:
                        ax.plot([self.trades.x1[i], self.trades.x2[i]], [self.trades.y1[i], self.trades.y2[i]],
                                     marker='<', color=c[self.trades.sym[i]], linewidth=2.0)
                    else:
                        ax.plot([self.trades.x1[i], self.trades.x2[i]], [self.trades.y1[i], self.trades.y2[i]],
                                marker='o', color=c[self.trades.sym[i]], linewidth=2.0)

        plt.show()

    def pltTradeHistogram(self):
        # assign a color to each symbol in the trade data
        c = {}
        colorList = "bgrcmykbgrcmykbgrcmykbgrcmyk"
        ticList = self.trades.sym.unique()
        fig = plt.figure(figsize=(20, 12))
        g = self.trades.groupby(['periodBin','sym'])[['profPct']].sum().round(2).unstack()
        for i in np.arange(0, len(ticList)):
            if not ticList[i].startswith('price'):
                c[ticList[i]] = colorList[i]
                plt.hist(self.trades.profPct[self.trades.sym == ticList[i]].values, bins=200, alpha=0.4, label=ticList[i], range=(-10,10))
            plt.legend(loc='upper right')
        plt.show()
