# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 19:28:47 2021

@author: Felhasználó
"""


# coding: utf-8

# In[2]:


#Installing stuff

#!pip install yfinance
#!pip install msgpack
#!pip install scipy



#Trying pandas_datareader instead of yahoo finance
import statistics
import yfinance as yf
from pandas_datareader import data as importer
import time
import datetime
from datetime import timedelta
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import fuckit




import random
from datetime import datetime





#Starting
start_time = time.time()


 #Plot
myPlots = {}


#significant technical levels for every 'good' ticker
significantLevels = {}


#Keeping the prices, volumes, moving averages, significant levels in this dict
goodData = {}
greatData = {}
myProfit = []
goodProfit = []
spyProfit = []


#Just to see if RSI<30 is worth calculating and actually a good strategy or not
greatRSI = []

#testing a parameter
row_364 = []

#Starting the outer 'Portfolio' or 'Test' run with random days
# =============================================================================
# randomDays = np.random.randint(100,1800,1)
# randomDays = np.ndarray.tolist(randomDays)
# =============================================================================

randomDays = [190]


for randomDay in randomDays:


    
    #Time settings
    #Define 'today' and 'startDate' to download the date from startDate to today
    #startDate is set to 365 days back from 'today'
    
    today = date.today() - timedelta(days= randomDay)
    startDate = today - timedelta(days=365)
    
    
    testDateStart = today + timedelta(days = 1)
    testDateEnd = today + timedelta(days = 60)
    testDays = (testDateEnd - testDateStart).days

    
    
    #Starting Python_Trading 
    # In[25]:
    
    
    #Getting all the tickers into one dataframe to be able to refer to them by positions: 'tickerList[i]'
    #'tickerList' is actually a dataframe so might be worth renaming in the future
    
    #So we download some kind of list from the stock exchange in a csv format, read this csv into 'tickerList' dataframe
    #In the next step we will use this tickerList and loop through it to query data from yahoo finance
    
    tickerList = pd.DataFrame()
    
    #tickerList1: tickers passed the first strategy, dataframe
    tickerList1 = pd.DataFrame() 
    tickerList2 = pd.DataFrame()
    #tickerPassed1: list of tickers passed the first strategy, this is for appending with the loop
    tickerPassed = []
    tickerPassed1 = []
    tickerPassed2 = []
    
    
    #Current price < SMA200
    awful = []
    
    bad = []
    
    normal = []
    
    good = []
    
    great = []
    
    
   
    
    
    
    
    #myFileObj = open(r"C:\Users\Felhasználó\Desktop\Machine Learning A-Z\Trading\Python_Trading\nasdaq_tickers.csv", encoding='latin-1')
    #myFileObj = open(r"C:\Users\Felhasználó\Desktop\Machine Learning A-Z\Trading\Python_Trading\nasdaq_aapl.csv", encoding='utf-8-sig')
    myFileObj = open(r"C:\Users\Felhasználó\Desktop\Machine Learning A-Z\Trading\Python_Trading\nasdaq_test.csv", encoding='utf-8-sig')
    #myFileObj = open(r"C:\Users\Felhasználó\Desktop\Machine Learning A-Z\Trading\Python_Trading\nasdaq_test_10.csv", encoding='utf-8-sig')
    #myFileObj = open(r"C:\Users\Felhasználó\Desktop\Machine Learning A-Z\Trading\Python_Trading\nasdaq_test_100.csv", encoding='utf-8-sig')
    #myFileObj = open(r"C:\Users\Felhasználó\Desktop\Machine Learning A-Z\Trading\Python_Trading\Mid_and_large_stocks.csv", encoding='latin-1')
    #myFileObj = open(r"C:\Users\Felhasználó\Desktop\Machine Learning A-Z\Trading\Python_Trading\Mid_and_large_test_10.csv", encoding='utf-8-sig')

    tickerList = pd.read_csv(myFileObj) 
    #tickerList
    
    
    
    
    
    
    #Spy
    try:
        spy = importer.DataReader('SPY', "yahoo", testDateStart, testDateEnd)
    except:
        pass
    
    
    
    spyList = spy.Close
    
    
    
    try:
        spyProfit.append(spyList[-1] / spyList[0])
    except:
        pass
        
    
    
    
    
        
    # Naming: - yf.Ticker dataframe is always 'tickerName' 
    #         - tickerName is querying yahoo finance using one ticker/symbol at a time
    #         - tickerName is then used to generate tickerData which contains the prices
    #         - Historical chart/price for ONE SPECIFIED TICKER at a time is always in 'tickerData'
    #         - 'tickerData' will get the prices from yahoo run through the first strategy (spit out pass or fail) and
    #           if it passes then store that ticker in something (df or list) and go to the next in the csv and do the same process
    #         - Anything that is calculated such as moving averages will be added to 'tickerData'
    #         - 'tickerData' will contain information from tickerData and calculated fields so that the given strategy (or signal)
    #           can always be calculated and evaluated on this 'tickerData' dataframe 
    #          - I switched from yf to pandas datareader, usage of yf was:
    #                tickerName = yf.Ticker(tickerList.Symbol[i]) # position 17 is AAPL so that will be used as an example
    #               tickerData = pd.DataFrame
    #               tickerData = tickerName.history(period ='1d', start = startDate , end = today)
                      
    
    
    
    
    
    #Outer most loop going through all the symbols
    #Every other strategy should be nested in the previous one
    
    
    for i in range(len(tickerList.Symbol)):
        
        tickerName = tickerList.Symbol[i] # position 17 is AAPL so that will be used as an example
        tickerData = pd.DataFrame
        try:
            tickerData = importer.DataReader(tickerName, "yahoo", startDate, today)
        except:
            continue
        
        
        
        
    
        
        #Strategy 1
        SMA200 = pd.DataFrame()
        SMA200['Close'] = tickerData['Close'].rolling(window = 200).mean()
        
        
        
        #First filter based on Close > SMA200
        #We reject the ticker if it's below it's 200 day moving average 
        #It goes to 'awful' so that we don't check again for 1-2 months
        #No else, just goes on to the next check
        #(we could also reject if a lot higher than SMA200 --> bubble?)
        
        if tickerData.Close[-1] < SMA200.Close[-1]:
            awful.append(tickerName)
            continue
        
        
         
            
        
        #Strategy 2
        #last close is lower than SMA100 --> goes to 'bad'
        SMA100 = pd.DataFrame()
        SMA100['Close'] = tickerData['Close'].rolling(window = 100).mean()
            
            
        if tickerData.Close[-1] < SMA100.Close[-1]*0.95:
            awful.append(tickerName)
            continue   
        
        
        
        
            
        #Strategy 3
        #Seeing if SMA200 is increasing or not
        #if not: goes to awful
        def trendSMA200(X_values, y_values, deg = 1):
            result = np.polyfit(X_values, list(y_values), deg = deg)
            slope = result[-2]
            return float(slope)
        
        beta = trendSMA200(pd.to_numeric(SMA200.Close[-40:].index), pd.to_numeric(SMA200.Close[-40:]) , deg = 1)
        
        if beta <= 0:
            awful.append(tickerName)
            continue
        else: 
            print("{} has an increasing SMA200 trend".format(tickerName))
                
            
        
        
    
        #Strategy 4
        #SMA100 is increasing or not
        #if not: goes to 'bad'
        
        def trendSMA100(X_values, y_values, deg = 1):
            result = np.polyfit(X_values, list(y_values), deg = deg)
            slope = result[-2]
            return float(slope)
        
        beta = trendSMA100(pd.to_numeric(SMA100.Close[-40:].index), pd.to_numeric(SMA100.Close[-40:]) , deg = 1)
        
        if beta <= 0:
            bad.append(tickerName)
            continue
        
        print("{} has an increasing SMA100 trend".format(tickerName))
        #Strategy 4 
        #Is SMA35 > SMA100
        #if not: goes to 'normal'
        
        SMA35 = pd.DataFrame()
        SMA35['Close'] = tickerData['Close'].rolling(window = 35).mean()
        
        
        if SMA35.Close[-1] < SMA100.Close[-1]:
            normal.append(tickerName)
            print("{} has SMA35 below SMA100".format(tickerName))
            
        
        else: 
            tickerPassed.append(tickerList.Symbol[i])
            
        print("{} has SMA35 above SMA100".format(tickerName))   
    # =============================================================================
    # Moving average strategy
    # 
    # Defining the 'buy_sell' function which returns 2 lists: buySig and sellSig
    # these are filled mostly with NaN-s but
    # at the point where the 2 moving averages cross each other one of the lists 
    # contains the price (which is tickerData['Close'] in this case)
    # The flag strats at -1 and when the shorter avg is above the longer it turns to 1
    # so when looping gets to the second "row/date/historical price" the flag
    #  is already 1 so we don't append anything to buySig
    # 
    # =============================================================================
       
        def crossSMA(tickerData):
             buy_sell_sig = [] #signal to buy: short SMA just crossed over long SMA
             flag = -1
         
             for i in range(len(tickerData)):
                 if SMA35.Close[i] > SMA100.Close[i]:
                     if flag !=1:
                         buy_sell_sig.append("1")           
                         flag = 1
                     else:
                         buy_sell_sig.append(np.nan)
                 elif SMA35.Close[i] < SMA100.Close[i]:
                     if flag!=0:
                         buy_sell_sig.append("0")
                         flag=0
                     else:
                         buy_sell_sig.append(np.nan)
                 else:
                     buy_sell_sig.append(np.nan)
             
             return(buy_sell_sig)
         
        
        tickerData['SMAcross'] = crossSMA(tickerData)
        
        
        
        
    
        # if the last item in SMAcross is 1 (buy signal) 
        # and if the cross happened less than 10 days ago: goes to 'good'
        
        try: 
            if  pd.to_numeric(tickerData.SMAcross[pd.DataFrame.last_valid_index(tickerData.SMAcross)])==1:
                crossDate = date(pd.DataFrame.last_valid_index(tickerData.SMAcross).year,
                                          pd.DataFrame.last_valid_index(tickerData.SMAcross).month,
                                          pd.DataFrame.last_valid_index(tickerData.SMAcross).day)
                if (today - crossDate) < timedelta(10):
                    
                        good.append(tickerName)
                        print(f'Recently crossed up: {tickerName}')
                        
                else:
                    normal.append(tickerName)
                    print(f'Crossed up some time ago: {tickerName}')
                        
                        
                
            # If it haven't crossed yet but the SMA35 is just below SMA100
            # if the diff is less than 5% : goes to good
            elif pd.to_numeric(tickerData.SMAcross[pd.DataFrame.last_valid_index(tickerData.SMAcross)])==0:
                if 0.9 < SMA100.Close[-1] / SMA35.Close[-1] <  1.0:
                    good.append(tickerName)
                    #normal.append(tickerName)
                    row_364.append(tickerName)
                    print(f'Might cross up: {tickerName}')
                else:
                    continue
         
            else:
                continue
          
            
        except:
            continue
        
        
    
    # =============================================================================
    # Finding peaks with scipy.signal.find_peaks; multiple useful arguments can be specified --> should experiment with it later
    # it returns an array plus a dictionary: the underscore means that we don't care about the second (it would be 'properties' dict)
    # findPeaks is an array with the index/row position of the peaks (so not the value just the position)
    # then we create an empty numpy array which we fill with NaNs
    # then we change the findPeaks positions to the values of Close at findPeaks
    # now we have a locational maxima array: 'localMax' which is the same lenght as tickerData and can be added to that dataset or plotted together
    # it might be renamed later to localMax_daily or_weekly or localMax_strong (e.g. a vector where maxima have high volumes)
    #  
    # =============================================================================
        
        #if tickerName in good or tickerName in normal:
        
        if tickerName in good:
             
            from scipy.signal import find_peaks
            findPeaks, _ = find_peaks(tickerData.Close, distance=4, threshold=1)
            
        
    # =============================================================================
    #         Adding the absolute max
    #         I am sure we can use find_peaks to not filter out the absolute maximum but we can add it back for now
    # =============================================================================
         
    # =============================================================================
    #         getting the index is tricky in pandas, had to 
    #         create tickerData.myIndex column just because tickerData.index is a date
    # =============================================================================
        
            
            
             
            
            #Minima
            findTroughs, _ = find_peaks(tickerData.Close*(-1), distance=4, threshold=1)
            
            
            
            
            
            peaks_and_troughs = findPeaks.tolist() + findTroughs.tolist()
                        
            #plus absolute max with np.argmax
            peaks_and_troughs.append(np.argmax(tickerData.Close))
            peaks_and_troughs.sort()
            peaks_and_troughs            
            
            
            techZone = {}
    
            for i in range(len(tickerData.Close[peaks_and_troughs])):
                techZone[i] = []
                for k in range(i+1,len(tickerData.Close[peaks_and_troughs])):
                    if abs(tickerData.Close[peaks_and_troughs][i] - tickerData.Close[peaks_and_troughs][k])  \
                           < tickerData.Close[peaks_and_troughs][i] * 0.005:
                                
                                techZone[i].append(tickerData.Close[peaks_and_troughs][i])
                                techZone[i].append(tickerData.Close[peaks_and_troughs][k])
                               
                                                            
            emptyZone = []
            
            for i in range(len(techZone)):
                if not techZone[i]:
                    emptyZone.append(i)
                    
            for k in emptyZone:
                techZone.pop(k, None)
            
            
            
            
            
            #Volume
            #maxVolume is the index and tickerData.Volume[maxVolume] is the actual volume
            maxVolume = np.argpartition(tickerData.Volume.values,-10)[-10:]
            
            #findVolume is peak detection among volumes if the height is more than average + 1.5 stdev    
            findVolume, _ = find_peaks(tickerData.Volume, height = tickerData.Volume.mean() + statistics.stdev(tickerData.Volume)*1.5)
                
            
            #highVolume is combining max and find_peaks values 
            #gets rid of duplicates by converting it to a set and then back to list
            highVolume = set(maxVolume.tolist() + findVolume.tolist())
            highVolume = sorted(list(highVolume))    
            
            
    # =============================================================================
    #                 
    #                 
    #             #Support    
    #                 
    #             supportZone = {}
    #     
    #             for i in range(len(tickerData.Close[findTroughs])):
    #                 supportZone[i] = []
    #                 for k in range(i+1,len(tickerData.Close[findTroughs])):
    #                     if abs(tickerData.Close[findTroughs][i] - tickerData.Close[findTroughs][k])  \
    #                            < tickerData.Close[findTroughs][i] * 0.01:
    #                                 
    #                                 supportZone[i].append(tickerData.Close[findTroughs][i])
    #                                 supportZone[i].append(tickerData.Close[findTroughs][k])
    #                                
    #                                                             
    #             emptyZone = []
    #             
    #             for i in range(len(supportZone)):
    #                 if not supportZone[i]:
    #                     emptyZone.append(i)
    #                     
    #             for k in emptyZone:
    #                 supportZone.pop(k, None)
    # =============================================================================
            
        
        
        
        
        
        
        
        #Plot 
        
            
            
            fig = plt.figure(figsize=(15.5,6))
            plt.plot(tickerData.Close)
            plt.title(tickerName)
            
            plt.plot(SMA35.Close)
            plt.plot(SMA100.Close)
            plt.plot(SMA200.Close)
            plt.scatter(tickerData.index[maxVolume], tickerData.Close[maxVolume], marker = '*' , color='black')
            plt.scatter(tickerData.index[findVolume], tickerData.Close[findVolume], marker = 'o', color = 'black')
            
            
            
            
                        
            
            
            
            if any(techZone) == False:
                fig.suptitle("No technical zones")
                
            
               
            else:
                 
                
                #coloring techzone
                
                #creating strongzone if multiple peaks_and_troughs (myAlpha is above 0.1) 
                #or if highVolume is in a techZone
      
                strongZone = [] 
                
                
                for i in techZone:
                    
                           
                    
                        myAlpha = min((len(techZone[i]))**2*0.01 , 1)
                        
                        
                        for k in range(len(highVolume)):
                            if sorted(techZone[i])[0]*0.985 <= tickerData.Close[highVolume][k] <= sorted(techZone[i])[-1]*1.015:
                                myColor = "red"
                            else:
                                myColor = "green"
                        
                        
                        
                        if myAlpha > 0.1:
                            strongZone.append(techZone[i])
                        elif myColor == "red":
                            if not techZone[i] in strongZone:
                                strongZone.append(techZone[i])
                            
                            
                        plt.fill_between(tickerData.index,
                                     sorted(techZone[i])[0], sorted(techZone[i])[-1],
                                     color = myColor, alpha = myAlpha)
                   
                    
                    
                    
                    
            goodData[tickerName] = {}
            goodData[tickerName]['Close'] = tickerData.Close
            goodData[tickerName]['Volume'] = tickerData.Volume
            goodData[tickerName]['SMA200'] = SMA200.Close    
            goodData[tickerName]['SMA100'] = SMA100.Close
            goodData[tickerName]['SMA35'] = SMA35.Close
            goodData[tickerName]['SMAcross'] = tickerData.SMAcross     
                    
                    
    # =============================================================================
    #                 #last techzone to be a bit darker    
    #                 myAlpha = 0.4
    #                 plt.fill_between(tickerData.index,
    #                                      sorted(techZone[list(techZone)[-1]])[0], sorted(techZone[list(techZone)[-1]])[-1],
    #                                      color="red",alpha=myAlpha)
    # =============================================================================
                
                
    
    
    
            try:
                significantLevels[tickerName] = strongZone
            except:
                significantLevels[tickerName] = 'No technical zones'
                
                
            plt.close()
            myPlots[tickerName] = fig
            
            
            
            for j in range(len(significantLevels[tickerName])):
                
                try:
                
                    significantLevels[tickerName][j] = [
                            
                            min(significantLevels[tickerName][j]),
                            max(significantLevels[tickerName][j])
                            ]
                except:
                    continue
    
            
            
            #Great
            try:
                for x in range(len(significantLevels[tickerName])):
                    if max(significantLevels[tickerName][x]) < tickerData.Close[-1] < min(significantLevels[tickerName][x+1]):
                        upside = min(significantLevels[tickerName][x+1]) - tickerData.Close[-1]
                        downside = tickerData.Close[-1] - max(significantLevels[tickerName][x])
                        if upside > downside:
                            if tickerName not in great:
                                great.append(tickerName)
                                greatData[tickerName] = {}
                                greatData[tickerName] = {}
                                greatData[tickerName]['Close'] = tickerData.Close
                                greatData[tickerName]['Volume'] = tickerData.Volume
                                greatData[tickerName]['SMA200'] = SMA200.Close    
                                greatData[tickerName]['SMA100'] = SMA100.Close
                                greatData[tickerName]['SMA35'] = SMA35.Close
                                greatData[tickerName]['SMAcross'] = tickerData.SMAcross
                                print(f'We have a great 1st time: {tickerName}!')
            except:
                print(f'{tickerName} could not go to great 1st')
                pass
            
            
            
            
            
            #Great second try
            try:
                for x in range(len(significantLevels[tickerName])):
                    if max(significantLevels[tickerName][x]) < tickerData.Close[-1]:
                        upside = max(tickerData.Close) - tickerData.Close[-1]
                        downside = tickerData.Close[-1] - max(significantLevels[tickerName][x])
                        if upside > downside:
                            if tickerName not in great:
                                great.append(tickerName)
                                greatData[tickerName] = {}
                                greatData[tickerName]['Close'] = tickerData.Close
                                greatData[tickerName]['Volume'] = tickerData.Volume
                                greatData[tickerName]['SMA200'] = SMA200.Close    
                                greatData[tickerName]['SMA100'] = SMA100.Close
                                greatData[tickerName]['SMA35'] = SMA35.Close
                                greatData[tickerName]['SMAcross'] = tickerData.SMAcross
                                print(f'We have a great 2nd time: {tickerName}!')
                        
            except:
                print(f'{tickerName} could not go to great 2nd')
                pass          
                        
            


            #Great: in good and RSI is below 30
            priceMoves14day = []
            try:
                for i in range(15):
                    priceMoves14day.append(tickerData.Close[-(i+1)] / tickerData.Close[-(i+2)])
                
                avgUp = sum([priceMoves14day[x]-1 for x in range(len(priceMoves14day)) if priceMoves14day[x] > 1]) / len(priceMoves14day)
                avgDown = sum([priceMoves14day[x]-1 for x in range(len(priceMoves14day)) if priceMoves14day[x] < 1]) / len(priceMoves14day)    
                myRSI = 100 - (100/(1+avgUp/abs(avgDown)))
                
                if myRSI < 30:
                    if tickerName not in great:
                                great.append(tickerName)
                                greatData[tickerName] = {}
                                greatData[tickerName]['Close'] = tickerData.Close
                                greatData[tickerName]['Volume'] = tickerData.Volume
                                greatData[tickerName]['SMA200'] = SMA200.Close    
                                greatData[tickerName]['SMA100'] = SMA100.Close
                                greatData[tickerName]['SMA35'] = SMA35.Close
                                greatData[tickerName]['SMAcross'] = tickerData.SMAcross
                                print(f'We have a great 3rd time (RSI): {tickerName}!')
                                greatRSI.append(tickerName, today)


            except:
                pass
            
        #Recommendations
        #Too much time
    # =============================================================================
    #     recommendations = yf.Ticker(tickerName).recommendations
    #     print(f'Got it for {tickerName}')
    # =============================================================================
        
        
        
        
        
        
        
        #Stock information
        #Too much time
    # =============================================================================
    #     try:
    #         stockInfo = yf.Ticker(tickerName).info
    #         print(f'Got the info: {tickerName}')
    #         
    #     except:
    #         continue
    # 
    #     
    #     with fuckit:
    #         
    #         betaCAPM = stockInfo.get('beta')
    #         bookValue = stockInfo.get('bookValue')
    #         marketCap = stockInfo.get('marketCap')
    #         _52High   = stockInfo.get('fiftyTwoWeekHigh')
    #         _52Low = stockInfo.get('fiftyTwoWeekLow')
    #         sharesShort = stockInfo.get('sharesShort')
    #         sharesOutstanding = stockInfo.get('sharesOutstanding')
    #         trailingEps = stockInfo.get('trailingEps')
    #         trailingPE = stockInfo.get('trailingPE')
    #         priceToBook = stockInfo.get('priceToBook')
    #         lastDividendValue = stockInfo.get('lastDividendValue')
    #         lastDividendDate = datetime.date(datetime.fromtimestamp(stockInfo.get('lastDividendDate')))
    #         averageVolume = stockInfo.get('averageVolume')
    #         averageVolume10days = stockInfo.get('averageVolume10days')
    #         
    #         shortRatio = (sharesShort / sharesOutstanding)
    # 
    # 
    #         daysToCover = (sharesShort / max(averageVolume, averageVolume10days))
    #         print(daysToCover)
    # =============================================================================
                
            
            
    #print(f'Tickers that passed so far {len(tickerList1.Symbol  )}, these could go to "good" or "great"') 
    
    print(f'Awful : {len(awful)}')
    print(f'Bad : {len(bad)}')
    print(f'Normal : {len(normal)}')
    print(f'Good : {good} on {today}')  
    print(f'Great : {great} on {today}')  
    end_time = round(time.time()-start_time,2)
    print(f'Time: {end_time}')
    #This ends the 'Trading' script which selected the 'good' for the given period
    
    
    
    
    
    
    #good portfolio
    #Second part of the 'Portfolio' script
    
    portfolioValue = []
    dailyValue = [] 
    testData = {}
    myWeights = {}
    
    
    
    
    if len(good) == 0:
        print(f'There is no good stock on {today}')
    
    else:                
        
        for k in range(len(good)):
            
            try:    
                testData[good[k]] = importer.DataReader(good[k], "yahoo", testDateStart, testDateEnd)
              
            except:
                continue
            
            
            
        for i in range(len(good)):    
            if good[i] in testData:        
                myWeights[good[i]] = (1/len(testData))/goodData[good[i]]['Close'][-1]
         
        for x in range(testDays):       
            dailyValue = []
            for j in range(len(good)): 
                if good[j] in testData:
                    try:
                        dailyValue.append(testData[good[j]]['Close'][x] * myWeights[good[j]])
                    except:
                        continue
            portfolioValue.append(sum(dailyValue))
                    
                
                
        
        goodProfit.append(portfolioValue[[i for i,j in enumerate(portfolioValue) if j != 0][-1]])
        












    #Great = myProfit
    #Second part of the 'Portfolio' script
    
    portfolioValue = []
    dailyValue = [] 
    testData = {}
    myWeights = {}
    
    
    
    
    if len(great) == 0:
        print(f'There is no great stock on {today}')
    
    else:                
        
        for k in range(len(great)):
            
            try:    
                testData[great[k]] = importer.DataReader(great[k], "yahoo", testDateStart, testDateEnd)
              
            except:
                continue
            
            
            
        for i in range(len(great)):    
            if great[i] in testData:        
                myWeights[great[i]] = (1/len(testData))/greatData[great[i]]['Close'][-1]
         
        for x in range(testDays):       
            dailyValue = []
            for j in range(len(great)): 
                if great[j] in testData:
                    try:
                        dailyValue.append(testData[great[j]]['Close'][x] * myWeights[great[j]])
                    except:
                        continue
            portfolioValue.append(sum(dailyValue))
                    
                
                
        
            
        
        myProfit.append(portfolioValue[[i for i,j in enumerate(portfolioValue) if j != 0][-1]])    
        
        
        #Plot
        portfolioPlot = plt.figure(figsize=(15.5,6)) 
        plt.scatter(range(testDays), portfolioValue)
        plt.title(f'Tickers: {great}, today= {today}' )    
        
        
        
        print(f'Testing from {testDateStart} to {testDateEnd} for great: {great}')
        
    
    
if len(myProfit) > 0:
    print(f' The average "great" profit after 60 days is {sum(myProfit) / len(myProfit)}')
    print(f' The S&P500 had a return of {sum(spyProfit) / len(spyProfit)} during these intervals')
    print(f' The average "good" profit over these 60 day intervals is {sum(goodProfit) / len(goodProfit)}' )
    
elif len(goodProfit) > 0:
    print(f' The S&P500 had a return of {sum(spyProfit) / len(spyProfit)} during these intervals')
    print(f' The average "good" profit over these 60 day intervals is {sum(goodProfit) / len(goodProfit)}' )
else:
    print(f'There might not be great, data for the testing period or something went wrong with myProfit')




