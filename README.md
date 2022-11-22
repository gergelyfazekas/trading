# trading
# algorithmic portfolio optimization and stock price prediction


**Goal:** 

This project is seeking to create an automated stock trading system which is capable of managing a portfolio without human supervision.
For this purpose some decision rules are needed to guide the program whether to buy, sell or do nothing with regards to a particular stock and time. Rather then a theory based portfolio optimization I choose to go with an "empirical decision rule", namely using a genetic algorithm to search for a "good enough" decision rule based on some input variables. These input variables are in line with the conventional risk-return approach, such that a decision whether to transact a certain stock is based on a forecast of its future performace, its estimated variance-covariance structure with the current portfolio and some other factors such as the level of diversification of the current portfolio. This genetic search approach does not impose restrictions on the decision, if the forecast tends to be accurate then that will receive a bigger weight, if the variance estimate is more robust then that will guide the decision and if neither is accurate the level of diversification or general market trends migth dominate the decision.

The project uses two main models. (1) The decision making (portfolio managing) genetic algorithm for which a NEAT (Neuroevolution of augmenting topologies) algorithm is used. (2) The forecasting model provides the decision making model with an expected return of the stock, for this model several techniques are considered (both linear and non-linear). The forecast model's inputs are 'direct' (such as lags of stock return, volume, seasonal dummies) and "derived" (these can be forecasts themselves from simpler Autoregressive models or smoothing methods or other indicators  such as distance from resistance/support levels or analyst's consensus, expected monetary policy environment etc.).

The data comes from the yahoo finance api through the pandas datareader package. It contains price and volume information of 400 stocks over 10 years,
all listed on the NASDAQ. The price frequency is daily, so that the total number of available data points is around 1 million. Since yahoo finance queries are limited and generally quite slow, an sql database is created which is stored on amazon's AWS RDS cloud storage to ease access to the data for quick calculations.


**Code:**

All .py files only contain functions and classes and cannot be run on their own. Since most of the analysis is done on a one-off basis 
(e.g. there is no need to calculate the historical returns of a stock multiple times) these functions are imported to jupyter notebooks to create adhoc pipelines.

- portfolio_class.py and stock_class.py establish classes for Stock and Portfolio objects. Methods of a Stock instance are used to calculate time series properties and technical indicators. Methods of a Portfolio instance are used to derive portfolio related attributes such as keeping track of the current portfolio, cash/invested balance, rate of diversification etc. 
- database.py is a collection of yahoo finance related querying functions and sql oriented functions, this script is used together with stock_class.py
- brain.py is the decision making part of the code where the NEAT algorithm is implemented (set up for model training purposes).
- forecast.py is comprised of several regression type models such as regression trees, random forests and boosting models
- tuning.py is concerned with some auxiliary funtions mainly used for tuning some parameters


**Problems and implemented solutions:**

1) Data:
- querying yahoo finance is very slow and since it is an I/O type bound this problem was mitigated with concurrent querying, so that one query is issued and before getting back the result from yahoo's server another (or rather several other) price queries are sent as well. This way the price query process was shortened about 10-fold. 
- Storing the data turned out to be difficult since a 100 columns * 1 M rows is a bit big for python pickeling so the whole dataset was migrated to sql.
This is done through sql_alchemy using mysql. One table was enough logically to store all the information and this table was migrated to amazon AWS RDS based on storage space considerations on my own Mac.
- pulling/pushing data from sql is quite easy to mess up and lot of time was spent on error handling and creating backup tables, finally functions using
connections between pandas dataframes and sql tables turned out to be easy to use and solved most of the previous problems.

2) Forecast model
- The forecast model's inputs are themselves forecasts (e.g. simple moving average or exponential smoothing). The idea of combining forecasts comes from 
the fact that the Variance of the average of identically distributed random variables is 1/N * sigma^2 if they are uncorrelated and 
rho*sigma^2 + (1-rho)/N * sigma^2 if they are correlated with corr coef being rho. This is the idea behind consensus forecasts on the stock market and also the idea behind random forests and other ensemble methods in machine learning. So if I can generate forecasts that are based on valid methods while being sufficiently "uncorrelated" then combining them will lower the overall variance of the forecast. Multiple models will be assesed here and we will see how much nonlinearity is needed from the forecast model and also if high correlation between the inputs will prompt the use of a ridge/lasso type shrinkage method. 

3) NEAT algorithm 
- the NEAT algorithm is implemented so that after a decision the fitness of a particular genome is assigned or updated. However since the success and thus
the fitness of a portfolio manging algorithm is based on successive trades over a long time horizon the whole trading cycle (for date in dates: for stock
in stocks: decide buy/sell, decide amount) had to be moved inside the eval_genomes function which is the NEAT package's main genome evaluation scheme.
This brough about the problem that the eval_genome function can only have 2 arguments while having the trading decisions within this frame required a lot
more variables to be controlled by the user (which input variables to use, time windows, plotting options, list of stocks to consider, verbosity ...). 
This problem was mitigated by moving the eval_genomes function inside the run_neat function so that the eval_genomes has access and visibility of the 
arguments of its parent run_neat which is not constrained by the number of args and kwargs it can take. 
- since neat is a neural net it takes inputs and produces one or more outputs. The inputs can be the current stock's expected return (forecast) and its
expected variance-covariance matrix etc. but the net's decision is made only based on these inputs. This means that at the time when the decision is made
it only sees the stock it is given and not the stocks that are potential alternatives for the same date. This problem is circumvented by creating a
ranking of top_10, top_20 ... stocks based on their forecast. So if a stock's top_10 dummy is 1 it means that based on the forecasts of all stocks for 
that date, this stock is in the top 10. This way the neural net has some information that relates every single stock to the others at the time of the 
decision (if it's not in the top 10 others will be coming up that have better potential so the decision is to do nothing with this stock). 
- the NEAT algorithm decides two things at once. The direction whether to buy or sell and the amount. This could be achieved several ways and I chose to 
have 1 sigmoid output node which is interpreted as follows. Output = 0.5: do nothing, output = 1 buy the stock with 50% of our current cash available, 
output = 0: sell 50% of the stock if we have it (or possibly short it but that is not implemented yet). So an output of 0.6 would trigger a buy signal
with 10% of the cash we have at that moment. This formulation places some constraint on the portfolio since we cannot buy or sell anything above 50%. 
If this turns out to be a limiting factor, the neural net's output can be changed to contain 2 nodes where one would decide the direction and the other
the amount. 
- The fitness function of the NEAT algorithm is determined by the total return the portfolio generated during the training period of around 8 years. This
does not take into account fluctuations of the total return (variance of the portfolio). A psychologically more ensuring way would be to reward genomes
that manage the portfolio such that mainly goes upwards at any moment even if that does not lead to a maximal return at the end. This would mean punishing
the genome every day if the portfolio's value decreased compared to the previous day. This can be implemented later with minimal effort. 

4) Parameter tuning
- one input variable that would be fed to the forecast model is the current price's distance to resistance/support levels. To find the levels automatically I used a peak finding algorithm from the package scipy. Since people identify technical levels by looking at the graph of the stock price the algorithm  needs to do the same. This presents the problem that the price of AAPL is not the same magnitude than that of MSFT. This means that to find the best hyperparamters of the peak finding algorithm we need to standardize the prices. Since we only search for resistance/support levels close to today's price we can just standardize the series by dividing with today's price. If the series has a dominant upward trend we will get small values (0.1, 0.001 ...) and the peaks of those periods won't be visible any more on a chart. However we do not care about these values since they are far away from today's price so they will not act as resistance/support in the near future anyway. Even after standardizing the series the peak finding algortihm needs a good set of hyperparameters to work with. Since drawing technical levels is not a supervised learning problem with X-y pairswe cannot find the hyperparameters by optimizing towards some true value. The way I chose to select the hyperparams is to visually inspect the algorithm on multiple stock charts and decide which combination seems to find the good levels. The algorithm calls something a technical level if the price took turns multiple times (had multiple peaks) within a close range to that price level. The tune_tech_levels function if tuning.py was used to plot multiple sets of hyperparameters and the resulting technical levels.









