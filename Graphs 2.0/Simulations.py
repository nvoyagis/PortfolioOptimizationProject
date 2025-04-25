import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import cvxpy
import sklearn
import fast_tmfg
#import riskfolio as rf



# Run simulations of optimized triad portfolios and find the ratio of successful/unsuccessful stocks within that triad
def run_simulations(sims: int, seed: int, stocks: list[str], start1: str, end1: str, start2: str, end2: str):
    portfolio_returns = []
    data = []
    np.random.seed(seed)
    beat_spy_count = 0
    for i in range(sims):
        daily_stock_percent_changes = pd.DataFrame()
        overall_stock_percent_changes = []

        # Generate random buy/sell dates using a random stock (doesn't matter which one)
        df = pd.read_csv(
                f'Data2015-2025/HistoricalPrices 2015 - 2025, {stocks[0]}.csv',
                parse_dates=['Date'],
                date_format="%m/%d/%y"
            )
        df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%y", errors='coerce')

        # Set Date as the index
        # df.set_index('Date', inplace=True)

        # Define your date range
        start1 = pd.Timestamp(start1)
        end1 = pd.Timestamp(end1)

        # Filter rows between start1 and end1
        filtered_df1 = df[df['Date'] > start1]
        filtered_df1.index = pd.to_datetime(filtered_df1.index)
        filtered_df1 = filtered_df1[filtered_df1.index < end1]


        # Get all valid dates in that range
        filtered_df1.set_index('Date', inplace=True)
        valid_dates1 = filtered_df1.index

        # Pick a random date
        random_buy_date = np.random.choice(valid_dates1)

        # Pick a random date to sell the portfolio 
        start2 = pd.Timestamp(start2)
        end2 = pd.Timestamp(end2)
        filtered_df2 = df[df['Date'] > start2]
        filtered_df2.index = pd.to_datetime(filtered_df2.index)
        filtered_df2 = filtered_df2[filtered_df2.index < end2]

        filtered_df2.set_index('Date', inplace=True)
        valid_dates2 = filtered_df2.index

        random_sell_date = np.random.choice(valid_dates2)
        
        # Calculate percent change of SPX (to use as a benchmark)
        spx = pd.read_csv(
                f'Data2015-2025/HistoricalPrices 2015 - 2025, SPX.csv',
                parse_dates=['Date'],
                date_format="%m/%d/%y"
            )
        spx.set_index('Date', inplace=True)
        spx_open_value = spx.loc[random_buy_date, " Open"]
        spx_close_value = spx.loc[random_sell_date, " Close"]
        spx_percent_change = (spx_close_value - spx_open_value) / spx_open_value * 100

        for s in stocks:
            df = pd.read_csv(
                f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
                parse_dates=['Date'],
                date_format="%m/%d/%y"
            )

            # Set Date as an index
            df.set_index('Date', inplace=True)

            # Get the opening and closing values for specific dates
            open_value = df.loc[random_buy_date, " Open"]
            close_value = df.loc[random_sell_date, " Close"]
            overall_stock_percent_changes.append((close_value - open_value)/open_value * 100) 

            # Change Date back into a column so it can be accessed as normal
            df.reset_index(inplace=True)
            # Remove spaces in column names.
            df.columns = df.columns.str.strip()
            # Remove some recent data to analyze historical data
            df = df[df['Date'] <= random_buy_date]
            # Calculate daily percent change for a stock and add it as a new column to df.
            df['Percent Change'] = ((df['Close'] - df['Open']) / df['Open']) * 100
            # Store percent changes into a single DataFrame, with each column representing a different stock.
            daily_stock_percent_changes[s] = df['Percent Change']

    

        # Make covariance matrix of a 3-stock portoflio
        cov_mat = daily_stock_percent_changes.cov()


        # Calculate variance-minimizing stock allocations
        mat = np.array(2 * cov_mat)
        mat = np.vstack([mat, np.ones((1, mat.shape[1]))])
        mat = np.hstack([mat, np.ones((mat.shape[0], 1))])
        mat[-1][-1] = 0
        mat_inv = np.linalg.inv(mat)
        B = np.zeros((mat.shape[1], 1))
        B[-1] = 1
        stock_allocations = np.dot(mat_inv, B)
        # Remove last row of stock_allocations (which represents the lagrange multiplier)
        stock_allocations = np.delete(stock_allocations, -1, axis=0)

        # Calculate variance of portfolio
        x = np.dot(np.transpose(stock_allocations), cov_mat)
        variance = np.dot(x, stock_allocations)

        portfolio_percent_change = 0
        for i in range(len(stock_allocations)):
            portfolio_percent_change += overall_stock_percent_changes[i] * stock_allocations[i][0]
        
        portfolio_returns.append(portfolio_percent_change)
        if portfolio_percent_change > spx_percent_change:
            beat_spy_count += 1

        data.append([stock_allocations, '% Change: ' + str(portfolio_percent_change), 'Variance: ' + str(variance)])



    avg_p_change = 0
    for r in portfolio_returns:
        avg_p_change += r
    avg_p_change = avg_p_change/len(portfolio_returns)
    return (avg_p_change, beat_spy_count)
    #TODO: add average spy change


    # Store simulations in a file
    # with open(filename, "w", newline="") as f:
    #     f.write('Portfolio minimum variance: ' + str(variance[0][0]))
    #     f.write('Portfolio allocations: ' + str(stocks[0]) + ': ' + str(stock_allocations[0][0]) + '\n                       ' + str(stocks[1]) + ': ' + str(stock_allocations[1][0]) + '\n                       ' + str(stocks[2]) + ': ' + str(stock_allocations[2][0]))
    #     f.write('Portfolio percent change: ' + str(portfolio_percent_change))
    #     f.write('----------------------------------------------------------------------------------------------------------')
    #     f.write('--|-- Average portfolio percent change: ' +  + '--|--')
    #     f.write('--|-- Median portfolio percent change: ' +  + '--|--')





# Create TMFG for each simualtion
def simulation2(cov_mat: str, sims: int, seed: int, stocks: list[str], start1: str, end1: str, start2: str, end2: str):

    # Create complete graph w/ covariance 
    cov_mat 


    # Create TMFG

    # Create dual graph

    # Run variance-minimized simulations with random date ranges

    # Store data as a new file


    model = fast_tmfg.TMFG()
    cliques, seps, adj_matrix = model.fit_transform(weights=corr, cov=cov, output='logo')

