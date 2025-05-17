import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import cvxpy
import sklearn
import fast_tmfg
import Graph_Theory_Functions
import Dual
import os
import time
from itertools import combinations
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

        # Generate random buy/sell dates using a random stock (doesn't matter which one since all stocks use the same dates)
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
            df = df[df['Date'] < random_buy_date]
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





# Create TMFG for each simulation
def simulation2(sims: int, seed: int, stocks: list[str], start1: str, end1: str, start2: str, end2: str):
    for i in range(sims): 
        # Initialize data
        stocks.sort()
        stock_dict = {}
        counter = 0
        for s in stocks:
            stock_dict[counter] = s
            counter += 1

        # Generate random buy/sell dates using a random stock (doesn't matter which one since all stocks use the same dates)
        df = pd.read_csv(
                f'Data2015-2025/HistoricalPrices 2015 - 2025, {stocks[0]}.csv',
                parse_dates=['Date'],
                date_format="%m/%d/%y"
            )
        df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%y", errors='coerce')

        # Define date range (buy)
        start1 = pd.Timestamp(start1)
        end1 = pd.Timestamp(end1)

        # Filter rows between start1 and end1 (buy)
        filtered_df1 = df[df['Date'] > start1]
        filtered_df1.index = pd.to_datetime(filtered_df1.index)
        filtered_df1 = filtered_df1[filtered_df1.index < end1]

        # Get all valid dates in that range (buy)
        filtered_df1.set_index('Date', inplace=True)
        valid_dates1 = filtered_df1.index

        # Pick a random date (buy)
        random_buy_date = np.random.choice(valid_dates1)

        # Define date range (sell)
        start2 = pd.Timestamp(start2)
        end2 = pd.Timestamp(end2)

        # Filter rows between start2 and end2 (sell)
        filtered_df2 = df[df['Date'] > start2]
        filtered_df2.index = pd.to_datetime(filtered_df2.index)
        filtered_df2 = filtered_df2[filtered_df2.index < end2]

        # Get all valid dates in that range (sell)
        filtered_df2.set_index('Date', inplace=True)
        valid_dates2 = filtered_df2.index

        # Pick a random date (sell)
        random_sell_date = np.random.choice(valid_dates2)


        # TODO: Re-make graph from email. Probably use quartiles to make the graph.


        # Create complete graph w/ covariance with a random date between start1 and end1
        cov_mat = Graph_Theory_Functions.get_weight_mat(stocks, random_buy_date) # Originally 2019-01-01
        cov_mat = pd.DataFrame(cov_mat)
        print("Covariance Matrix:")
        print(cov_mat)

        # Create TMFG
        model = fast_tmfg.TMFG()
        w = pd.DataFrame(np.ones((len(stocks), len(stocks))))
        cliques, seps, adj_matrix = model.fit_transform(weights=cov_mat, output='weighted_sparse_W_matrix') # This only works with Pandas dataframes for whatever reason
        g = nx.from_numpy_array(adj_matrix)
        print('Adjacency Matrix:')
        print(adj_matrix)

        # Draw the graph
        g = nx.relabel_nodes(g, stock_dict)
        pos0 = nx.planar_layout(g)
        nx.draw(g, pos=pos0, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        plt.title("TMFG Graph")
        plt.show()



        # Make dual graph of TMFG
        triads = model.triangles
        print('Triads:')
        print(triads)
        dual = Dual.build_dual_from_faces(triads)

        face_list = {}
        

        for node in dual.nodes:
            new_node = tuple(stock_dict[i] for i in node)
            face_list[node] = new_node
        print(face_list)

        # Relabel nodes properly
        dual = nx.relabel_nodes(dual, face_list)

        # Fix node labels
        labels = {
            node: f"({', '.join(node)})"
            for node in dual.nodes
        }

        # Draw graph
        plt.figure(figsize=(8, 6))
        pos = nx.kamada_kawai_layout(dual)
        nx.draw_networkx_edges(dual, pos, width=2)
        nx.draw(
            dual, pos,
            with_labels=True,
            labels=labels,
            node_color="lightcoral",
            edge_color="black",
            node_size=1000,
            font_size=10
        )
        plt.title("Dual Graph")
        plt.show()
        
        
        # Run variance-minimized simulations with random date ranges
        # Minimize risk in triads
        portfolio_returns = []
        for node in dual.nodes:
            percent_changes = pd.DataFrame()
            random_period_percent_changes = []
            for s in node:
                df = pd.read_csv(
                f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
                parse_dates=['Date'],
                date_format="%m/%d/%y")

                # Set Date as index
                df.set_index('Date', inplace=True)

                # Use pd.Timestamp for the date lookup
                target_date1 = pd.Timestamp(random_buy_date)
                target_date2 = pd.Timestamp(random_sell_date)

                # Get the opening value for a specific date
                open_value = df.loc[target_date1, " Open"]
                close_value = df.loc[target_date2, " Close"]
                random_period_percent_changes.append((close_value - open_value)/open_value * 100)

                # Change Date back into a column so it can be accessed as normal
                df.reset_index(inplace=True)
                # Remove spaces in column names.
                df.columns = df.columns.str.strip()
                # Remove some recent data to analyze profits in the past
                cutoff_date = pd.to_datetime(random_buy_date)
                df = df[df['Date'] < cutoff_date]
                # Calculate daily percent change for a stock and add it as a new column to df.
                df['Percent Change'] = ((df['Close'] - df['Open']) / df['Open']) * 100
                # Store percent changes into a single DataFrame, with each column representing a different stock.
                percent_changes[s] = df['Percent Change']
            
            # Make covariance matrix of a 3-stock portoflio
            cov_mat = percent_changes.cov()
            # Solve for optimal stock allocations
            mat = np.array(2*cov_mat)
            mat = np.vstack([mat, np.ones((1, mat.shape[1]))])
            mat = np.hstack([mat, np.ones((mat.shape[0], 1))])
            mat[-1][-1] = 0
            mat_inv = np.linalg.inv(mat)
            B = np.zeros((mat.shape[1], 1))
            B[-1] = 1
            stock_allocations = np.dot(mat_inv, B)
            # Remove last row of stock_allocations
            stock_allocations = np.delete(stock_allocations, -1, axis=0)
            x = np.dot(np.transpose(stock_allocations), cov_mat)
            variance = np.dot(x, stock_allocations)

            print('Portfolio minimum variance: ' + str(variance[0][0]))
            print('Portfolio allocations: ' + str(node[0]) + ': ' + str(stock_allocations[0][0]) + '\n                       ' + str(node[1]) + ': ' + str(stock_allocations[1][0]) + '\n                       ' + str(node[2]) + ': ' + str(stock_allocations[2][0]))
            

            random_period_portfolio_percent_change = 0
            for i in range(len(random_period_percent_changes)):
                random_period_portfolio_percent_change += random_period_percent_changes[i] * stock_allocations[i][0]
            
            print(f'Portfolio percent change from {random_buy_date} to {random_sell_date}: ' + str(random_period_portfolio_percent_change))
            print('--------------------------------------------')

            portfolio_returns.append(random_period_portfolio_percent_change)


        # TODO: Store data in a new file





# Create TMFG for each simulation with a chosen buy date
def simulation3(sims: int, seed: int, stocks: list[str], buy_date: str, sell1: str, sell2: str):
    start_time = time.time()
    # Store data as a new file
    with open(os.path.join('Simulations', f'{sims}_Sims2_{buy_date}_and_{sell1}_to_{sell2}.txt'), "w", newline="") as f:
        f.write(f'Stocks ({len(stocks)}): {stocks}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Simulations: {sims}\n')
        f.write(f'Starting date: {buy_date}\n')
        f.write(f'Ending date range: {sell1} to {sell2}\n')
        f.write('------------------------------------------------------\n')

        # Initialize data
        stocks.sort()
        stock_dict = {}
        counter = 0
        for s in stocks:
            stock_dict[counter] = s
            counter += 1

        # Create complete graph w/ covariance using the selected start date
        cov_mat = Graph_Theory_Functions.get_weight_mat(stocks, buy_date)
        cov_mat = pd.DataFrame(cov_mat)

        # Create TMFG
        model = fast_tmfg.TMFG()
        w = pd.DataFrame(np.ones((len(stocks), len(stocks))))
        cliques, seps, adj_matrix = model.fit_transform(weights=cov_mat, output='weighted_sparse_W_matrix') # This only works with Pandas dataframes for whatever reason
        g = nx.from_numpy_array(adj_matrix)


        # Draw the graph
        # g = nx.relabel_nodes(g, stock_dict)
        # pos0 = nx.planar_layout(g)
        # nx.draw(g, pos=pos0, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        # plt.title("TMFG Graph")
        # plt.show()

        # Make dual graph of TMFG
        triads = model.triangles
        dual = Dual.build_dual_from_faces(triads)

        face_list = {}
        for node in dual.nodes:
            new_node = tuple(stock_dict[i] for i in node)
            face_list[node] = new_node
        print(face_list)

        # Relabel nodes properly
        dual = nx.relabel_nodes(dual, face_list)

        # Fix node labels
        labels = {
            node: f"({', '.join(node)})"
            for node in dual.nodes
        }

        # Draw graph
        plt.figure(figsize=(8, 6))
        pos = nx.kamada_kawai_layout(dual)
        nx.draw_networkx_edges(dual, pos, width=2)
        nx.draw(
            dual, pos,
            with_labels=True,
            labels=labels,
            node_color="lightcoral",
            edge_color="black",
            node_size=1000,
            font_size=10
        )
        plt.title("Dual Graph")
        plt.show()
        
        # Generate random sell dates using a random stock (doesn't matter which one since all stocks use the same dates)
        df = pd.read_csv(
                f'Data2015-2025/HistoricalPrices 2015 - 2025, {stocks[0]}.csv',
                parse_dates=['Date'],
                date_format="%m/%d/%y"
            )
        df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%y", errors='coerce')

        # Convert sell range strings to timestamps
        sell1 = pd.Timestamp(sell1)
        sell2 = pd.Timestamp(sell2)

        # Filter rows between sell1 and sell2
        filtered_sell_df = df[df['Date'] > sell1]
        filtered_sell_df.index = pd.to_datetime(filtered_sell_df.index)
        filtered_sell_df = filtered_sell_df[filtered_sell_df.index < sell2]

        # Get all valid dates in that range
        filtered_sell_df.set_index('Date', inplace=True)
        valid_dates = filtered_sell_df.index


        # Run variance-minimized simulations of triads and count wins
        for node in dual.nodes:
            
            SPX_beat_count = 0
            triad_percent_changes = []
            portfolio_returns = []

            for i in range(sims):
                # Pick a sell random date
                random_sell_date = np.random.choice(valid_dates)

                percent_changes = pd.DataFrame()
                random_period_percent_changes = []
                for s in node:
                    df = pd.read_csv(
                    f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
                    parse_dates=['Date'],
                    date_format="%m/%d/%y")

                    # Set Date as index
                    df.set_index('Date', inplace=True)

                    # Use pd.Timestamp for the date lookup
                    target_date1 = pd.Timestamp(buy_date)
                    target_date2 = pd.Timestamp(random_sell_date)

                    # Get the opening value for a specific date
                    open_value = df.loc[target_date1, " Open"]
                    close_value = df.loc[target_date2, " Close"]
                    random_period_percent_changes.append((close_value - open_value)/open_value * 100)

                    # Change Date back into a column so it can be accessed as normal
                    df.reset_index(inplace=True)
                    # Remove spaces in column names.
                    df.columns = df.columns.str.strip()
                    # Remove some recent data to analyze profits in the past
                    cutoff_date = pd.to_datetime(buy_date)
                    df = df[df['Date'] < cutoff_date]
                    # Calculate daily percent change for a stock and add it as a new column to df.
                    df['Percent Change'] = ((df['Close'] - df['Open']) / df['Open']) * 100
                    # Store percent changes into a single DataFrame, with each column representing a different stock.
                    percent_changes[s] = df['Percent Change']
                
                # Make covariance matrix of a 3-stock portoflio
                cov_mat = percent_changes.cov()
                # Calculate variance-minimizing stock allocations
                mat = np.array(2*cov_mat)
                mat = np.vstack([mat, np.ones((1, mat.shape[1]))])
                mat = np.hstack([mat, np.ones((mat.shape[0], 1))])
                mat[-1][-1] = 0
                mat_inv = np.linalg.inv(mat)
                B = np.zeros((mat.shape[1], 1))
                B[-1] = 1
                stock_allocations = np.dot(mat_inv, B)
                # Remove last row of stock_allocations and retrieve its important values
                stock_allocations = np.delete(stock_allocations, -1, axis=0)
                x = np.dot(np.transpose(stock_allocations), cov_mat)
                variance = np.dot(x, stock_allocations)

                print('Portfolio minimum variance: ' + str(variance[0][0]))
                print('Portfolio allocations: ' + str(node[0]) + ': ' + str(stock_allocations[0][0]) + '\n                       ' + str(node[1]) + ': ' + str(stock_allocations[1][0]) + '\n                       ' + str(node[2]) + ': ' + str(stock_allocations[2][0]))
                

                random_period_portfolio_percent_change = 0
                for i in range(len(random_period_percent_changes)):
                    random_period_portfolio_percent_change += random_period_percent_changes[i] * stock_allocations[i][0]
                
                print(f'Portfolio percent change from {buy_date} to {random_sell_date}: ' + str(random_period_portfolio_percent_change))
                print('--------------------------------------------\n')

                portfolio_returns.append(random_period_portfolio_percent_change)
                

                df = pd.read_csv(
                    f'Data2015-2025/HistoricalPrices 2015 - 2025, SPX.csv',
                    parse_dates=['Date'],
                    date_format="%m/%d/%y")

                # Set Date as index
                df.set_index('Date', inplace=True)

                # Use pd.Timestamp for the date lookup
                target_date1 = pd.Timestamp(buy_date)
                target_date2 = pd.Timestamp(random_sell_date)

                open_value = df.loc[target_date1, " Open"]
                close_value = df.loc[target_date2, " Close"]
                SPX_percent_change = (close_value - open_value)/open_value * 100
                print(f'SPX percent change from {buy_date} to {random_sell_date}: ' + str(SPX_percent_change))
                print('--------------------------------------------\n')
                if random_period_portfolio_percent_change > SPX_percent_change:
                    SPX_beat_count += 1
                triad_percent_changes.append(random_period_portfolio_percent_change)
            
            avg_return = sum(triad_percent_changes)/len(triad_percent_changes)

            f.write(f'Portfolio: {node}\n')
            f.write(f'Average portfolio return: {avg_return}\n')
            f.write(f'Number of times SPX was beat: {SPX_beat_count}\n')
            f.write('------------------------------------------------------\n')

                
            
        stock_frequency = {}
        for s1 in stocks:
            stock_frequency[s1] = 0
            for node in dual.nodes:
                for s2 in node:
                    if s1 == s2:
                        stock_frequency[s1] += 1
        f.write(f'Stock frequencies: {stock_frequency}\n')
        f.write('------------------------------------------------------\n')
        end_time = time.time()
        f.write(f'Execution time: {end_time - start_time:.4f} seconds')




# Identify dates that are below the n-day EMA of a portfolio
def find_dips(stocks: list, n: int, first_date: str, last_date: str):
    overlap_dates = None
    first_date = pd.to_datetime(first_date)
    last_date = pd.to_datetime(last_date)
    for s in stocks:
        df = pd.read_csv(
            f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
            parse_dates=['Date'],
            date_format="%m/%d/%y")
        df = df[(df['Date'] >= first_date) & (df['Date'] <= last_date)]
        ema_open = df[' Open'].ewm(span=n, adjust=False).mean()
        # Find dates where the current opening price is less than the n-day EMA
        mask = df[' Open'] < ema_open
        dates_below_ema = set(df.loc[mask, 'Date'])
        if overlap_dates is None:
            overlap_dates = dates_below_ema  # Initialize on first iteration
        else:
            overlap_dates = overlap_dates & dates_below_ema  # Intersect sets
    return list(overlap_dates)

# Identify dates that are above the n-day EMA of a portfolio
def find_peaks(stocks: list, n: int, first_date: str, last_date: str):
    overlap_dates = None
    first_date = pd.to_datetime(first_date)
    last_date = pd.to_datetime(last_date)
    for s in stocks:
        df = pd.read_csv(
            f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
            parse_dates=['Date'],
            date_format="%m/%d/%y")
        df = df[(df['Date'] >= first_date) & (df['Date'] <= last_date)]
        df['Date'] = df['Date'].dt.date
        ema_open = df[' Open'].ewm(span=n, adjust=False).mean()
        # Find dates where the current opening price is less than the n-day EMA
        mask = df[' Open'] > ema_open
        dates_below_ema = set(df.loc[mask, 'Date'])
        if overlap_dates is None:
            overlap_dates = dates_below_ema  # Initialize on first iteration
        else:
            overlap_dates = overlap_dates & dates_below_ema  # Intersect sets
    return list(overlap_dates)



def simulate_3combos(sims: int, seed: int, stocks: list[str], buy_date: str, sell1: str, sell2: str):
    start_time = time.time()
    with open(os.path.join('Simulations', f'{sims}_3ComboSims_{buy_date}_and_{sell1}_to_{sell2}.txt'), "w", newline="") as f:
        f.write(f'Stocks ({len(stocks)}): {stocks}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Simulations: {sims}\n')
        f.write(f'Starting date: {buy_date}\n')
        f.write(f'Ending date range: {sell1} to {sell2}\n')
        f.write('------------------------------------------------------\n')
        # Generate random sell dates using a random stock (doesn't matter which one since all stocks use the same dates)
        df = pd.read_csv(
                f'Data2015-2025/HistoricalPrices 2015 - 2025, {stocks[0]}.csv',
                parse_dates=['Date'],
                date_format="%m/%d/%y"
            )
        df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%y", errors='coerce')

        # Convert sell range strings to timestamps
        sell1 = pd.Timestamp(sell1)
        sell2 = pd.Timestamp(sell2)

        # Filter rows between sell1 and sell2
        filtered_sell_df = df[df['Date'] > sell1]
        filtered_sell_df.index = pd.to_datetime(filtered_sell_df.index)
        filtered_sell_df = filtered_sell_df[filtered_sell_df.index < sell2]

        # Get all valid dates in that range
        filtered_sell_df.set_index('Date', inplace=True)
        valid_dates = filtered_sell_df.index


        # Run variance-minimized simulations of all possible triads and count wins
        combos = list(combinations(stocks, 3))
        for c in combos:
            
            SPX_beat_count = 0
            triad_percent_changes = []
            portfolio_returns = []

            for i in range(sims):
                # Pick a sell random date
                random_sell_date = np.random.choice(valid_dates)

                percent_changes = pd.DataFrame()
                random_period_percent_changes = []
                for s in c:
                    df = pd.read_csv(
                    f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
                    parse_dates=['Date'],
                    date_format="%m/%d/%y")

                    # Set Date as index
                    df.set_index('Date', inplace=True)

                    # Use pd.Timestamp for the date lookup
                    target_date1 = pd.Timestamp(buy_date)
                    target_date2 = pd.Timestamp(random_sell_date)

                    # Get the opening value for a specific date
                    open_value = df.loc[target_date1, " Open"]
                    close_value = df.loc[target_date2, " Close"]
                    random_period_percent_changes.append((close_value - open_value)/open_value * 100)

                    # Change Date back into a column so it can be accessed as normal
                    df.reset_index(inplace=True)
                    # Remove spaces in column names.
                    df.columns = df.columns.str.strip()
                    # Remove some recent data to analyze profits in the past
                    cutoff_date = pd.to_datetime(buy_date)
                    df = df[df['Date'] < cutoff_date]
                    # Calculate daily percent change for a stock and add it as a new column to df.
                    df['Percent Change'] = ((df['Close'] - df['Open']) / df['Open']) * 100
                    # Store percent changes into a single DataFrame, with each column representing a different stock.
                    percent_changes[s] = df['Percent Change']
                
                # Make covariance matrix of a 3-stock portoflio
                cov_mat = percent_changes.cov()
                # Calculate variance-minimizing stock allocations
                mat = np.array(2*cov_mat)
                mat = np.vstack([mat, np.ones((1, mat.shape[1]))])
                mat = np.hstack([mat, np.ones((mat.shape[0], 1))])
                mat[-1][-1] = 0
                mat_inv = np.linalg.inv(mat)
                B = np.zeros((mat.shape[1], 1))
                B[-1] = 1
                stock_allocations = np.dot(mat_inv, B)
                # Remove last row of stock_allocations and retrieve its important values
                stock_allocations = np.delete(stock_allocations, -1, axis=0)
                x = np.dot(np.transpose(stock_allocations), cov_mat)
                variance = np.dot(x, stock_allocations)

                print('Portfolio minimum variance: ' + str(variance[0][0]))
                print('Portfolio allocations: ' + str(c[0]) + ': ' + str(stock_allocations[0][0]) + '\n                       ' + str(c[1]) + ': ' + str(stock_allocations[1][0]) + '\n                       ' + str(c[2]) + ': ' + str(stock_allocations[2][0]))
                

                random_period_portfolio_percent_change = 0
                for i in range(len(random_period_percent_changes)):
                    random_period_portfolio_percent_change += random_period_percent_changes[i] * stock_allocations[i][0]
                
                print(f'Portfolio percent change from {buy_date} to {random_sell_date}: ' + str(random_period_portfolio_percent_change))
                print('--------------------------------------------\n')

                portfolio_returns.append(random_period_portfolio_percent_change)
                

                df = pd.read_csv(
                    f'Data2015-2025/HistoricalPrices 2015 - 2025, SPX.csv',
                    parse_dates=['Date'],
                    date_format="%m/%d/%y")

                # Set Date as index
                df.set_index('Date', inplace=True)

                # Use pd.Timestamp for the date lookup
                target_date1 = pd.Timestamp(buy_date)
                target_date2 = pd.Timestamp(random_sell_date)

                open_value = df.loc[target_date1, " Open"]
                close_value = df.loc[target_date2, " Close"]
                SPX_percent_change = (close_value - open_value)/open_value * 100
                print(f'SPX percent change from {buy_date} to {random_sell_date}: ' + str(SPX_percent_change))
                print('--------------------------------------------\n')
                if random_period_portfolio_percent_change > SPX_percent_change:
                    SPX_beat_count += 1
                triad_percent_changes.append(random_period_portfolio_percent_change)
            
            avg_return = sum(triad_percent_changes)/len(triad_percent_changes)

            f.write(f'Portfolio: {c}\n')
            f.write(f'Average portfolio return: {avg_return}\n')
            f.write(f'Number of times SPX was beat: {SPX_beat_count}\n')
            f.write('------------------------------------------------------\n')

        end_time = time.time()
        f.write(f'Execution time: {end_time - start_time:.4f} seconds')
        # TODO: Calculate average return and average SP500 wins across all portfolios


    


stocks = ['DIS', 'KO', 'ADBE', 'MRK', 'KMI', 'AAPL', 'JNJ', 'CVS', 'COST', 'T', 'BA', 'EA', 'HAS', 'HD', 'HSY', 'LLY', 'NFLX', 'NKE', 'V', 'JPM', 'FDX', 'KR', 'KHC', 'LULU', 'MA', 'BBY', 'ALL', 'ABNB', 'GOOG', 'KMX']


# simulation2(1, 1, stocks, '2021-01-01', '2022-01-01', '2024-01-01', '2025-01-01')

# buy_dates = find_dips(['ADBE', 'HD', 'HSY'], 50, '2022-01-04', '2023-01-01')
# sell_dates = find_peaks(['ADBE', 'EA', 'LLY'], 50, '2023-01-04', '2024-01-01')
# print(buy_dates)
# print(sell_dates)


# simulation3(100, 1, stocks, '2022-01-04', '2023-01-01', '2025-01-01')
simulate_3combos(50, 1, stocks, '2022-01-04', '2023-01-01', '2025-01-01')