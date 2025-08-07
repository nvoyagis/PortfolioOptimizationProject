import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
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
import Charts
import random
import ta
import Simulations

def optimize_hot_stocks(stocks: list[str], date_tuples: list[tuple]):
    '''
    - date_tuples contains 4-tuples of date1, date2, date3, date4. Choose one tuple.
    - Choose a value n to highlight portfolios with the top n percentile of stocks
    - For a tuple, take all dates within [date3, date4] and count the number of times highlighted portfolios (H) and non-highlighted (N) are above SPX
    - Divide H by the total amount of highlighted portfolios and N by the total amount of non-highlighted portoflios
    - Repeat last 2 steps for each tuple
    - Repeat all steps for each n
    '''
    highlighted_wins1 = {}
    nonhighlighted_wins1 = {}
    highlighted_wins2 = {}
    nonhighlighted_wins2 = {}
    highlighted_wins3 = {}
    nonhighlighted_wins3 = {}

    highlighted_losses1 = {}
    nonhighlighted_losses1 = {}
    highlighted_losses2 = {}
    nonhighlighted_losses2 = {}
    highlighted_losses3 = {}
    nonhighlighted_losses3 = {}

    highlighted_win_percentage1 = {}
    nonhighlighted_win_percentage1 = {}
    highlighted_win_percentage2 = {}
    nonhighlighted_win_percentage2 = {}
    highlighted_win_percentage3 = {}
    nonhighlighted_win_percentage3 = {}

    for n in range(100, 80, -1):
        highlighted_wins1[n] = 0
        nonhighlighted_wins1[n] = 0
        highlighted_wins2[n] = 0
        nonhighlighted_wins2[n] = 0
        highlighted_wins3[n] = 0
        nonhighlighted_wins3[n] = 0
        highlighted_losses1[n] = 0
        nonhighlighted_losses1[n] = 0
        highlighted_losses2[n] = 0
        nonhighlighted_losses2[n] = 0
        highlighted_losses3[n] = 0
        nonhighlighted_losses3[n] = 0
        for date_tuple in date_tuples:
            date1 = date_tuple[0]
            date2 = date_tuple[1]
            date3 = date_tuple[2]
            date4 = date_tuple[3]
            stock_dict = {}
            for i in range(len(stocks)):
                stock_dict[i] = stocks[i]

            # Create TMFG
            cov_mat = Graph_Theory_Functions.get_weight_mat(stocks, date1, date2)
            cov_mat = pd.DataFrame(cov_mat)
            model = fast_tmfg.TMFG()
            w = pd.DataFrame(np.ones((len(stocks), len(stocks))))
            cliques, seps, adj_matrix = model.fit_transform(weights=cov_mat, output='weighted_sparse_W_matrix') # This only works with Pandas dataframes for whatever reason
            g = nx.from_numpy_array(adj_matrix)

            # Add node weights to TMFG. These weights are the percent change of a stock from begin_data_date to buy_date. Color nodes by their weight.
            stock_percent_changes = {}
            for s in stocks:
                df = pd.read_csv(
                                f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
                                parse_dates=['Date'],
                                date_format='%m/%d/%y')
                df.set_index('Date', inplace=True)
                open_value = df.loc[pd.Timestamp(date1), ' Open']
                close_value = df.loc[pd.Timestamp(date2), ' Close']
                stock_percent_changes[s] = (close_value - open_value) / open_value * 100
            g = nx.relabel_nodes(g, stock_dict)
            # Assign weights (percent changes) to nodes/stocks
            nx.set_node_attributes(g, stock_percent_changes, name='weight')


            '''
            1. Take node value
            2. Diffuse it across adjacent edges based off of edge weights. Keep some of it contained at the original node.
            3. Display graph using new node values
            '''
            # Calculate ADX for all stocks. ADX measures the strength of a stock's trend.
            # Convert strings to datetime objects
            d1 = pd.to_datetime(date1)
            d2 = pd.to_datetime(date2)

            # Calculate the difference in days
            day_difference = abs((d2 - d1).days)
            ADX_dict = {}
            for stock in stocks:
                df = pd.read_csv(
                                f'Data2015-2025/HistoricalPrices 2015 - 2025, {stock}.csv',
                                parse_dates=['Date'],
                                date_format='%m/%d/%y')
                df['ADX'] = ta.trend.adx(df[' High'], df[' Low'], df[' Close'], window=day_difference+1)
                df['+DI'] = ta.trend.adx_pos(df[' High'], df[' Low'], df[' Close'], window=day_difference+1)
                df['-DI'] = ta.trend.adx_neg(df[' High'], df[' Low'], df[' Close'], window=day_difference+1)
                print(df.min())
                print(df.max())
                df.set_index('Date', inplace=True)
                ADX_dict[stock] = df.loc[date2, 'ADX']


            def normalize_dict(data: dict) -> dict:
                """
                Normalize the values of a dictionary to the range [0, 1].
                """
                values = list(data.values())
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val
                normalized = {}

                if range_val == 0:
                    # All values are the same --> map everything to 0.5
                    for key in data:
                        normalized[key] = 0.5

                else:
                    for k, v in data.items():
                        normalized[k] = (v - min_val) / range_val
                
                return normalized

            # Normalize ADX_dict to be in [0, 1]
            ADX_dict = normalize_dict(ADX_dict)

            # Calculate "heat flow" out of each stock. Flow from one stock to another depends on the edge weight between them AND the total edge weight from the source stock.
            directed_edges = {}
            for v in g.nodes:
                total_local_edge_weight = 0
                for e in nx.edges(g, nbunch=v):
                    u = e[1]
                    total_local_edge_weight += g[v][u]['weight']
                for e in nx.edges(g, nbunch=v):
                    u = e[1]
                    directed_edges[(v, u)] = g[v][u]['weight'] / total_local_edge_weight
            
            # Create a new TMFG after heat flows across edges
            predicting_TMFG = g.copy()

            # Recalculate node weights
            for v in predicting_TMFG.nodes:
                predicting_TMFG.nodes[v]['weight'] = 0
            
            # For each node v, calculate total incoming heat from each adjacent node, u
            for v in predicting_TMFG.nodes:
                for e in nx.edges(predicting_TMFG, nbunch=v):
                    u = e[1]
                    '''
                    Formula: 1) g.nodes[u]['weight']: Heat from an adjacent node u
                            2) (1 - ADX_dict[u]): More signficant trend => (1 - ADX_dict[u]) approaches 0 => We care more about the stock with the strong trend instead of its adjacent stocks
                            3) directed_edges[(u, v)]: Percentage of POSSIBLE heat from u that goes into the direction of v. Possible heat: (1 - ADX_dict[u]) * g.nodes[u]['weight']
                    '''
                    predicting_TMFG.nodes[v]['weight'] += directed_edges[(u, v)] * (1 - ADX_dict[u]) * g.nodes[u]['weight']
            
            # Add back each node's retained heat. This heat is was not shared with other nodes since ADX_dict[v] is within [0, 1].
            for v in predicting_TMFG.nodes:
                predicting_TMFG.nodes[v]['weight'] += ADX_dict[v] * g.nodes[v]['weight']

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

            allocations = {}

            # Run variance-minimized simulations of triads and count wins
            for node in dual.nodes:
                # Find variance-minimizing portfolio allocations
                percent_changes = pd.DataFrame()
                for s in node:
                    df = pd.read_csv(
                                    f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
                                    parse_dates=['Date'],
                                    date_format='%m/%d/%y')
                    df = df[df['Date'] <= date2]
                    df = df[df['Date'] >= date1]
                    # Calculate daily percent change for a stock and add it as a new column to df.
                    df['Percent Change'] = ((df[' Close'] - df[' Open']) / df[' Open']) * 100
                    percent_changes[s] = df['Percent Change']
                small_cov_mat = percent_changes.cov()
                small_cov_mat = pd.DataFrame(small_cov_mat)
                print(small_cov_mat)
                # Calculate variance-minimizing stock allocations
                mat = np.array(2*small_cov_mat)
                mat = np.vstack([mat, np.ones((1, mat.shape[1]))])
                mat = np.hstack([mat, np.ones((mat.shape[0], 1))])
                mat[-1][-1] = 0
                mat_inv = np.linalg.inv(mat)
                B = np.zeros((mat.shape[1], 1))
                B[-1] = 1
                stock_allocations = np.dot(mat_inv, B)
                # Remove last row of stock_allocations and retrieve its important values
                stock_allocations = np.delete(stock_allocations, -1, axis=0)
                x = np.dot(np.transpose(stock_allocations), small_cov_mat)
                variance = np.dot(x, stock_allocations)
                print(f'Portfolio minimum variance: {variance[0][0]}')
                print(f'Portfolio allocations:  {node[0]}: {stock_allocations[0][0]}\n                       {node[1]}: {stock_allocations[1][0]}\n                       {node[2]}: {stock_allocations[2][0]}')
                allocations[node] = [stock_allocations[0][0], stock_allocations[1][0], stock_allocations[2][0]]

                # NOTE: Significant new code starts after this line
                # Evaluate performance of portfolio on each day from date3 to date4 (inclusive)
                d3 = pd.to_datetime(date3)
                d4 = pd.to_datetime(date4)

                SPX_df = pd.read_csv(
                    f'Data2015-2025/HistoricalPrices 2015 - 2025, SPX.csv',
                    parse_dates=['Date'],
                    date_format='%m/%d/%y',)
                market_dates = SPX_df['Date'].tolist()
                SPX_df.set_index('Date', inplace=True)
                SPX_opening_price = SPX_df.loc[d1, ' Open']

                df_dict = {s: pd.read_csv(
                        f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
                        parse_dates=['Date'],
                        date_format='%m/%d/%y',
                        index_col='Date') for s in stocks}

                day_counter = 0
                date_range = pd.date_range(d3, d4, freq='D')

                # Count number of times portfolios beat SPX in date_range
                for d in date_range:
                    if d in market_dates:
                        day_counter += 1

                        SPX_closing_price = SPX_df.loc[d, ' Close']
                        SPX_percent_change = (SPX_closing_price - SPX_opening_price) / SPX_opening_price

                        df0 = df_dict[node[0]]
                        df1 = df_dict[node[1]]
                        df2 = df_dict[node[2]]
                        stock0_opening_price = df0.loc[d3, ' Open']
                        stock1_opening_price = df1.loc[d3, ' Open']
                        stock2_opening_price = df2.loc[d3, ' Open']
                        stock0_closing_price = df0.loc[d, ' Close']
                        stock1_closing_price = df1.loc[d, ' Close']
                        stock2_closing_price = df2.loc[d, ' Close']
                        stock0_percent_change = (stock0_closing_price - stock0_opening_price) / stock0_opening_price
                        stock1_percent_change = (stock1_closing_price - stock1_opening_price) / stock1_opening_price
                        stock2_percent_change = (stock2_closing_price - stock2_opening_price) / stock2_opening_price

                        portfolio_percent_change = stock_allocations[0][0] * stock0_percent_change + stock_allocations[1][0] * stock1_percent_change + stock_allocations[2][0] * stock2_percent_change
                        
                        # NOTE: before heat spread
                        # Get node weights as a dictionary
                        node_weights = nx.get_node_attributes(g, 'weight')

                        # Calculate the nth percentile threshold
                        weights = list(node_weights.values())
                        threshold = np.percentile(weights, n)

                        # Select nodes whose weight is greater than or equal to the threshold
                        hot_stocks1 = []
                        for stock, weight in node_weights.items():
                            if weight >= threshold:
                                hot_stocks1.append(stock)

                        # NOTE: after heat spread
                        # Get node weights as a dictionary
                        node_weights = nx.get_node_attributes(predicting_TMFG, 'weight')

                        # Calculate the nth percentile threshold
                        weights = list(node_weights.values())
                        threshold = np.percentile(weights, n)

                        # Select nodes whose weight is greater than or equal to the threshold
                        hot_stocks2 = []
                        for stock, weight in node_weights.items():
                            if weight >= threshold:
                                hot_stocks2.append(stock)

                        hot_stocks3 = list(set(hot_stocks1).intersection(set(hot_stocks2)))

                        if portfolio_percent_change > SPX_percent_change:
                            if len(set(hot_stocks1).intersection(set(node))) != 0:
                                highlighted_wins1[n] = highlighted_wins1[n] + 1
                            else:
                                nonhighlighted_wins1[n] = nonhighlighted_wins1[n] + 1
                            if len(set(hot_stocks2).intersection(set(node))) != 0:
                                highlighted_wins2[n] = highlighted_wins2[n] + 1
                            else:
                                nonhighlighted_wins2[n] = nonhighlighted_wins2[n] + 1
                            if len(set(hot_stocks3).intersection(set(node))) != 0:
                                highlighted_wins3[n] = highlighted_wins3[n] + 1
                            else:
                                nonhighlighted_wins3[n] = nonhighlighted_wins3[n] + 1
                        else:
                            if len(set(hot_stocks1).intersection(set(node))) != 0:
                                highlighted_losses1[n] = highlighted_losses1[n] + 1
                            else:
                                nonhighlighted_losses1[n] = nonhighlighted_losses1[n] + 1
                            if len(set(hot_stocks2).intersection(set(node))) != 0:
                                highlighted_losses2[n] = highlighted_losses2[n] + 1
                            else:
                                nonhighlighted_losses2[n] = nonhighlighted_losses2[n] + 1
                            if len(set(hot_stocks3).intersection(set(node))) != 0:
                                highlighted_losses3[n] = highlighted_losses3[n] + 1
                            else:
                                nonhighlighted_losses3[n] = nonhighlighted_losses3[n] + 1


        print(highlighted_wins1)
        print(nonhighlighted_wins1)
        print(highlighted_wins2)
        print(nonhighlighted_wins2)
        print(highlighted_wins3)
        print(nonhighlighted_wins3)

        
        for k, v in highlighted_wins1.items():
            losses = highlighted_losses1[k]
            highlighted_win_percentage1[k] = v / (losses + v)
        for k, v in nonhighlighted_wins1.items():
            losses = nonhighlighted_losses1[k]
            nonhighlighted_win_percentage1[k] = v / (losses + v)
        for k, v in highlighted_wins2.items():
            losses = highlighted_losses2[k]
            highlighted_win_percentage2[k] = v / (losses + v)
        for k, v in nonhighlighted_wins2.items():
            losses = nonhighlighted_losses2[k]
            nonhighlighted_win_percentage2[k] = v / (losses + v)
        for k, v in highlighted_wins3.items():
            if v != 0 and highlighted_losses3[k] != 0:
                losses = highlighted_losses3[k]
                highlighted_win_percentage3[k] = v / (losses + v)
        for k, v in nonhighlighted_wins3.items():
            if v != 0 and nonhighlighted_losses3[k] != 0:
                losses = nonhighlighted_losses3[k]
                nonhighlighted_win_percentage3[k] = v / (losses + v)
        
        print(highlighted_win_percentage1)
        print(nonhighlighted_win_percentage1)
        print(highlighted_win_percentage2)
        print(nonhighlighted_win_percentage2)
        print(highlighted_win_percentage3)
        print(nonhighlighted_win_percentage3)
    Charts.graph_dict(highlighted_win_percentage1, 'n-values', 'Percentage of Wins (Historically High Growth)')
    Charts.graph_dict(highlighted_win_percentage2, 'n-values', 'Percentage of Wins (Predicted High Growth)')
    Charts.graph_dict(highlighted_win_percentage3, 'n-values', 'Percentage of Wins (Historically & Predicted High Growth Intersection)')
    Charts.graph_dict(nonhighlighted_win_percentage1, 'n-values', 'Percentage of Wins (Historically non-High Growth)')
    Charts.graph_dict(nonhighlighted_win_percentage2, 'n-values', 'Percentage of Wins (Predicted non-High Growth)')
    Charts.graph_dict(nonhighlighted_win_percentage3, 'n-values', 'Percentage of Wins (Historically & Predicted non-High Growth Intersection)')

        



random.seed(1)
stocks = ['DIS', 'KO', 'ADBE', 'MRK', 'KMI', 'AAPL', 'JNJ', 'CVS', 'COST', 'T', 'BA', 'EA', 'HAS', 'HD', 'HSY', 'LLY', 'NFLX', 'NKE', 'V', 'JPM', 'FDX', 'KR', 'KHC', 'LULU', 'MA', 'BBY', 'GOOG', 'ALL', 'KMX', 'MNST', 
          'AZO', 'COF', 'WFC', 'XOM', 'CVX', 'MAR', 'MCD', 'ORCL', 'UNH', 'EBAY', 'CPB', 'DPZ', 'JBHT', 'TSN', 'WYNN', 'DLTR', 'EXPE', 'JNPR', 'SJM', 'NTAP', 'ATO', 'AFL', 'DXCM', 'MCHP', 'DAL', 'TGT', 'PCG',
          'CSGP', 'WTW', 'AXON']

# optimize_hot_stocks(stocks, [('2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05'), ('2021-07-29', '2021-10-28', '2021-10-29', '2021-12-17'), ('2022-03-29', '2022-04-22', '2022-04-25', '2022-05-20'), ('2023-07-21', '2023-09-12', '2023-09-13', '2023-11-02')])
# hwp1 = {100: 0.25586854460093894, 99: 0.25586854460093894, 98: 0.37258347978910367, 97: 0.37258347978910367, 96: 0.307005171603197, 95: 0.307005171603197, 94: 0.32463768115942027, 93: 0.2972417840375587, 92: 0.2972417840375587, 91: 0.3258088788562829, 90: 0.3258088788562829, 89: 0.3204334365325077, 88: 0.3329292929292929, 87: 0.3329292929292929, 86: 0.33594478642044395, 85: 0.33594478642044395, 84: 0.33998100664767333, 83: 0.33734056589942824, 82: 0.33734056589942824, 81: 0.31348285640629026}
# nwp1 = {100: 0.3691262619967593, 99: 0.3691262619967593, 98: 0.3657232294248076, 97: 0.3657232294248076, 96: 0.3749738584872778, 95: 0.3749738584872778, 94: 0.37456242707117854, 93: 0.3841855480710349, 92: 0.3841855480710349, 91: 0.379094913896676, 90: 0.379094913896676, 89: 0.38351464435146443, 88: 0.3804894983509807, 87: 0.3804894983509807, 86: 0.3807938079380794, 85: 0.3807938079380794, 84: 0.3825093559188497, 83: 0.386592062998653, 82: 0.386592062998653, 81: 0.41312829928850126}
# hwp2 = {100: 0.1590087764584409, 99: 0.1590087764584409, 98: 0.13755395683453236, 97: 0.13755395683453236, 96: 0.13133583021223472, 95: 0.13133583021223472, 94: 0.18724279835390947, 93: 0.18211191070641797, 92: 0.18211191070641797, 91: 0.19732532751091703, 90: 0.19732532751091703, 89: 0.2023121387283237, 88: 0.2161058881741712, 87: 0.2161058881741712, 86: 0.2441900400686892, 85: 0.2441900400686892, 84: 0.2651646447140381, 83: 0.2730386385529273, 82: 0.2730386385529273, 81: 0.2678737713398862}
# nwp2 = {100: 0.3938080495356037, 99: 0.3938080495356037, 98: 0.4273293837039317, 97: 0.4273293837039317, 96: 0.4416459452955803, 95: 0.4416459452955803, 94: 0.4642857142857143, 93: 0.49591224257476973, 92: 0.49591224257476973, 91: 0.5015310586176728, 90: 0.5015310586176728, 89: 0.5069977426636569, 88: 0.5108488316642823, 87: 0.5108488316642823, 86: 0.5039420964198009, 85: 0.5039420964198009, 84: 0.4950276243093923, 83: 0.4882872773179969, 82: 0.4882872773179969, 81: 0.5058028500073454}
# hwp3 = {100: 0.0, 99: 0.0, 98: 0.35195530726256985, 97: 0.35195530726256985, 96: 0.21778447626224567, 95: 0.21778447626224567, 94: 0.3019502353732347, 93: 0.3052691867124857, 92: 0.3052691867124857, 91: 0.26050830889540566, 90: 0.26050830889540566, 89: 0.2815764482431149, 88: 0.3653004377238361, 87: 0.3653004377238361, 86: 0.33249290891900407, 85: 0.33249290891900407, 84: 0.304867634500427, 83: 0.3422004132231405, 82: 0.3422004132231405, 81: 0.27386451116243266}
# nwp3 = {100: 0.36854646544876885, 99: 0.36854646544876885, 98: 0.36667712582365863, 97: 0.36667712582365863, 96: 0.3792010564542753, 95: 0.3792010564542753, 94: 0.3725725725725726, 93: 0.37342115985332064, 92: 0.37342115985332064, 91: 0.3811867461527797, 90: 0.3811867461527797, 89: 0.3786022553250731, 88: 0.3663586216777706, 87: 0.3663586216777706, 86: 0.3742386645612452, 85: 0.3742386645612452, 84: 0.3828227486688788, 83: 0.37357142857142855, 82: 0.37357142857142855, 81: 0.40874423554451933}
# Charts.graph_dict(hwp1, 'n-values', 'Percentage of Wins (Historically High Growth)')
# Charts.graph_dict(nwp1, 'n-values', 'Percentage of Wins (Historically non-High Growth)')
# Charts.graph_dict(hwp2, 'n-values', 'Percentage of Wins (Predicted High Growth)')
# Charts.graph_dict(nwp2, 'n-values', 'Percentage of Wins (Predicted non-High Growth)')
# Charts.graph_dict(hwp3, 'n-values', 'Percentage of Wins (Historically & Predicted High Growth Intersection)')
# Charts.graph_dict(nwp3, 'n-values', 'Percentage of Wins (Historically & Predicted non-High Growth Intersection)')

# optimize_hot_stocks(stocks, [('2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05'), ('2021-07-29', '2021-10-28', '2021-10-29', '2021-12-17')])
# hwp1= {100: 0.0, 99: 0.0, 98: 0.1037593984962406, 97: 0.1037593984962406, 96: 0.052434456928838954}
# nwp1={100: 0.0862051015096304, 99: 0.0862051015096304, 98: 0.08254486133768352, 97: 0.08254486133768352, 96: 0.0889149560117302}
# hwp2={100: 0.05498489425981873, 99: 0.05498489425981873, 98: 0.05328836424957842, 97: 0.05328836424957842, 96: 0.04599708879184862}
# nwp2={100: 0.08982327848872639, 99: 0.08982327848872639, 98: 0.09717186366932559, 97: 0.09717186366932559, 96: 0.10428015564202335}
# hwp3={100: 0.0, 99: 0.0, 98: 0.07301587301587302, 97: 0.07301587301587302, 96: 0.024365482233502538}
# nwp3={100: 0.0848795489492568, 99: 0.0848795489492568, 98: 0.08433734939759036, 97: 0.08433734939759036, 96: 0.09059154929577465}
# Charts.graph_dict(hwp1, 'n-values', 'Percentage of Wins (Historically High Growth)')
# Charts.graph_dict(nwp1, 'n-values', 'Percentage of Wins (Historically non-High Growth)')
# Charts.graph_dict(hwp2, 'n-values', 'Percentage of Wins (Predicted High Growth)')
# Charts.graph_dict(nwp2, 'n-values', 'Percentage of Wins (Predicted non-High Growth)')
# Charts.graph_dict(hwp3, 'n-values', 'Percentage of Wins (Historically & Predicted High Growth Intersection)')
# Charts.graph_dict(nwp3, 'n-values', 'Percentage of Wins (Historically & Predicted non-High Growth Intersection)')

# optimize_hot_stocks(stocks, [('2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05')])
# optimize_hot_stocks(stocks, [('2021-07-29', '2021-10-28', '2021-10-29', '2021-12-17')])


# optimize_hot_stocks(stocks, [('2022-03-29', '2022-04-22', '2022-04-25', '2022-05-20')])
# hwp1 = {100: 0.8, 99: 0.8, 98: 0.9142857142857143, 97: 0.9142857142857143, 96: 0.895, 95: 0.895, 94: 0.925, 93: 0.9382352941176471, 92: 0.9382352941176471, 91: 0.8685185185185185, 90: 0.8685185185185185, 89: 0.8816666666666667, 88: 0.8787878787878788, 87: 0.8787878787878788, 86: 0.8823529411764706, 85: 0.8823529411764706, 84: 0.8905405405405405, 83: 0.9035714285714286, 82: 0.9035714285714286, 81: 0.91}
# nwp1 = {100: 0.9132743362831859, 99: 0.9132743362831859, 98: 0.9100917431192661, 97: 0.9100917431192661, 96: 0.9117924528301887, 95: 0.9117924528301887, 94: 0.9083333333333333, 93: 0.9055555555555556, 92: 0.9055555555555556, 91: 0.9230337078651686, 90: 0.9230337078651686, 89: 0.9203488372093023, 88: 0.9228915662650602, 87: 0.9228915662650602, 86: 0.9219512195121952, 85: 0.9219512195121952, 84: 0.919620253164557, 83: 0.9141891891891892, 82: 0.9141891891891892, 81: 0.9105633802816901}
# hwp2 = {100: 0.85, 99: 0.85, 98: 0.9357142857142857, 97: 0.9357142857142857, 96: 0.895, 95: 0.895, 94: 0.925, 93: 0.9382352941176471, 92: 0.9382352941176471, 91: 0.95625, 90: 0.95625, 89: 0.958, 88: 0.9611111111111111, 87: 0.9611111111111111, 86: 0.965, 85: 0.965, 84: 0.9652777777777778, 83: 0.9652777777777778, 82: 0.9652777777777778, 81: 0.9652777777777778}
# nwp2 = {100: 0.9119469026548672, 99: 0.9119469026548672, 98: 0.9087155963302752, 97: 0.9087155963302752, 96: 0.9117924528301887, 95: 0.9117924528301887, 94: 0.9083333333333333, 93: 0.9055555555555556, 92: 0.9055555555555556, 91: 0.8983695652173913, 90: 0.8983695652173913, 89: 0.8972527472527473, 88: 0.8949438202247191, 87: 0.8949438202247191, 86: 0.8912790697674419, 85: 0.8912790697674419, 84: 0.885625, 83: 0.885625, 82: 0.885625, 81: 0.885625}
# hwp3 = {96: 0.825, 95: 0.825, 94: 0.925, 93: 0.925, 92: 0.925, 91: 0.925, 90: 0.925, 89: 0.9382352941176471, 88: 0.9382352941176471, 87: 0.9382352941176471, 86: 0.9475, 85: 0.9475, 84: 0.9475, 83: 0.958, 82: 0.958, 81: 0.958}
# nwp3 = {100: 0.9103448275862069, 99: 0.9103448275862069, 98: 0.9103448275862069, 97: 0.9103448275862069, 96: 0.915, 95: 0.915, 94: 0.9083333333333333, 93: 0.9083333333333333, 92: 0.9083333333333333, 91: 0.9083333333333333, 90: 0.9083333333333333, 89: 0.9055555555555556, 88: 0.9055555555555556, 87: 0.9055555555555556, 86: 0.9026041666666667, 85: 0.9026041666666667, 84: 0.9026041666666667, 83: 0.8972527472527473, 82: 0.8972527472527473, 81: 0.8972527472527473}
# Charts.graph_dict(hwp1, 'n-values', 'Percentage of Wins (Historically High Growth)')
# Charts.graph_dict(nwp1, 'n-values', 'Percentage of Wins (Historically non-High Growth)')
# Charts.graph_dict(hwp2, 'n-values', 'Percentage of Wins (Predicted High Growth)')
# Charts.graph_dict(nwp2, 'n-values', 'Percentage of Wins (Predicted non-High Growth)')
# Charts.graph_dict(hwp3, 'n-values', 'Percentage of Wins (Historically & Predicted High Growth Intersection)')
# Charts.graph_dict(nwp3, 'n-values', 'Percentage of Wins (Historically & Predicted non-High Growth Intersection)')

# optimize_hot_stocks(stocks, [('2022-08-12', '2022-09-21', '2022-09-22', '2022-10-14')])
# hwp1 = {100: 1.0, 99: 1.0, 98: 1.0, 97: 1.0, 96: 1.0, 95: 1.0, 94: 1.0, 93: 1.0, 92: 1.0, 91: 1.0, 90: 1.0, 89: 1.0, 88: 1.0, 87: 1.0, 86: 1.0, 85: 1.0, 84: 1.0, 83: 1.0, 82: 1.0, 81: 0.9900110987791343}
# nwp1 = {100: 0.9906298802706923, 99: 0.9906298802706923, 98: 0.9903743315508021, 97: 0.9903743315508021, 96: 0.9901044529961517, 95: 0.9901044529961517, 94: 0.9889705882352942, 93: 0.9886148007590133, 92: 0.9886148007590133, 91: 0.9881031064111038, 90: 0.9881031064111038, 89: 0.9879679144385026, 88: 0.987688098495212, 87: 0.987688098495212, 86: 0.9872430900070872, 85: 0.9872430900070872, 84: 0.986764705882353, 83: 0.9862490450725745, 82: 0.9862490450725745, 81: 0.9915966386554622}
# hwp2 = {100: 1.0, 99: 1.0, 98: 1.0, 97: 1.0, 96: 1.0, 95: 1.0, 94: 1.0, 93: 1.0, 92: 1.0, 91: 1.0, 90: 1.0, 89: 1.0, 88: 1.0, 87: 1.0, 86: 1.0, 85: 1.0, 84: 1.0, 83: 0.9747899159663865, 82: 0.9747899159663865, 81: 0.9767801857585139}
# nwp2 = {100: 0.9906298802706923, 99: 0.9906298802706923, 98: 0.9903743315508021, 97: 0.9903743315508021, 96: 0.9901044529961517, 95: 0.9901044529961517, 94: 0.9898190045248869, 93: 0.9895165987186954, 92: 0.9895165987186954, 91: 0.9891956782713085, 90: 0.9891956782713085, 89: 0.9887359198998749, 88: 0.9883645765998708, 87: 0.9883645765998708, 86: 0.9878296146044625, 85: 0.9878296146044625, 84: 0.9873949579831933, 83: 0.9978213507625272, 82: 0.9978213507625272, 81: 0.997737556561086}
# hwp3 = {}
# nwp3 = {100: 0.9908722109533469, 99: 0.9908722109533469, 98: 0.9903743315508021, 97: 0.9903743315508021, 96: 0.9901044529961517, 95: 0.9901044529961517, 94: 0.9901044529961517, 93: 0.9898190045248869, 92: 0.9898190045248869, 91: 0.9898190045248869, 90: 0.9898190045248869, 89: 0.9895165987186954, 88: 0.9895165987186954, 87: 0.9895165987186954, 86: 0.9895165987186954, 85: 0.9895165987186954, 84: 0.9895165987186954, 83: 0.9891956782713085, 82: 0.9891956782713085, 81: 0.9891956782713085}
# Charts.graph_dict(hwp1, 'n-values', 'Percentage of Wins (Historically High Growth)')
# Charts.graph_dict(nwp1, 'n-values', 'Percentage of Wins (Historically non-High Growth)')
# Charts.graph_dict(hwp2, 'n-values', 'Percentage of Wins (Predicted High Growth)')
# Charts.graph_dict(nwp2, 'n-values', 'Percentage of Wins (Predicted non-High Growth)')
# Charts.graph_dict(hwp3, 'n-values', 'Percentage of Wins (Historically & Predicted High Growth Intersection)')
# Charts.graph_dict(nwp3, 'n-values', 'Percentage of Wins (Historically & Predicted non-High Growth Intersection)')

# optimize_hot_stocks(stocks, [('2023-07-21', '2023-09-12', '2023-09-13', '2023-11-02')])
# hwp1 = {100: 0.5495495495495496, 99: 0.5495495495495496, 98: 0.6816816816816816, 97: 0.6816816816816816, 96: 0.6824324324324325, 95: 0.6824324324324325, 94: 0.7405405405405405, 93: 0.6430180180180181, 92: 0.6430180180180181, 91: 0.66008316008316, 90: 0.66008316008316, 89: 0.66008316008316, 88: 0.7054054054054054, 87: 0.7054054054054054, 86: 0.7256347256347256, 85: 0.7256347256347256, 84: 0.7113022113022113, 83: 0.6807432432432432, 82: 0.6807432432432432, 81: 0.6872586872586872}
# nwp1 = {100: 0.7249461851231763, 99: 0.7249461851231763, 98: 0.7236675928264713, 97: 0.7236675928264713, 96: 0.7264864864864865, 95: 0.7264864864864865, 94: 0.7162162162162162, 93: 0.7405992949471211, 92: 0.7405992949471211, 91: 0.7378378378378379, 90: 0.7378378378378379, 89: 0.7378378378378379, 88: 0.7256442489000628, 87: 0.7256442489000628, 86: 0.7183327906219472, 85: 0.7183327906219472, 84: 0.725975975975976, 83: 0.7484101748807631, 82: 0.7484101748807631, 81: 0.7446551028640581}
# hwp2 = {100: 0.7477477477477478, 99: 0.7477477477477478, 98: 0.5108108108108108, 97: 0.5108108108108108, 96: 0.5108108108108108, 95: 0.5108108108108108, 94: 0.5675675675675675, 93: 0.5695695695695696, 92: 0.5695695695695696, 91: 0.5563839701770736, 90: 0.5563839701770736, 89: 0.5850043591979076, 88: 0.6084733382030679, 87: 0.6084733382030679, 86: 0.6264264264264264, 85: 0.6264264264264264, 84: 0.6502384737678856, 83: 0.6656656656656657, 82: 0.6656656656656657, 81: 0.6687960687960688}
# nwp2 = {100: 0.7189189189189189, 99: 0.7189189189189189, 98: 0.7401835798062213, 97: 0.7401835798062213, 96: 0.7401835798062213, 95: 0.7401835798062213, 94: 0.7645645645645646, 93: 0.7661706650470695, 92: 0.7661706650470695, 91: 0.775085430257844, 90: 0.775085430257844, 89: 0.7697933227344992, 88: 0.772836127266507, 87: 0.772836127266507, 86: 0.7799771602588504, 85: 0.7799771602588504, 84: 0.7754677754677755, 83: 0.7680906713164778, 82: 0.7680906713164778, 81: 0.7669472751439964}
# hwp3 = {98: 0.7477477477477478, 97: 0.7477477477477478, 96: 0.7477477477477478, 95: 0.7477477477477478, 94: 0.7477477477477478, 93: 0.5197505197505198, 92: 0.5197505197505198, 91: 0.5197505197505198, 90: 0.5197505197505198, 89: 0.5197505197505198, 88: 0.6475225225225225, 87: 0.6475225225225225, 86: 0.6475225225225225, 85: 0.6475225225225225, 84: 0.6475225225225225, 83: 0.6355710549258936, 82: 0.6355710549258936, 81: 0.6494676494676495}
# nwp3 = {100: 0.7204100652376515, 99: 0.7204100652376515, 98: 0.7189189189189189, 97: 0.7189189189189189, 96: 0.7189189189189189, 95: 0.7189189189189189, 94: 0.7189189189189189, 93: 0.7457360272894253, 92: 0.7457360272894253, 91: 0.7457360272894253, 90: 0.7457360272894253, 89: 0.7457360272894253, 88: 0.7394242068155111, 87: 0.7394242068155111, 86: 0.7394242068155111, 85: 0.7394242068155111, 84: 0.7394242068155111, 83: 0.7513513513513513, 82: 0.7513513513513513, 81: 0.7486160859654836}
# Charts.graph_dict(hwp1, 'n-values', 'Percentage of Wins (Historically High Growth)')
# Charts.graph_dict(nwp1, 'n-values', 'Percentage of Wins (Historically non-High Growth)')
# Charts.graph_dict(hwp2, 'n-values', 'Percentage of Wins (Predicted High Growth)')
# Charts.graph_dict(nwp2, 'n-values', 'Percentage of Wins (Predicted non-High Growth)')
# Charts.graph_dict(hwp3, 'n-values', 'Percentage of Wins (Historically & Predicted High Growth Intersection)')
# Charts.graph_dict(nwp3, 'n-values', 'Percentage of Wins (Historically & Predicted non-High Growth Intersection)')