import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import MATLAB_to_Python as m2p
import pandas as pd
import numpy as np
from datetime import datetime
# import Portfolio_Optimization as po

def build_dual_from_faces(faces):
    """
    Build the dual graph from faces from the TMFG MATLAB code.
    Each face is a node labeled by its sorted vertex tuple.
    Edges connect nodes that share exactly 2 vertices.
    """
    dual = nx.Graph()

    # Add each face as a node for the dual graph
    for face in faces:
        face = tuple(sorted(face))
        dual.add_node(face)

    # Connect dual vertices with exactly two stocks in common
    for f1, f2 in combinations(faces, 2):
        if len(set(f1) & set(f2)) == 2:
            dual.add_edge(tuple(sorted(f1)), tuple(sorted(f2)))

    return dual

# Example faces (from TMFG))
faces = '''
     3     9    10
     9    17    16
     3     9     1
     3    17     6
     3    17    20
     9    17     4
     3     2    18
     9     2    18
     3     1     6
    17     1     6
     3     2    20
    17     2     7
     3    18     8
     9    18     8
     9     2     4
    17     2     4
    17    20     7
     2    20     7
     3     8    12
     9     8    12
     3    12    13
     9    12    13
     9     1    15
    17     1    15
     3    13     5
     9    13     5
     3     5    19
     9     5    19
     3    19    14
     9    19    14
     3    14    10
     9    14    10
     9    15    11
    17    15    16
     9    16    11
    15    16    11
    '''

# Replace face numbers with stock tuples
stock_list = ['DIS', 'KO', 'ADBE', 'MRK', 'KMI', 'AAPL', 'JNJ', 'CVS', 'COST', 'T', 'BA', 'EA', 'HAS', 'HD', 'HSY', 'LLY', 'NFLX', 'NKE', 'V', 'JPM']
stock_list.sort()
# faces2 = []
# for face in faces:
#     faces2.append((stock_list[face[0]-1], stock_list[face[1]-1], stock_list[face[2]-1]))


stock_faces = m2p.matlab_to_python_faces(faces, stock_list)


dual = build_dual_from_faces(stock_faces)

# Format node labels
labels = {
    node: f"({', '.join(node)})"
    for node in dual.nodes
}

# Draw graph
plt.figure(figsize=(8, 6))
pos = nx.kamada_kawai_layout(dual)



# Add covariances of two unique stocks connected by an edge. Also record the standard deviations of all stocks. NOTE: THIS IS WRONG
# mapping = {}
# standard_deviations = {}
# complete = nx.complete_graph(20)
# percent_changes = pd.DataFrame()
# for stock in stock_list:
#     df = pd.read_csv(f'Data2024/{stock}.csv', parse_dates=['Date'])
#     df.columns = df.columns.str.strip()
#     df['Percent Change'] = ((df['Close'] - df['Open']) / df['Close']) * 100.
#     percent_changes[stock] = df['Percent Change']
#     mapping[stock_list.index(stock)] = stock
#     standard_deviations[stock] = df['Percent Change'].std()



# Minimize risk in triads
portfolio_returns = []
for node in dual.nodes:
    percent_changes = pd.DataFrame()
    percent_changes_2023 = []
    for s in node:
        df = pd.read_csv(
        f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
        parse_dates=['Date'],
        date_format="%m/%d/%y")

        # Set Date as index
        df.set_index('Date', inplace=True)

        # Use pd.Timestamp for the date lookup
        target_date1 = pd.Timestamp("2023-01-03")
        target_date2 = pd.Timestamp("2024-01-02")

        # Get the opening value for a specific date
        open_value = df.loc[target_date1, " Open"]
        close_value = df.loc[target_date2, " Close"]
        percent_changes_2023.append((close_value - open_value)/open_value)

        # Change Date back into a column so it can be accessed as normal
        df.reset_index(inplace=True)
        # Remove spaces in column names.
        df.columns = df.columns.str.strip()
        # Remove some recent data to analyze profits in the past
        cutoff_date = pd.to_datetime('2023-01-01')
        df = df[df['Date'] <= cutoff_date]
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
    

    portfolio_percent_change_2023 = 0
    for i in range(len(percent_changes_2023)):
        portfolio_percent_change_2023 += percent_changes_2023[i] * stock_allocations[i][0]
    
    print('Portfolio percent change in 2023-2024: ' + str(portfolio_percent_change_2023))
    print('--------------------------------------------')

    portfolio_returns.append(portfolio_percent_change_2023)




# MOVE TO Portfolio_Optimization.py
# positive_return_count = 0
# better_than_spx_count = 0
# for p in portfolio_returns:
#     if p > 0:
#         positive_return_count += 1
#     if p > 0.3:
#         better_than_spx_count += 1
# negative_return_count = len(portfolio_returns) - positive_return_count
# print('Number of profitable portfolios: ' + str(positive_return_count))
# print('Number of portfolios with negative returns: ' + str(negative_return_count))
# print('Number of portfolios outperforming the SPX: ' + str(better_than_spx_count))

'''
# NOTE: YOU MUST CHANGE THE GRAPH DATA DEPENDING ON THE 
# START DATE OF THE SIMULATION
with open('100_Sims_2018-2019_and_2020-2021.txt', "w", newline="") as f: # Make file for data
    seed = 2
    sims = 100
    start1 = '01/01/2018'
    end1 = '01/01/2019'
    start2 = '01/01/2020'
    end2 = '01/01/2021'
    f.write('Stocks: ' + str(stock_list) + '\n')
    f.write('Seed: ' + str(seed) + '\n')
    f.write('Simulations: ' + str(sims) + '\n')
    f.write('Starting date range: ' + start1 + ' - ' + end1 + '\n')
    f.write('Ending date range: ' + start2 + ' - ' + end2 + '\n')
    f.write('------------------------------------------------------' + '\n')
    for node in dual.nodes:
        avg_return, spy_beat_count = po.run_simulations(sims, seed, node, start1, end1, start2, end2)
        f.write('Portfolio: ' + str(node) + '\n')
        f.write('Expected portfolio return: ' + str(avg_return) + '\n')
        f.write('Number of times SPX was beat: ' + str(spy_beat_count) + '\n')
        f.write('------------------------------------------------------' + '\n')

        
    
    stock_frequency = {}
    for s1 in stock_list:
        stock_frequency[s1] = 0
        for node in dual.nodes:
            for s2 in node:
                if s1 == s2:
                    stock_frequency[s1] += 1
    f.write('Stock frequencies: ' + str(stock_frequency) + '\n')
    f.write('------------------------------------------------------')
















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
plt.tight_layout()
plt.show()
'''