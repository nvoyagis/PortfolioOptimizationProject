import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import cvxpy
import sklearn


def matlab_to_python_weights(s: str, stocks: list) -> any:
    copies = []
    l = s.split()

    g = nx.Graph()
    for stock in stocks:
        g.add_node(stock)

    for i in range(len(l)):
        if i % 2 == 0 and i < len(l) - 1:
            tpl = (int(l[i].split(',')[0][1:]), int(l[i].split(',')[1][:-1]))
            if tpl not in copies:
                copies.append((tpl[1], tpl[0]))
                g.add_edge(stocks[tpl[0] - 1], stocks[tpl[1] - 1])
                g[stocks[tpl[0] - 1]][stocks[tpl[1] - 1]]['weight'] = l[i+1]

    return g

def matlab_to_python_faces(s: str, stocks: list) -> any:
    l = []
    trios = s.split()
    for i in range(len(trios)):
        if i % 3 == 2:
            l.append((stocks[int(trios[i-2])-1], stocks[int(trios[i-1])-1], stocks[int(trios[i])-1]))
    return l

faces_2 = '''
     1     9     5
     9     3     8
     1     9     6
     1     3    13
     1     3     7
     9     3     2
     1     2    14
     9     2     4
     1     4    17
     9     4    17
     1    17    18
     9    17    18
     1    18    20
     9    18    20
     9     6     8
     3     6     8
     1     6    12
     3     6    19
     1    12    13
     3    12    13
     1     2     7
     3     2     7
     1    20    15
     9    20    15
     1    15    16
     9    15     5
     3    12    19
     6    12    19
     1     5    16
    15     5    11
    15    16    11
     5    16    11
     1     4    10
     2     4    14
     1    14    10
     4    14    10
    '''



A = '''
   (2,1)        4
   (3,1)        4
   (4,1)        4
   (5,1)        3
   (6,1)        4
   (7,1)        4
   (9,1)        4
  (10,1)        1
  (12,1)        3
  (13,1)        3
  (14,1)        2
  (15,1)        3
  (16,1)        2
  (17,1)        4
  (18,1)        4
  (20,1)        4
   (1,2)        4
   (3,2)        4
   (4,2)        4
   (7,2)        4
   (9,2)        4
  (14,2)        2
   (1,3)        4
   (2,3)        4
   (6,3)        4
   (7,3)        3
   (8,3)        4
   (9,3)        4
  (12,3)        4
  (13,3)        4
  (19,3)        3
   (1,4)        4
   (2,4)        4
   (9,4)        4
  (10,4)        2
  (14,4)        2
  (17,4)        4
   (1,5)        3
   (9,5)        3
  (11,5)        3
  (15,5)        3
  (16,5)        3
   (1,6)        4
   (3,6)        4
   (8,6)        4
   (9,6)        4
  (12,6)        4
  (19,6)        3
   (1,7)        4
   (2,7)        4
   (3,7)        3
   (3,8)        4
   (6,8)        4
   (9,8)        4
   (1,9)        4
   (2,9)        4
   (3,9)        4
   (4,9)        4
   (5,9)        3
   (6,9)        4
   (8,9)        4
  (15,9)        3
  (17,9)        4
  (18,9)        4
  (20,9)        4
   (1,10)       1
   (4,10)       2
  (14,10)       3
   (5,11)       3
  (15,11)       3
  (16,11)       3
   (1,12)       3
   (3,12)       4
   (6,12)       4
  (13,12)       4
  (19,12)       3
   (1,13)       3
   (3,13)       4
  (12,13)       4
   (1,14)       2
   (2,14)       2
   (4,14)       2
  (10,14)       3
   (1,15)       3
   (5,15)       3
   (9,15)       3
  (11,15)       3
  (16,15)       4
  (20,15)       3
   (1,16)       2
   (5,16)       3
  (11,16)       3
  (15,16)       4
   (1,17)       4
   (4,17)       4
   (9,17)       4
  (18,17)       4
   (1,18)       4
   (9,18)       4
  (17,18)       4
  (20,18)       4
   (3,19)       3
   (6,19)       3
  (12,19)       3
   (1,20)       4
   (9,20)       4
  (15,20)       3
  (18,20)       4
  '''

'''
stock_list = ['DIS', 'KO', 'ADBE', 'MRK', 'KMI', 'AAPL', 'JNJ', 'CVS', 'COST', 'T', 'BA', 'EA', 'HAS', 'HD', 'HSY', 'LLY', 'NFLX', 'NKE', 'V', 'JPM']
stock_list.sort()

print(matlab_to_python_faces(faces_2, stock_list))

g = matlab_to_python_weights(A, stock_list)
print('__________')

pos0 = nx.planar_layout(g)
nx.draw_networkx_nodes(g, pos0, node_color='lightblue', node_size=650)
nx.draw_networkx_labels(g, pos0, font_size=8, font_family='sans-serif')
nx.draw_networkx_edges(g, pos0, width=2)
plt.figure(1,figsize=(12,12)) 
plt.show()
'''



# dual = dual.build_dual_from_faces(g)

# # Plotting both graphs
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# # Original
# pos = nx.planar_layout(g)
# nx.draw(g, pos, with_labels=True, ax=axs[0])
# axs[0].set_title("Original Planar Graph")

# # Dual
# dual_pos = nx.spring_layout(dual)
# nx.draw(dual, dual_pos, with_labels=True, node_color="lightcoral", ax=axs[1])
# axs[1].set_title("Dual Graph")

# plt.show()