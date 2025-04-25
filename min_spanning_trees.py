from dataclasses import dataclass
from typing import TypeAlias, Dict, Union, List
from stack_array import * # Needed for Depth First Search
from queue_array import * # Needed for Breadth First Search
from graph import *

def prims_tree(g: Graph, filename: str) -> None:
    '''Prim's algorithm for minimum spanning tree'''
    vlist = [g.get_vertices()[0]] # Start enumeration list with v01
    with open(filename, "w", newline="") as f: # Make file for graph
        while vlist != g.get_vertices():
            remainingverts = g.get_vertices()
            for v in vlist: 
                remainingverts.remove(v)
            # Goal: choose edge with minimum weight
            if remainingverts != []:
                minweight = float('inf')
                minedge = None
                if minweight is not None:
                    # Get all edges and find the one with the minimum weight
                    for v1 in vlist:
                        for v2 in remainingverts:
                            if g.get_edge_weight(v1, v2) is not None and g.get_edge_weight(v1, v2) < minweight:
                                minweight = g.get_edge_weight(v1, v2)
                                minedge = (v1, v2)
                    print(str(minedge[0]) + " " + str(minedge[1]) + " " + str(minweight) + "\n")
                    f.write(str(minedge[0]) + " " + str(minedge[1]) + " " + str(minweight) + "\n")
                    vlist.append(minedge[1])
            else:
                return


def kruskals_tree(g: Graph, filename: str) -> None:
    n = len(g.get_vertices())
    verts = g.get_vertices()
    tree_edge_count = 0
    graph_edges = g.get_weights()
    with open(filename, "w", newline="") as f: # Make file for graph
        while tree_edge_count < n - 1:
            edges_and_weights = g.get_weights().items()
            # Choose edge with minimum weight
            weight = float('inf')
            for pair in edges_and_weights:
                if weight > pair[1]:
                    weight = pair[1]
                    edge = pair[0]
                    v1 = edge[0]
                    v2 = edge[1]
            print(g.get_vertex(v1).adjacent_to)
            print(g.get_vertex(v2).adjacent_to)
            if g.get_vertex(v1).adjacent_to.sort() != g.get_vertex(v2).adjacent_to.sort():
                
                #????????
                for v in g.get_vertices(v1):
                    g.get_vertices(v).update(g.get_vertices(v2))
                for v in g.get_vertices(v2):
                    g.get_vertices(v).update(g.get_vertices(v1))
                tree_edge_count += 1
                f.write(str(v1) + " " + str(v2) + " " + str(weight) + "\n")
                print(str(v1) + " " + str(v2) + " " + str(weight) + "\n")
            g.remove_edge(v1, v2)
            




# NOTE: does not produce a minimum spanning tree but rather a spanning tree such that each vertex is as close as possible to v01
def djikstras_tree(g: Graph, filename: str) -> None:
    verts = g.get_vertices()
    initial = verts[0]
    minweight = float('inf')
    tree_builder = {}
    minvert = None
    d = {}
    for v in verts:
        if v != verts[0]: d.update({v: float('inf')})
        else: d.update({v: 0})

    while verts != []:
        # Choose key of d with minimal value
        min_d = float('inf')
        for v in verts:
            if d.get(v) < min_d:
                min_d = d.get(v)
                min_d_vert = v
        if min_d_vert in verts:
            print('min: ' + str(min_d_vert))
            verts.remove(min_d_vert)

        # Choose minimal weight starting with the initial vertex
        for v in g.get_adjacent_vertices(min_d_vert):
            print('adj: ' + str(g.get_adjacent_vertices(min_d_vert)))
            delta = float(d.get(min_d_vert)) + float(g.get_edge_weight(min_d_vert, v))
            if delta < d.get(v):
                d.update({v: delta})
                tree_builder.update({v: min_d_vert})
                print((str(v), str(min_d_vert)))

    with open(filename, "w", newline="") as f: # Make file for graph
        for v in tree_builder:
            f.write(str(v) + " " + str(tree_builder.get(v)) + " " + str(g.get_edge_weight(v, tree_builder.get(v))) + "\n")
                



g = Graph('test8.txt')
djikstras_tree(g, 'test8_djikstras_tree.txt')