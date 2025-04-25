from dataclasses import dataclass
from typing import TypeAlias, Dict, Union, List
from stack_array import * # Needed for Depth First Search
from queue_array import * # Needed for Breadth First Search
from graph import *

def bfs_search(g: Graph) -> List:
    '''Breadth-first search of a tree, using v01 as first vertex'''
    traversed = []
    next = []
    vlist = [g.get_vertices()[0]] # Start enumeration list with v01
    while vlist != []:
        for v1 in vlist:
            edges = []
            adj = []
            traversed.append(v1)
            edges = g.get_vertex(v1).adjacent_to
            for e in edges:
                adj.append(e.split(" ")[1]) # adj: list of adjacent vertices
            for v2 in g.get_vertices():
                if (v2 not in traversed) and (v2 in adj):
                    next.append(v2)
        vlist = next
        next = []
    return traversed



def bfs_tree(g: Graph, filename: str) -> None:
    '''Breadth-first creation of a tree from a graph g, using v01 as first vertex'''
    f1 = open(filename, "w", newline='')
    traversed = []
    next = []
    vlist = [g.get_vertices()[0]] # Start enumeration list with v01
    while vlist != []:
        for v1 in vlist:
            edges = []
            adj = []
            traversed.append(v1)
            edges = g.get_vertex(v1).adjacent_to
            for e in edges:
                adj.append(e.split(" ")[1]) # adj: list of adjacent vertices
            for v2 in g.get_vertices():
                if (v2 not in traversed) and (v2 in adj):
                    next.append(v2)
                    f1.write(str(v1) + " " + str(v2) + " 0\n")
        vlist = next
        next = []
    f1.truncate()
    return f1




def dfs_search(g: Graph) -> List:
    '''Depth-first search of a tree, using v01 as first vertex'''
    vlist = [g.get_vertices()[0]] # Start enumeration list with v01
    search = dfs_search_helper(g, vlist[0], vlist)
    return search


def dfs_search_helper(g: Graph, cur: str, vlist: List) -> List:
    for v in g.get_vertices():
        adj = []
        edges = g.get_vertex(cur).adjacent_to
        for e in edges:
            adj.append(e.split(" ")[1]) # adj: list of adjacent vertices
        if (v not in vlist) and (v in adj):
            vlist.append(v)
            dfs_search_helper(g, v, vlist)
    return vlist





def dfs_tree(g: Graph, filename: str) -> None:
    '''Depth-first creation of a tree from a graph g, using v01 as first vertex'''
    vlist = [g.get_vertices()[0]] # Start enumeration list with v01
    with open(filename, "w", newline="") as f:
        dfs_tree_helper(g, vlist[0], vlist, f)

def dfs_tree_helper(g: Graph, cur: str, vlist: List, f) -> None:
    for v in g.get_vertices():
        adj = []
        edges = g.get_vertex(cur).adjacent_to
        for e in edges:
            adj.append(e.split(" ")[1]) # adj: list of adjacent vertices
        if (v not in vlist) and (v in adj):
            vlist.append(v)
            f.write(str(cur) + " " + str(v) + " 0\n")
            dfs_tree_helper(g, v, vlist, f)




#g = Graph('tree2.txt')
#print(dfs_search(g))
g = Graph('tree2.txt')
