from dataclasses import dataclass
from typing import TypeAlias, Dict, Union, List
from stack_array import * # Needed for Depth First Search
from queue_array import * # Needed for Breadth First Search

MaybeVertex: TypeAlias = Union[None, 'Vertex']

@dataclass
class Vertex:
    id: Any

    def __post_init__(self) -> None:
        '''Add other attributes as necessary'''
        self.adjacent_to: List = []
        self.visited = False
        self.color = ""

class Graph:
    '''Add additional helper methods if necessary.'''
    def __init__(self, filename: str):
        '''reads in the specification of a graph and creates a graph using an adjacency list representation.  
           You may assume the graph is not empty and is a correct specification.  E.g. each edge is 
           represented by a pair of vertices.  Note that the graph is not directed so each edge specified 
           in the input file should appear on the adjacency list of each vertex of the two vertices associated 
           with the edge.'''
        # opens file, creates an empty list to store vertices, and constructs a graph
        self.f = open(filename, "r")
        self.verts = []
        self.weights = {}
        self.graph = self.make_graph()

        # closes file
        self.f.close()

    def make_graph(self) -> Dict:
        '''constructs a graph with the given vertex pairs from a text file'''
        # creates empty dictionary to store information
        adjlist = {}
        # separates each edge
        for edge in self.f.readlines():
            # separates the two vertices that make an edge
            edgelist = edge.split(" ")
            v1 = Vertex(edgelist[0])
            if edgelist[1][len(edgelist[1]) - 1] == "\n":
                v2 = Vertex(edgelist[1][0: len(edgelist[1]) - 1])
            else:
                v2 = Vertex(edgelist[1])
            # adds the two vertices that form a given edge. if the vertices are already in the vertex list, they will not be added (this is checked in the add_vertex function)
            self.add_vertex(v1)
            self.add_vertex(v2)
            self.add_edge(v1.id, v2.id)
            # Update weights
            if edgelist[2][len(edgelist[2]) - 1] == "\n":
                self.weights.update({(v1.id, v2.id): float(edgelist[2][0: len(edgelist[2]) - 1])})
                self.weights.update({(v2.id, v1.id): float(edgelist[2][0: len(edgelist[2]) - 1])})
            else:
                self.weights.update({(v1.id, v2.id): float(edgelist[2])})
                self.weights.update({(v2.id, v1.id): float(edgelist[2])})

        # updates the dictionary using the list returned by the get_vertices() function to make sure that each vertex is included once in the correct order
        l = self.get_vertices()
        for vertex_id in l:
            for vertex in self.verts:
                if vertex.id == vertex_id:
                    adjlist.update({vertex.id: vertex.adjacent_to})
        # returns the graph information stored in a dictionary
        return adjlist

    def add_vertex(self, key: Any) -> None:
        '''Add vertex to graph, only if the vertex is not already in the graph.'''
        if self.get_vertex(key.id) is None:
            self.verts.append(key)

    def get_vertex(self, key: Any) -> MaybeVertex:
        '''Return the Vertex object associated with the id. If id is not in the graph, return None'''
        for vert in self.verts:
            if key == vert.id:
                return vert
        return None

    def add_edge(self, v1: Any, v2: Any) -> None:
        '''v1 and v2 are vertex id's. As this is an undirected graph, add an 
           edge from v1 to v2 and an edge from v2 to v1.  You can assume that
           v1 and v2 are already in the graph'''
        # creates 2 edges with the same vertices. each vertex will have an adjacent edge that begins with that vertex
        edge1 = str(v1) + " " + str(v2)
        self.get_vertex(v1).adjacent_to.append(edge1)
        edge2 = str(v2) + " " + str(v1)
        self.get_vertex(v2).adjacent_to.append(edge2)

    def remove_edge(self, v1: str, v2: str) -> None:
        adj1 = self.get_vertex(v1).adjacent_to
        adj2 = self.get_vertex(v2).adjacent_to
        if (v1 + " " + v2) in adj1:
            self.get_vertex(v1).adjacent_to.remove(v1 + " " + v2)
        if (v2 + " " + v1) in adj2:
            self.get_vertex(v2).adjacent_to.remove(v2 + " " + v1)


    def get_vertices(self) -> List:
        '''Returns a list of id's representing the vertices in the graph, in ascending order'''
        l = []
        for v in self.verts:
            l.append(str(v.id))
        l.sort()
        return l
    
    def get_adjacent_vertices(self, v: str) -> List:
        '''Returns a list of id's representing the vertices adjacent to v, in any order'''
        adjacency_list = self.get_vertex(v).adjacent_to
        l = []
        for edge in adjacency_list:
            l.append(edge.split(" ")[1])
        return l
    
    def get_weights(self) -> List:
        '''Returns the weights of all edges connecting two vertices in ascending order'''
        return self.weights
    
    def get_edge_weight(self, v1: str, v2: str) -> float:
        '''Returns the weight of the edge connecting two vertices'''
        if self.weights.get((v1, v2)) is not None:
            return self.weights.get((v1, v2))

    def conn_components(self) -> List:
        '''Returns a list of lists.  For example, if there are three connected components 
           then you will return a list of three lists.  Each sub list should contain the
           vertices (in 'Python List Sort' order) in the connected component represented by that list.
           The overall list of lists should also be in order based on the first item of each sublist.
           This method MUST use Depth First Search logic!'''
        # creates a stack that will fit all the vertices and an empty list to be used later
        vertstack = Stack(len(self.graph))
        master_list = []

        # gets all the vertex id's from the get_vertices function
        for vert_id in self.get_vertices():
            # gets the vertex that correspond with the given id
            real_vert = self.get_vertex(vert_id)
            # checks if the vertex has not been accessed yet
            if real_vert.visited is False:
                # pushes vertex into a stack and changes the visited boolean to True. also begins making a smaller list to contain a single component of the graph
                vertstack.push(real_vert)
                real_vert.visited = True
                sub_list = [vert_id]
                # adds vertices that are both adjacent to the given vertex and not visited to the stack. a vertex is popped if it does not have more adjacent vertices that have not been visited yet
                while vertstack.num_items > 0:
                    adj_edge_list = real_vert.adjacent_to
                    for edge in adj_edge_list:
                        adj_vert_id = edge.split(" ")[1]
                        self.get_vertex(adj_vert_id)
                        if self.get_vertex(adj_vert_id).visited is False:
                            self.get_vertex(adj_vert_id).visited = True
                            sub_list.append(adj_vert_id)
                            vertstack.push(self.get_vertex(adj_vert_id))
                    real_vert = vertstack.pop()
                # sorts a component of the graph and appends it to a list
                sub_list.sort()
                master_list.append(sub_list)

        # sets all the .visited values to False so functions can continue to be used on the graph
        for vert_id in self.get_vertices():
            self.get_vertex(vert_id).visited = False

        # returns a list of all the components of the graph
        return master_list

    def is_bipartite(self) -> bool:
        '''Returns True if the graph is bicolorable and False otherwise.
        This method MUST use Breadth First Search logic!'''
        # creates a queue that will fit each vertex
        vert_queue = Queue(len(self.graph))
        # makes a list of all the components of the graph. sets bipartite as True and first_color as True (these may be changed later)
        l = self.conn_components()
        bipartite = True
        first_color = True
        # separates all the components of the graph
        for component in l:
            # enqueues unvisited vertices to a queue. the order of the vertices depends on which vertices are adjacent to the first vertices in the list
            for vert_id in component:
                # if the vertex has not been visited yet, enqueues the vertex and mark it as queued. then considers the adjacent vertices
                if self.get_vertex(vert_id).visited == False:
                    vert = self.get_vertex(vert_id)
                    vert_queue.enqueue(vert)
                    self.get_vertex(vert_id).visited = True
                    # creates the adjacent vertices from the list of adjacent edges
                    adj_edge_list = vert.adjacent_to
                    for edge in adj_edge_list:
                        adj_vert_id = edge.split(" ")[1]
                        adj_vert = self.get_vertex(adj_vert_id)
                        # checks if an adjacent vertex has been accessed yet. if not, then the vertex is enqueued and "visited" is changed to True
                        if adj_vert.visited is False:
                            self.get_vertex(adj_vert_id).visited = True
                            vert_queue.enqueue(adj_vert)
            # colors each vertex one of two colors. if a vertex must be assigned the same color as an adjacent vertex, then bipartite is set to False
            while vert_queue.num_items > 0:
                temp = vert_queue.dequeue()
                if first_color:
                    self.get_vertex(temp.id).color = "red"
                    first_color = False
                adj_edge_list2 = self.get_vertex(temp.id).adjacent_to
                for edge2 in adj_edge_list2:
                    if temp.color == "red" and self.get_vertex(edge2.split(" ")[1]).color == "":
                        self.get_vertex(edge2.split(" ")[1]).color = "blue"
                    elif temp.color == "blue" and self.get_vertex(edge2.split(" ")[1]).color == "":
                        self.get_vertex(edge2.split(" ")[1]).color = "red"
                    elif temp.color == self.get_vertex(edge2.split(" ")[1]).color and temp.color != "":
                        bipartite = False

        # sets all the .visited values to False so functions can continue to be used on the graph
        for vert_id in self.get_vertices():
            self.get_vertex(vert_id).visited = False
            self.get_vertex(vert_id).colored = False
            self.get_vertex(vert_id).color = ""

        # returns bipartite, which is set to False if the graph is not two-colorable or is kept at True if the graph is two-colorable
        return bipartite
    
