import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 
import pandas as pd
import heapq
import itertools
import os
from copy import deepcopy


def get_locations(path):
    """
    Read the file and return a list of all locations
    """
    with open(path, 'r') as file:
        city_names = []
        for i in file.readlines():
            city_names.append(i.replace("\n",""))
    return city_names


def get_distance_matrix(dist_path):
    """
    Read the file and return a list of lists or a 2D array with
    distances such that dist_matrix[loc1][loc2] returns the distance from loc1 to loc2
    e.g. dist_matrix = [[0,4,9],
                        [4,0,8],
                        [9,8,0]] for 3  sample locations
    """
    with open(dist_path,'r') as file:
        dist_matrix = []
        for line in file.readlines():
            dist_matrix .append([int(i) for i in line.replace("\n","").split(",")])
    return dist_matrix

class Node:
    def __init__(self,name):
        self.name = name
        self.is_start = False
        self.is_visited = False
    
    def get_location_details(self):
        """
        Return a tuple representing the location with all its
        properties
        """
        return self.name, self.is_start, self.is_visited


def dist_locations(from_node,to_node):
    from_ = location_names.index(from_node.name)
    to_ = location_names.index(to_node.name)
    return dist_matrix[from_][to_]


class StateSpace:
    """Abstract class for defining State spaces for our search"""
    n = 0 # Counter for number of states, will also act as identifier for 
            # different states
    
    def __init__(self, action, gval, parent):
        """
           self.action === the name of the action used to generate
            this state from parent. If it is the initial state we will
            supply the action name "START"
           self.gval === a number (integer or real) that is the cost
            of getting to this state.
           self.parent === parent the state from which this state was generated (by
              applying "action"
        """
        self.action = action
        self.gval = gval
        self.parent = parent
        self.index = StateSpace.n
        StateSpace.n = StateSpace.n + 1

    def print_path(self):
        """print the sequence of actions used to reach self"""
        #can be over ridden to print problem specific information
        s = self
        states = []
        while s:
            states.append(s)
            s = s.parent
        states.pop().print_state()
        while states:
            print(" ==> ", end="")
            states.pop().print_state()
        print("")
 
    def has_path_cycle(self):
        """Returns true if self is equal to a prior state on its path"""
        s = self.parent
        hc = self.hashable_state()
        while s:
            if s.hashable_state() == hc:
                return True
            s = s.parent
        return False



class tsp(StateSpace):
    def __init__(self, current_location, locations, action, gval, parent=None):
        StateSpace.__init__(self, action, gval, parent)
        self.locations = locations    # locations = [node1, node2, ... nodek]
        self.current_location = current_location


    def successors(self):
        """Return list of tsp objects that are the
        successors of the current object"""
        States = []
        current_location = self.current_location
        unvisited_locations = self.get_unvisited()

        
        for location in unvisited_locations:
            new_gval = self.gval + dist_locations(current_location, location)
            new_locations = deepcopy(self.locations)
            current_index = self.locations.index(location)
            new_locations[current_index].is_visited = True
            States.append(tsp(new_locations[current_index], new_locations, 
                'Move to {}'.format(location.name), new_gval, self))
            
        if not unvisited_locations:
            # move from current location back to start
            start_location = self.get_start()
            new_gval = self.gval + dist_locations(current_location, start_location)
            new_locations = deepcopy(self.locations)
            current_index = self.locations.index(start_location)
            new_locations[current_index].is_visited = True
            States.append(tsp(new_locations[current_index], new_locations, 
                'Move to {}'.format(start_location.name), new_gval, self))            
        return States
    
    def hashable_state(self):
        """
        Return a data item that can be used as a dictionary key to UNIQUELY represent
         the state.
        """
        hash_list = []
        for location in self.locations:
            hash_list.append(location.get_location_details())
            
        hash_list.insert(0, self.current_location.get_location_details())
        
        return tuple(sorted(hash_list))
    
    def print_state(self):
        """
        Helper function to print the state
        """
        if self.parent:
            print("Action= \"{}\", S{}, g-value = {}, (From S{})".format(self.action, self.index, self.gval, self.parent.index))
        else:
            print("Action= \"{}\", S{}, g-value = {}, (Initial State)".format(self.action, self.index, self.gval))
                   
        for location in self.locations:
            print ('Name={}\tVisited={}\tStart={}\tCurrent={}'.format(location.name, location.is_visited, location.is_start,location == self.current_location))        

        print ('')


    def get_unvisited(self): # returns list of unvisited locations
        return [location for location in self.locations if not location.is_visited]
    
    def get_start(self): # returns Node of start location
        for location in self.locations:
            if location.is_start:
                return location


def tsp_goal_fn(state_now):
    """
    All states should be visited and our current location should be
    the starting location so as to complete a tour
    """
    return (not state_now.get_unvisited()) and state_now.current_location.is_start


def make_init_state(locations,start_location = 1):
    """
    locations = list of locations. 
    Returns a tsp object with initialized state
    """
    i=1
    all_locations = []
    for location in locations:
        new_location = Node(location)
        
        if i==start_location:
            new_location.is_start = True
            all_locations.insert(0,new_location)
        else:
            all_locations.append(new_location)
        i = i+1
    current_location = all_locations[0]
    current_location.is_visited = True
    
    return tsp(current_location, all_locations, "START", 0)


def heur_zero(state_now):
    """Zero Heuristic use to make A* search perform uniform cost search"""
    return 0


def heur_Euclidean(state_now):
    """
    The MIN from all unvisited locations {
       The euclidean distance to the unvisited location from the current location +
       The euclidean distance from that unvisited location back to the start location }
    """
    current_location = state_now.current_location
    d1 = [dist_locations(current_location, location) for location in state_now.get_unvisited()]
    
    start_location = state_now.get_start()
    d2 = [dist_locations(start_location, location) for location in state_now.get_unvisited()]
    
    if not d1 and not d2:
        return 0
    else:
        return min(d1) + min(d2)


def heur_MST_Euclidean(state_now):
    """
    Estimated Euclidean distance to travel all the unvisited nodes
       starting from the current location + heur_Euclidean.
    """
    return MST(state_now, dist_locations) + heur_Euclidean(state_now)



def MST(state_now, func):
    """
    We will use Kruskal's algorithm. But we could also use Prim's if we wanted to.
        1. Sort the edges of G in ascending (non-decreasing) order
        2. Return the edge costs of the Minimum Spanning Tree
    """
    unvisited_locations = state_now.get_unvisited()
    
    location_pairs = itertools.combinations(unvisited_locations, 2)
    
    edges = []
    for x,y in location_pairs:
        edges.append((func(x, y), (x,y)))
    edges = sorted(edges, key=lambda x: x[0])
    
    G = {}
    for c in unvisited_locations:
        G[c.name] = {}
        
    for e in edges:
        location1 = e[1][0].name
        location2 = e[1][1].name
        edge = e[0]
        G[location1][location2] = edge
        G[location2][location1] = edge

    return MinimumSpanningCost(G)


class UnionFind:
    """
    Union-find data structure.
    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:
    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.
    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.
    """

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root
        
    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r],r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def MinimumSpanningCost(g):
    """
    Return the minimum spanning tree of an undirected graph G.
    G should be represented in such a way that iter(G) lists its
    vertices, iter(G[u]) lists the neighbors of u, G[u][v] gives the
    length of edge u,v, and G[u][v] should always equal G[v][u].
    The tree is returned as a list of edges.
    """
    if not isUndirected(g):
        raise ValueError("MinimumSpanningTree: input is not undirected")
    for u in g:
        for v in g[u]:
            if g[u][v] != g[v][u]:
                raise ValueError("MinimumSpanningTree: asymmetric weights")

    # Kruskal's algorithm: sort edges by weight, and add them one at a time.
    # We use Kruskal's algorithm, first because it is very simple to
    # implement once UnionFind exists, and second, because the only slow
    # part (the sort) is sped up by being built in to Python.
    subtrees = UnionFind()
    #tree = []
    cost = 0
    for W,u,v in sorted((g[u][v],u,v) for u in g for v in g[u]):
        if subtrees[u] != subtrees[v]:
            cost += g[u][v]
            subtrees.union(u,v)

    return cost


def isUndirected(g):
    """Check that G represents a simple undirected graph."""
    for v in g:
        if v in g[v]:
            return False
        for w in g[v]:
            if v not in g[w]:
                return False
    return True


_SUM_HG = 0

class sNode:
    """Object of this class form the nodes of the search space.  Each
    node consists of a search space object (determined by the problem
    definition) along with the h and g values (the g values is
    redundant as it is stored in the state, but we make a copy in the
    node object for convenience), and a the number of the node"""
    
    n = 0
    lt_type = _SUM_HG
    
    def __init__(self, state, hval):
        self.state = state
        self.hval = hval
        self.gval = state.gval
        self.index = sNode.n
        sNode.n = sNode.n + 1

    def __lt__(self, other):
        """ For the f-value
           we wish to break ties by letting node1 < node2 if they both
           have identical f-values but if node1 has a SMALLER g
           value. This means that we expand nodes along shorter paths
           first causing the search to proceed directly to the goal"""
        
        if sNode.lt_type == _SUM_HG:
            if (self.gval+self.hval) == (other.gval+other.hval):
                #break ties by smallest gval. 
                return self.gval < other.gval
            else: return (self.gval + self.hval) < (other.gval + other.hval)
        if sNode.lt_type == _G:
            return self.gval < other.gval
        if sNode.lt_type == _H:
            return self.hval < other.hval
        print('sNode class has invalid comparator setting!')
        return self.gval < other.gval


class Open:
    """Open objects hold the search frontier---the set of unexpanded
       nodes. Depending on the search strategy used we want to extract
       nodes from this set in different orders, so set up the object's
       functions to operate as needed by the particular search
       strategy"""
    
    def __init__(self):
        #use priority queue for OPEN (first out is node with
        #lowest fval = gval+hval)
        self.open = []
        #set node less than function to compare sums of hval and gval
        sNode.lt_type = _SUM_HG
        self.insert = lambda node: heapq.heappush(self.open, node)
        self.extract = lambda: heapq.heappop(self.open)
            

    def empty(self): 
        return not self.open

    def print_open(self):
        print("{", end="")
        if len(self.open) == 1: 
            print("OPEN   <S{}:{}:{}, g={}, h={}, f=g+h={}>".format(self.open[0].state.index, self.open[0].state.action, self.open[0].state.hashable_state(), self.open[0].gval, self.open[0].hval, self.open[0].gval+self.open[0].hval), end="")
        else:
            for nd in self.open:
                print("OPEN   <S{}:{}:{}, g={}, h={}, f=g+h={}>".format(nd.state.index, nd.state.action, nd.state.hashable_state(), nd.gval, nd.hval, nd.gval+nd.hval), end="")
        print("}")

    def clear(self): 
        self.open = []


def _zero_hfn(state):
    """Null heuristic (zero)"""
    return 0


class IDASearchEngine:
    def __init__(self):
        self.trace = 0

    def initStats(self):
        sNode.n = 0
        StateSpace.n = 1    #initial state already generated on call so search
        self.total_search_time = 0
        self.cycle_check_pruned = 0
        self.total_search_time = os.times()[0]

    def trace_on(self, level = 1):
        """For debugging, set tracking level 1 or 2"""
        self.trace = level

    def trace_off(self):
        """Turn off tracing"""
        self.trace = 0

    def get_strategy(self):
        rval = 'ida* with full cycle checking'
        return rval

    def search(self, initState, goal_fn, ss, heur_fn=_zero_hfn, LIMIT=1):
        #Perform full cycle checking as follows
        #a. check state before inserting into OPEN. If we had already reached
        #   the same state via a cheaper path, don't insert into OPEN.
        #b. Sometimes we find a new cheaper path to a state (after the older
        #   more expensive path to the state has already been inserted.
        #   We deal with this lazily. We check states extracted from OPEN
        #   and if we have already expanded that state via a cheaper path
        #   we don't expand it. If we had expanded the state via a more
        #   expensive path, we re-expand it.
     
        ### Step1: Call the function to initialize the Search Stats (1 line)
        self.initStats()

        #BEGIN TRACING
        if self.trace:
            print("   TRACE: Search Strategy: ", self.get_strategy())
            print("   TRACE: Initial State:", end="")
            initState.print_state()
        #END TRACING

        # Step 2: Instantiate an object of Open class (1 line)
        OPEN = Open()

        # Step 3: Create the first node in search space using the argument passed as initState and value of
        # heuristic function for that initState (1-3 lines)
        first = sNode(initState,heur_fn(initState))

        # Step 4: Create a cycle check dictionary. This will store the cheapest path (g-val) found
        #so far to a state. (1 lines)
        self.cc_dictionary =dict()

        # Step 5: Store the first state as a hashable as key and it's gval as value(1-2 lines)
        self.cc_dictionary[initState.hashable_state()] = initState.gval

        # Step 6: insert the search space node in object of Open data structure using the insert function(1 line)
        OPEN.insert(first)

        ###Step 7: NOW perform the search by calling searchOpen and return the result as Node in search space.
            #(1 line)
        goal_node = self.searchOpen(OPEN, goal_fn, heur_fn, ss, LIMIT, initState)# Insert code here


        if goal_node:
            print("Search Successful!")
            print("   Strategy = '{}, Depth Level = {}'".format(self.get_strategy(), LIMIT))
            print("   Solution cost = {}".format(round(goal_node.gval)))

            goal_node.state.print_path()
            self.total_search_time = os.times()[0] - self.total_search_time
            print("Search time = {}, nodes expanded = {}, states generated = {}, states cycle check pruned = {}".format(self.total_search_time,sNode.n, StateSpace.n, self.cycle_check_pruned))
            return goal_node.state
        else:
            if LIMIT < len(initState.locations):
                # Insert code here (1 line). What happens to the limit if current limit
                #does not find a solution in IDA* ?
                LIMIT += heur_fn(initState)
                return self.search(initState, goal_fn, ss, heur_fn, LIMIT)

        #exited the while without finding goal---search failed

            print("Search Failed! (strategy '{}') No solution found".format(self.get_strategy()))
            self.total_search_time = os.times()[0] - self.total_search_time
            print("Search time = {}, nodes expanded = {}, states generated = {}, states cycle check pruned = {}".format(self.total_search_time,sNode.n, StateSpace.n, self.cycle_check_pruned))
            return False

    def searchOpen(self, OPEN, goal_fn, heur_fn, ss, LIMIT, initState):
        """Open has some nodes on it, now search from that state of OPEN"""

        #BEGIN TRACING
        if self.trace:
            print("   TRACE: Initial OPEN: ", OPEN.print_open())

            print("   TRACE: Initial CC_Dict:", self.cc_dictionary)
        #END TRACING

        DEPTH = 1       # USED FOR IDA_STAR
        while not OPEN.empty():
            # Extract the node in OPEN using function defined in init and assign it to a
            #  variable called node. (1 line)
            node = OPEN.extract()


            #BEGIN TRACING
            if self.trace:
                print("   TRACE: Next State to expand: <S{}:{}:{}, g={}, h={}, f=g+h={}>".format(node.state.index, node.state.action, node.state.hashable_state(), node.gval, node.hval, node.gval+node.hval))
                if node.state.gval != node.gval:
                    print("ERROR: Node gval not equal to state gval!")
            #END TRACING
            #Check if current state of node is the goal state using previously defined function to
            # check goal.


            if goal_fn(node.state):
                #Insert 1 line code here:
                #node at front of OPEN is a goal...search is completed.
                return node


             #All states reached by a search node on OPEN have already
             #been hashed into the self.cc_dictionary. However,
             #before expanding a node we might have already expanded
             #an equivalent state with lower g-value. So only expand
             #the node if the hashed g-value is no greater than the
             #node's current g-value. 

            #BEGIN TRACING
            if self.trace:
                print("   TRACE: CC_dict gval={}, node.gval={}".format(self.cc_dictionary[node.state.hashable_state()], node.gval))
            #END TRACING

            if self.cc_dictionary[node.state.hashable_state()] < node.gval:
                continue

            successors = [] 
            # LIMIT THE DEPTH FOR ITERATIVE DEEPENING A_STAR
            if LIMIT == DEPTH:
                pass
            else:
                # Get successors from current state of the node.(1 line)
                successors = node.state.successors()

                  
            #Increment depth by 1. (1 line)
            DEPTH += 1

            #BEGIN TRACING
            if self.trace:
                print("   TRACE: Expanding Node. Successors = {", end="")
                for ss in successors:
                    print("<S{}:{}:{}, g={}, h={}, f=g+h={}>, ".format(ss.index, ss.action, ss.hashable_state(), ss.gval, heur_fn(ss), ss.gval+heur_fn(ss)), end="")
                print("}")
            #END TRACING


            for succ in successors:
                # Get a hashable state for each successor. (1 line)
                hash_state = succ.hashable_state()

                #BEGIN TRACING
                if self.trace > 1:
                    print("   TRACE: Successor State:", end="")
                    print("<S{}:{}:{}, g={}, h={}, f=g+h={}>, ".format(succ.index, succ.action, succ.hashable_state(), succ.gval, heur_fn(succ), succ.gval+heur_fn(succ)), end="")
                    if hash_state in self.cc_dictionary:
                        print("   TRACE: Already in CC_dict, CC_dict gval={}, successor state gval={}".format(self.cc_dictionary[hash_state], succ.gval))
                #END TRACING
                # Check for conditions if we want to remove this succ. i.e. same state should not already
                # exist in the cc_dictionary and cost of reaching this succ is costlier than one in cc
                # (1-2 lines)

                prune_succ = (hash_state in self.cc_dictionary) and (succ.gval > self.cc_dictionary[hash_state])


                if prune_succ:
                    # Increment counter for pruned succ (1 line)
                    self.cycle_check_pruned += 1

                    #BEGIN TRACING
                    if self.trace > 1:
                        print(" TRACE: Successor State pruned by cycle checking")
                    #END TRACING
                    continue

                #passed all cycle checks...add to open
                # i.e. New state reached, create a new search space node and add it to OPEN for
                # extracting it in next iteration. (1 -2 lines)
                current = sNode(succ,heur_fn(succ))
                OPEN.insert(current)


                #BEGIN TRACING
                if self.trace > 1:
                    print(" TRACE: Successor State added to OPEN")
                #END TRACING

                #record cost of this path in cc dictionary.(1 line)
                self.cc_dictionary[hash_state] = succ.gval

        #end of while--OPEN is empty and no solution
        return False


def draw_final_path(state):
    
    if not state:
        return None

    
    states = []
    while state:
        states.append(state)
        state = state.parent

        
    locations = [state.current_location for state in states]
    locations.reverse()
    
    for i, location in enumerate(locations):
        if i != len(locations)-1:
            print(location.name+ " ==> ",end='')
        else:
            print(location.name)
    return locations



if __name__ == '__main__':
    dist_filepath = "distances.csv" #Enter path to distances.csv file
    loc_filepath = "locations.txt" # Enter path to locations.txt file

    #Get distance matrix and location list
    dist_matrix = get_distance_matrix(dist_filepath)
    location_names = get_locations(loc_filepath)

    se = IDASearchEngine()
    state = make_init_state(location_names,1)
    ss = 1
    final = se.search(state, tsp_goal_fn, ss,eval('heur_MST_Euclidean'))
    locations = draw_final_path(final)
    direction = []
    for i in locations:
        direction.append(i.name.replace(" ","").lower())

with open('location_coordinates.csv','r') as file:
    coordinate = pd.DataFrame()
    header = file.readline()
    f = [i.replace("\",",";") for i in  file.readlines()]

coordinate[header.split(",")[0]]= [i.split(";")[0].replace("\"","").replace(" ","").lower() for i in f]
longitude = [i.split(";")[1].replace("\"","").split(",")[0].replace("¡","") for i in f]
latitude = [i.split(";")[1].replace("\"","").split(",")[1].replace("\n","").replace("¡","") for i in f]


for i in range(len(longitude)):
    if longitude[i].replace(" ",",").split(",")[1] == "N":
        longitude[i] = float(longitude[i].replace(" ",",").split(",")[0])
    else:
        longitude[i] = float("-" + longitude[i].replace(" ",",").split(",")[0])

for i in range(len(latitude )):
    if latitude [i].replace(" ",",").split(",")[1] == "E":
        latitude [i] = float(latitude [i].replace(" ",",").split(",")[0])
    else:
        latitude [i] = float("-" + latitude [i].replace(" ",",").split(",")[0])
coordinate[header.split(",")[1]] = longitude
coordinate[header.split(",")[2].replace("\n","")] = latitude

landscape = [ dict(
        type = 'scattergeo',
        lon = coordinate['Longitude'],
        lat = coordinate['Latitude'],
        text = coordinate['Location'],
        mode = 'markers',
        marker = dict( 
            size=2, 
            color='rgb(255, 0, 0)',
            line = dict(
                width=3,
                color='rgba(68, 68, 68, 0)'
            )

        ))]



coordinate.set_index('Location', inplace=True)


flight_paths = []
for i in range( len( coordinate )):
    flight_paths.append(
        dict(
            type = 'scattergeo',
            lon = [ coordinate.loc[direction[i]]['Longitude'],coordinate.loc[direction[i+1]]['Longitude']],
            lat = [  coordinate.loc[direction[i]]['Latitude'],coordinate.loc[direction[i+1]]['Latitude']],
            mode = 'lines',
            line = dict(
                width = 1,
                color = 'red',
            )
        )
    )


layout = dict(
        title = 'Flight path',
        showlegend = False, 
        geo = dict(
            projection=dict( type='azimuthal equal area' ),
            showland = True,
            landcolor = 'rgb(243, 243, 243)',
            countrycolor = 'rgb(204, 204, 204)',
        )
    )


choromap = go.Figure(data = flight_paths + landscape, layout = layout)
plot( choromap,validate=False)
