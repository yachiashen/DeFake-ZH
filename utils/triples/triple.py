
from .. import text_sim

class Edge:
    def __init__(self, subj:"Node", verb: str, obj:"Node", meta: list[tuple[str]]):
        self.subj = subj
        self.verb = verb
        self.obj = obj
        self.meta = meta
        if subj.name == 'None':
            self.vector = text_sim.get_text_vector(f"{verb}{obj}")
        elif obj.name == 'None':
            self.vector = text_sim.get_text_vector(f"{subj}{verb}")
        else:
            self.vector = text_sim.get_text_vector(f"{subj}{verb}{obj}")
    
    def __repr__(self) -> str:
        return f"Edge({self.subj.name} -[{self.verb}]-> {self.obj.name})"

class Node:
    def __init__(self, name: str):
        self.name = name
        # relations = [(is_subj: bool, edge: Edge)]
        self.relations = []
        self.vector = text_sim.get_text_vector(name)
    
    def insert_relation(self, edge: "Edge", is_subj: bool):
        self.relations.append((is_subj, edge.verb))
        return 
    
    def __repr__(self) -> str:
        return f"Node({self.name})"
    
class Graph:
    def __init__(self):
        self.nodes = dict()
        self.edges = [] 
        self.relations = [] # list[Edge]
    
    def __len__(self) -> None:
        return len(self.edges)
    
    def insert(self, subj_name: str, obj_name: str, verb: str, meta: list[tuple[str]] = None) -> None:
        if subj_name is None:
            subj_name = 'None'
        if obj_name is None:
            obj_name = 'None'

        search_subj = self.search_node(subj_name)
        search_obj = self.search_node(obj_name)
        
        if search_subj is None:
            self.nodes[subj_name] = Node(subj_name)
            search_subj = self.nodes[subj_name]

        if search_obj is None:
            self.nodes[obj_name] = Node(obj_name)
            search_obj = self.nodes[obj_name]

        subj_node = search_subj
        obj_node = search_obj

        edge = Edge(subj_node, verb, obj_node, meta)
        self.edges.append(edge)

        subj_node.relations.append((True, edge))
        obj_node.relations.append((True, edge)) 

    def search_node(self, target_node_name: str) -> Node:

        if target_node_name is None: return None
        target_vec = text_sim.get_text_vector(target_node_name)
        for node in self.nodes:
            if text_sim.get_vec_sim_value([self.nodes[node].vector, target_vec]) >= 0.8:
                return self.nodes[node]
        return None
    
    def merge_graph(self, other_graph):
        for edge in other_graph.edges:
            self.insert(subj_name = edge.subj.name, obj_name = edge.obj.name, verb = edge.verb, meta = edge.meta)
        return 

    def compare_graphs(self, other_graph):
        sim_edges = []
        for edge in self.edges:
            for other_edge in other_graph.edges:
                if text_sim.get_vec_sim_value([edge.vector, other_edge.vector]) >= 0.8:
                    sim_edges.append((str(edge), str(other_edge)))
                    break

        return sim_edges

    def display_graph(self) -> None:
        for edge in self.edges:
            print(f"{edge.subj.name} -[{edge.verb}]-> {edge.obj.name}")