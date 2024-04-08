import networkx as nx
import random

class Nominator:
    """Class responsible for nominating nodes for transactions.
    """    
    def __init__(self, g, degree_threshold):
        self.g = g
        self.remaining_count_dict = dict()
        self.used_count_dict = dict()
        self.model_params_dict = dict()
        
        self.type_candidates = dict()
        self.type_index = dict()

    def initialize_count(self, type, count, schedule_id, min_accounts, max_accounts, min_period, max_period):
        """Counts the number of nodes of a given type.

        Args:
            type (string): type of node to count
            count (int): number of types to count
        """        
        if type in self.remaining_count_dict:
            self.remaining_count_dict[type] += count
            param_list = [(schedule_id, min_accounts, max_accounts, min_period, max_period) for i in range(count)]
            self.model_params_dict[type] = param_list.append(param_list)
        else:
            self.remaining_count_dict[type] = count
            param_list = [(schedule_id, min_accounts, max_accounts, min_period, max_period) for i in range(count)]
            self.model_params_dict[type] = param_list
        self.used_count_dict[type] = 0
        
    
    def initialize_candidates(self):
        for type in self.remaining_count_dict:
            if type == 'fan_in':
                self.type_candidates[type] = self.get_fan_in_candidates()
            elif type == 'fan_out':
                self.type_candidates[type] = self.get_fan_out_candidates()
            elif type == 'forward':
                self.type_candidates[type] = self.get_forward_candidates()
            elif type == 'single':
                self.type_candidates[type] = self.get_single_candidates()
            elif type == 'mutual':
                if self.single_candidates is None:
                    self.type_candidates[type] = self.get_single_candidates()
                else:
                    self.mutual_candidates = self.single_candidates.copy()
            elif type == 'periodical':
                if self.single_candidates is None:
                    self.type_candidates[type] = self.get_single_candidates()
                else:
                    self.type_candidates[type] = self.single_candidates.copy()
            else:
                raise ValueError('Invalid type: {}'.format(type))
            
            self.type_index[type] = 0
    
    def get_fan_in_candidates(self):
        """Returns a list of node ids that have at least degree_threshold incoming edges.

        Returns:
            list: list of node ids that have at least degree_threshold incoming edges
        """        
        
        nm_min_requirement = min(self.model_params_dict["fan_in"] , key=lambda x: x[1]) # get the normal model parameters with the smallest minimum number of accounts
        self.min_fan_in_threshold = nm_min_requirement[1] - 1 # get the minimum number of accounts in normal model (note that min_accts = total number of accounts in pattern)

        # return a list of nodes with at least enough incoming edges as the minimum number of accounts, sorted with respect to how many in degrees there are
        candidate_list = [n for n in self.g.nodes() if self.g.in_degree(n) >= self.min_fan_in_threshold]
        return candidate_list


    def get_fan_out_candidates(self):
        """Returns a list of node ids that have at least degree_threshold outgoing edges.

        Returns:
            list: list of node ids that have at least degree_threshold outgoing edges
        """        
        nm_min_requirement = min(self.model_params_dict["fan_out"] , key=lambda x: x[1]) # get the normal model parameters with the smallest minimum number of accounts
        self.min_fan_out_threshold = nm_min_requirement[1] - 1 # get the minimum number of accounts in normal model (note that min_accts = total number of accounts in pattern)
        candidate_list = [n for n in self.g.nodes() if self.g.out_degree(n) >= self.min_fan_out_threshold]
        return candidate_list


    def number_unused(self):
        """Returns the number of unused nodes in the graph.

        Returns:
            int: Total number of unused nodes in the graph
        """        
        count = 0
        for type in self.remaining_count_dict:
            count += self.remaining_count_dict[type]
        return count


    def has_more(self):
        """Returns True if there are more unused nodes in the graph.

        Returns:
            bool: True if there are more unused nodes in the graph
        """        
        return self.number_unused() > 0


    def next(self, type):
        """Returns the next node id of a given type.

        Args:
            type (string): type of node to return

        Returns:
            int: node id of next node of given type.
        """         
        _, node_id = self.next_node_id(self.type_index[type], self.type_candidates[type]) # get next node id from fan in candidates

        if node_id is None:
            self.conclude(type)
        else:
            self.decrement(type)
            self.increment_used(type)
        return node_id


    def types(self):
        """Returns the types available in normal transactions.

        Returns:
            list: list of types available in normal transactions
        """        
        return self.remaining_count_dict.keys()


    def decrement(self, type):
        """Decrements the number of nodes of a given type.

        Args:
            type (string): type of node to decrement
        """        
        self.remaining_count_dict[type] -= 1


    def conclude(self, type):
        """Set the number of remaining nodes of a given type to 0.

        Args:
            type (string): type of node to conclude
        """        
        self.remaining_count_dict[type] = 0

    
    def increment_used(self, type):
        """Increments the number of used nodes of a given type.

        Args:
            type (string): type of node to increment
        """        
        self.used_count_dict[type] += 1

    
    def count(self, type):
        """Counts the number of nodes of a given type.

        Args:
            type (string): type of node to count

        Returns:
            int: number of nodes of a given type
        """        
        return self.remaining_count_dict[type]


    def post_update(self, node_id, type):
        
        if self.is_done(node_id, type): # check if node will be able to be main in other fan_in patterns
            self.type_candidates[type].pop(self.type_index[type]) # remove node_id from list of candidates (if popped, we dont need to increase index)
        else:
            self.type_index[type] += 1 # increment index (this is to allow next node considered to be different)


    def get_forward_candidates(self):
        """Return all vertices with at least one outgoing edge and at least one incoming edge.

        Returns:
            list: A list of vertices ranked by the number of in- and outgoing edges.
        """        
        return sorted(
            (n for n in self.g.nodes() if self.g.in_degree(n) >= 1 and self.g.out_degree(n) >= 1),
            key=lambda n: max(self.g.in_degree(n), self.g.out_degree(n))
        )


    def get_single_candidates(self):
        """Return all vertices with outgoing edges.

        Returns:
            list: A list of vertices ranked by the number of outgoing edges.
        """        
        return sorted(
            (n for n in self.g.nodes() if self.g.out_degree(n) >= 1),
            key=lambda n: self.g.out_degree(n)
        )


    def next_node_id(self, index, list):
        """Return the next node id from the list.

        Args:
            index (int): The current index.
            list (list): The list of node ids.

        Returns:
            int: The next index.
            int: The next node id.
        """        
        try:
            node_id = list[index]
        except IndexError:
            index = 0
            if len(list) == 0:
                return index, None
            node_id = list[index]
        return index, node_id

    
    def is_done(self, node_id, type):
        if type == 'fan_in':
            return self.is_done_fan_in(node_id, type)
        elif type == 'fan_out':
            return self.is_done_fan_out(node_id, type)
        elif type == 'forward':
            return self.is_done_forward(node_id, type)
        elif type == 'single':
            return self.is_done_single(node_id, type)
        elif type == 'mutual':
            return self.is_done_mutual(node_id, type)
        elif type == 'periodical':
            return self.is_done_periodical(node_id, type)


    def is_done_fan_in(self, node_id, type):
        pred_ids = self.g.predecessors(node_id) # get node-ids of incoming edges
        fan_in_or_not_list = [self.is_in_type_relationship(type, node_id, {node_id, pred_id}) for pred_id in pred_ids] #get boolean list of whether each predecessor has a fan_in relationship with node_id
        num_to_work_with = fan_in_or_not_list.count(False) # count the number of predecessors that do not have a fan_in relationship with node_id
        
        return num_to_work_with < self.min_fan_in_threshold
        

    def is_done_fan_out(self, node_id, type):
        succ_ids = self.g.successors(node_id) # get successors
        fan_out_or_not_list = [self.is_in_type_relationship(type, node_id, {node_id, succ_id}) for succ_id in succ_ids] # check if each successor has a fan_out relationship
        num_to_work_with = fan_out_or_not_list.count(False) # count the number of successors that do not have a fan_out relationship

        return num_to_work_with < self.min_fan_out_threshold 
        

    def is_done_forward(self, node_id, type):
        # forward is done when when all combinations of forwards have been found
        pred_ids = self.g.predecessors(node_id)
        succ_ids = self.g.successors(node_id)

        sets = ({node_id, pred_id, succ_id} for pred_id in pred_ids for succ_id in succ_ids)

        return all(self.is_in_type_relationship(type, node_id, set) for set in sets)

    
    def is_done_mutual(self, node_id, type):
        succ_ids = self.g.successors(node_id)
        return all(self.is_in_type_relationship(type, node_id, {node_id, succ_id}) for succ_id in succ_ids)


    def is_done_periodical(self, node_id, type):
        succ_ids = self.g.successors(node_id)
        return all(self.is_in_type_relationship(type, node_id, {node_id, succ_id}) for succ_id in succ_ids)


    def is_done_single(self, node_id, type):
        """Single is done when all the sucessors have been made into singles with this one
        because each directional can be a legal single as well as being part of another model.

        Args:
            node_id (int): The node id.
            type (string): The type of relationship.

        Returns:
            bool: True if the single is done.
        """        
        succ_ids = self.g.successors(node_id)
        return all(self.is_in_type_relationship(type, node_id, {node_id, succ_id}) for succ_id in succ_ids)


    def is_in_type_relationship(self, type, main_id, node_ids=set()):
        """Return True if any of the node_ids are already in a type relationship with the main_id.

        Args:
            type (string): The type of relationship.
            main_id (int): The main id.
            node_ids (set, optional): A set of node ids. Defaults to set().

        Returns:
            bool: True if any of the node_ids are already in a type relationship with the main_id.
        """        
        node_ids = set(node_ids) # make sure it's a set
        normal_models = self.g.node[main_id]['normal_models'] # get the transaction models associated with main_id
        filtereds = (nm for nm in normal_models if nm.type == type and nm.main_id == main_id) # filter out the normal models that are of the type and have the main_id
        return any(node_ids.issubset(filtered.node_ids) for filtered in filtereds) # return True if any of the filtered normal models contain the node_ids


    def normal_models_in_type_relationship(self, type, main_id, node_ids=set()):
        """Return a list of normal models where main_id is the main_id and the node_ids are a subset of the node_ids in the normal model.

        Args:
            type (string): The type of relationship.
            main_id (int): The main id.
            node_ids (set, optional): A set of node ids. Defaults to set().

        Returns:
            list: A list of normal models where main_id is the main_id and the node_ids are a subset of the node_ids in the normal model.
        """        
        node_ids = set(node_ids)
        normal_models = self.g.node[main_id]['normal_models'] # get the transaction models associated with main_id
        filtereds = (nm for nm in normal_models if nm.type == type and nm.main_id == main_id) # filter out the normal models that are of the type and have the main_id
        return [filtered for filtered in filtereds if node_ids.issubset(filtered.node_ids)] # return list of filtered normal models where node_ids are a subset of the node_ids


    def nodes_in_type_relation(self, type, node_id):
        """Return a list of sets of node_ids that are in a type relationship with the node_id.

        Args:
            type (string): The type of relationship.
            node_id (int): The node id.

        Returns:
            list: A list of sets of node_ids that are in a fan relationship with the node_id.
        """        
        normal_models = self.g.node[node_id]['normal_models'] # get the transaction models associated with node_id
        filtereds = (nm for nm in normal_models if nm.type == type and nm.main_id == node_id) # filter out the normal models that are of the type and have the node_id as the main_id
        return (filtered.node_ids_without_main() for filtered in filtereds) # return the node_ids without the main_id

    def find_available_candidate_neighbors(self, type, node_id):
        # This funcion is explicitly made for fan_in/fan_out
        
        if type == 'fan_in':
            neighbor_ids = self.g.predecessors(node_id) # get the predecessors of the node_id
        elif type == 'fan_out':
            neighbor_ids = self.g.successors(node_id)
            
        neighbors_in_type = set(self.nodes_in_type_relation(type, node_id)) # get the node_ids that are in a type relationship with the node_id being main
        candidates = set(neighbor_ids) - neighbors_in_type # subtract the nodes that are already in a fan relationship with node_id from the neighbor_ids
        
        indx = self.type_index[type] # get the index of the current type
        min_threshold = self.model_params_dict[type][indx][1]-1 # get the minimum number of accounts required        
        
        if len(candidates) >= min_threshold: 
            max_threshold = self.model_params_dict[type][indx][1]-1 # get the maximum number of allowed accounts
            n_max = min(len(candidates), max_threshold) # get the maximum number of accounts that can be used
            n_candidates = random.randint(min_threshold, n_max) # decide number of nodes in fan pattern
            candidates = set(random.sample(candidates, n_candidates)) # randomly select nodes from the candidates
            return candidates
        else:
            raise ValueError(f"There are not enough candidate for {type}")

