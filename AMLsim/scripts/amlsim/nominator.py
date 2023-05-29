import networkx as nx

class Nominator:
    """Class responsible for nominating nodes for transactions.
    """    
    def __init__(self, g, degree_threshold):
        self.g = g
        self.degree_threshold = degree_threshold
        self.remaining_count_dict = dict()
        self.used_count_dict = dict()
        self.fan_in_candidates = self.get_fan_in_candidates()
        self.fan_out_candidates = self.get_fan_out_candidates()
        self.alt_fan_in_candidates = []
        self.alt_fan_out_candidates = []
        self.forward_candidates = self.get_forward_candidates()
        self.single_candidates = self.get_single_candidates()
        self.mutual_candidates = self.single_candidates.copy()
        self.periodical_candidates = self.single_candidates.copy()
        self.empty_list_message = 'pop from empty list'

      
        self.type_index = 0
        self.forward_index = 0
        self.fan_in_index = 0
        self.fan_out_index = 0
        self.alt_fan_in_index = 0
        self.alt_fan_out_index = 0
        self.single_index = 0
        self.mutual_index = 0
        self.periodical_index = 0


    def initialize_count(self, type, count):
        """Counts the number of nodes of a given type.

        Args:
            type (string): type of node to count
            count (int): number of types to count
        """        
        if type in self.remaining_count_dict:
            self.remaining_count_dict[type] += count
        else:
            self.remaining_count_dict[type] = count
        self.used_count_dict[type] = 0


    def get_fan_in_candidates(self):
        """Returns a list of node ids that have at least degree_threshold incoming edges.

        Returns:
            list: list of node ids that have at least degree_threshold incoming edges
        """        
        return sorted(
            (n for n in self.g.nodes() if self.is_fan_in_candidate(n)),
            key=lambda n: self.g.out_degree(n)
        )

    def get_fan_out_candidates(self):
        """Returns a list of node ids that have at least degree_threshold outgoing edges.

        Returns:
            list: list of node ids that have at least degree_threshold outgoing edges
        """        
        return sorted(
            (n for n in self.g.nodes() if self.is_fan_out_candidate(n)),
            key=lambda n: self.g.in_degree(n)
        )

    def is_fan_in_candidate(self, node_id):
        """Returns True if node_id has at least degree_threshold incoming edges.

        Args:
            node_id (_type_): node id of node to check

        Returns:
            bool: True if node_id has at least degree_threshold incoming edges
        """        
        return self.g.in_degree(node_id) >= self.degree_threshold

    def is_fan_out_candidate(self, node_id):
        """Returns True if node_id has at least degree_threshold outgoing edges.

        Args:
            node_id (_type_): node id of node to check

        Returns:
            bool: True if node_id has at least degree_threshold outgoing edges
        """        
        return self.g.out_degree(node_id) >= self.degree_threshold

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
        node_id = None
        if type == 'fan_in':
            node_id = self.next_fan_in(type)
        elif type == 'fan_out':
            node_id = self.next_fan_out(type)
        elif type == 'forward':
            node_id = self.next_forward(type)
        elif type == 'single':
            node_id = self.next_single(type)
        elif type == 'mutual':
            node_id = self.next_mutual(type)
        elif type == 'periodical':
            node_id = self.next_periodical(type)

        if node_id is None:
            self.conclude(type)
        else:
            self.decrement(type)
            self.increment_used(type)
        return node_id


    def current_type(self):
        types = list(self.remaining_count_dict)
        return types[self.type_index]


    def increment_type_index(self):
        if not self.has_more():
            raise StopIteration
        count = 0
        while (count == 0):
            types = list(self.remaining_count_dict)
            self.type_index += 1
            try:
                types[self.type_index]
            except IndexError:
                self.type_index = 0
            type = types[self.type_index]
            count = self.remaining_count_dict[type]


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


    def next_fan_in(self, type):
        
        self.fan_in_index, node_id = self.next_node_id(self.fan_in_index, self.fan_in_candidates) # get next node id from fan in candidates
        if node_id is None:
            return self.next_alt_fan_in(type)

        try:
            self.fan_out_candidates.remove(node_id) # remove from opposite
        except ValueError:
            pass
        
        return node_id


    def next_fan_out(self, type):
        self.fan_out_index, node_id = self.next_node_id(self.fan_out_index, self.fan_out_candidates)
        if node_id is None:
            return self.next_alt_fan_out(type)

        try: 
            self.fan_in_candidates.remove(node_id) # remove from opposite
        except ValueError:
            pass

        return node_id


    def next_alt_fan_in(self, type):
        self.alt_fan_in_index, node_id = self.next_node_id(self.alt_fan_in_index, self.alt_fan_in_candidates)

        if node_id is None:
            return self.conclude(type)
        return node_id


    def next_alt_fan_out(self, type):
        self.alt_fan_out_index, node_id = self.next_node_id(self.alt_fan_out_index, self.alt_fan_out_candidates)

        if node_id is None:
            return self.conclude(type)
        return node_id


    def next_forward(self, type):
        self.forward_index, node_id = self.next_node_id(self.forward_index, self.forward_candidates)
        if node_id is None:
            return self.conclude(type)
        return node_id

    
    def next_single(self, type):
        """Returns the next node_id of provided type.

        Args:
            type (string): type of node to return

        Returns:
            int: node_id of next node of provided type
        """        
        self.single_index, node_id = self.next_node_id(self.single_index, self.single_candidates)
        if node_id is None:
            return self.conclude(type)
        return node_id


    def next_periodical(self, type):
        self.periodical_index, node_id = self.next_node_id(self.periodical_index, self.periodical_candidates)
        if node_id is None:
            return self.conclude(type)
        return node_id
    

    def next_mutual(self, type):
        self.mutual_index, node_id = self.next_node_id(self.mutual_index, self.mutual_candidates)
        if node_id is None:
            return self.conclude(type)
        return node_id


    def post_single(self, node_id, type):
        """Removes the node_id from the list of single candidates if all successors have a type relationship with node_id.

        Args:
            node_id (int): node_id of node to check
            type (string): type of relationship to check
        """        
        if self.is_done(node_id, type): # if all successors have a type relationship with node_id
            self.single_candidates.pop(self.single_index) # remove from single candidates
        else:
            self.single_index += 1 # otherwise increment index


    def post_fan_in(self, node_id, type):
        if not self.fan_in_candidates:
            return self.post_alt_fan_in(node_id, type)
        
        if self.is_done(node_id, type):
            candidate = self.fan_in_candidates.pop(self.fan_in_index)
            if not self.is_done(node_id, 'fan_out'):
                self.alt_fan_out_candidates.append(candidate)
        else:
            self.fan_in_index += 1



    def post_alt_fan_in(self, node_id, type):
        if self.is_done(node_id, type):
            self.alt_fan_in_candidates.pop(self.alt_fan_in_index)
        else:
            self.alt_fan_in_index += 1

    
    def post_alt_fan_out(self, node_id, type):
        if self.is_done(node_id, type):
            self.alt_fan_out_candidates.pop(self.alt_fan_out_index)
        else:
            self.alt_fan_out_index += 1


    def post_fan_out(self, node_id, type):
        if not self.fan_out_candidates:
            return self.post_alt_fan_out(node_id, type)

        if self.is_done(node_id, type):
            candidate = self.fan_out_candidates.pop(self.fan_out_index) # remove from fan out candidates
            if not self.is_done(node_id, 'fan_in'): # check in the other direction
                self.alt_fan_in_candidates.append(candidate) # add to alt fan in candidates
        else:
            self.fan_out_index += 1


    def post_mutual(self, node_id, type):
        if self.is_done(node_id, type):
            self.mutual_candidates.pop(self.mutual_index)
        else:
            self.mutual_index += 1


    def post_periodical(self, node_id, type):
        if self.is_done(node_id, type):
            self.periodical_candidates.pop(self.periodical_index)
        else:
            self.periodical_index += 1
    
    
    def post_forward(self, node_id, type):
        if self.is_done(node_id, type):
            self.forward_candidates.pop(self.forward_index)
        else:
            self.forward_index += 1


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
        # num to work with is 
        # fan_ins mod threshold plus those not fan_in
        # if num to work with is less than threshold, ya done.
        pred_ids = self.g.predecessors(node_id)
        fan_in_or_not_list = [self.is_in_type_relationship(type, node_id, {node_id, pred_id}) for pred_id in pred_ids]
        num_not = fan_in_or_not_list.count(False)
        num_fan_in = fan_in_or_not_list.count(True)

        num_to_work_with = (num_fan_in % self.degree_threshold) + num_not #how many can be spared from current fan-in assignments and how many are not assigned fan-in
        return num_to_work_with < self.degree_threshold
        

    def is_done_fan_out(self, node_id, type):
        # num to work with is 
        # fan_outs mod threshold plus those not fan_out
        # num to work with is less than threshold, ya done.
        succ_ids = self.g.successors(node_id) # get successors
        fan_out_or_not_list = [self.is_in_type_relationship(type, node_id, {node_id, succ_id}) for succ_id in succ_ids] # check if each successor has a fan_out relationship
        num_not = fan_out_or_not_list.count(False) # count the number of successors that do not have a fan_out relationship
        num_fan_out = fan_out_or_not_list.count(True) # count the number of successors that do have a fan_out relationship

        num_to_work_with = (num_fan_out % self.degree_threshold) + num_not
        return num_to_work_with < self.degree_threshold
        

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
        return [filtered for filtered in filtereds if node_ids.issubset(filtered.node_ids)] # return True if any of the filtered normal models contain the node_ids


    def fan_clumps(self, type, node_id):
        """Return a list of sets of node_ids that are in a fan relationship with the node_id.

        Args:
            type (string): The type of relationship.
            node_id (int): The node id.

        Returns:
            list: A list of sets of node_ids that are in a fan relationship with the node_id.
        """        
        normal_models = self.g.node[node_id]['normal_models'] # get the transaction models associated with node_id
        filtereds = (nm for nm in normal_models if nm.type == type and nm.main_id == node_id) # filter out the normal models that are of the type and have the node_id as the main_id
        return (filtered.node_ids_without_main() for filtered in filtereds) # return the node_ids without the main_id


    def fan_in_breakdown(self, type, node_id):
        """Return a set of node_ids that are potentially in a fan-in relationship with the node_id.

        Args:
            type (string): The type of relationship.
            node_id (int): The node id.

        Returns:
            set: A set of node_ids that are potentially in a fan-in relationship with the node_id.
        """        
        pred_ids = self.g.predecessors(node_id) # get the predecessors of the node_id
        return self.fan_breakdown_candidates(type, node_id, set(pred_ids))


    def fan_out_breakdown(self, type, node_id):
        """Return a set of node_ids that are potentially in a fan-out relationship with the node_id.

        Args:
            type (string): The type of relationship.
            node_id (int): The node id.

        Returns:
            set: A set of node_ids that are potentially in a fan-out relationship with the node_id.
        """        
        succ_ids = self.g.successors(node_id) # get the successors of the node_id
        return self.fan_breakdown_candidates(type, node_id, set(succ_ids))
    

    def fan_breakdown_candidates(self, type_, node_id, neighbor_ids):
        """ Return a list of sets of node_ids that are in a fan relationship with the node_id.

        Args:
            type_ (string): The type of relationship.
            node_id (int): The node id.
            neighbor_ids (set): A set of node ids.
            
        Raises:
            ValueError: _description_

        Returns:
            set: A set of node ids that are candidates for being in a fan relationship with the node_id.
        """        
        candidates = set()

        fan_clumps = self.fan_clumps(type_, node_id) # get the node_ids that are in a fan relationship with the node_id
        fan_nodes = set()
        for fan_clump in fan_clumps:
            fan_nodes = fan_nodes | fan_clump # get the union of the fan_clumps

        candidates = neighbor_ids - fan_nodes # get the set difference between the neighbors and the fan_nodes
        if len(candidates) >= self.degree_threshold: 
            return candidates
        else:
            # if there are not enough candidates, then we need to add some from the fan_clumps
            while (len(candidates) < self.degree_threshold):
                touched = False
                for clump in fan_clumps:
                    if len(clump) > self.degree_threshold:
                       n_id = clump.pop()
                       candidates.add(n_id)
                       touched = True
                if touched == False:
                    raise ValueError('something broke in breakdown')
            return candidates
        

