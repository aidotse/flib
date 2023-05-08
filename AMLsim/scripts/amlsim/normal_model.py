class NormalModel:
    """Class that represents a normal model of a given node.
    """    
    def __init__(self, id, type, node_ids, main_id):
        """Constructor of the class.

        Args:
            id (int): normal model id
            type (string): type of transaction associated with the node
            node_ids (set): set of node ids that are assciated with the transaction
            main_id (int): id of the main node
        """        
        self.id = id 
        self.type = type
        self.node_ids = node_ids or set()
        self.main_id = main_id


    def add_account(self, id):
        """Adds an account to the model.

        Args:
            id (int): The id of the account to add.
        """        
        self.node_ids.add(id)


    def is_main(self, node_id):
        """Checks if the node is the main node of the model.

        Args:
            node_id (int): The id of the node to check.

        Returns:
            bool: True if the node is the main node of the model, False otherwise.
        """        
        return node_id == self.main_id


    def remove_node_ids(self, node_ids):
        """Removes a set of node ids from the model.

        Args:
            node_ids (set): The set of node ids to remove.
        """        
        self.node_ids = self.node_ids - node_ids


    def node_ids_without_main(self):
        """Returns the set of node ids without the main node.

        Returns:
            set: The set of node ids without the main node.
        """        
        return self.node_ids - { self.main_id }


