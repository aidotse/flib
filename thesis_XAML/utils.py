from collections import deque
import torch
from torch_geometric.data import Data


def build_subgraph(data, node_to_explain, layers=2):
    edge_index = data.edge_index
    nodes = [node_to_explain]
    
    queue = deque([(node_to_explain, 0)])  # Initialize queue with starting node and its depth
    visited=[]
    edges=[]
    while queue!=deque([]):
        #print(queue)
        node, depth = queue.popleft()
        if depth==layers:
            break
        visited.append(node)
        #print('popped node',node)
        #src and dst neighbors
        src = edge_index[1][edge_index[0]==node].tolist()
        dst = edge_index[0][edge_index[1]==node].tolist()
        neighbors = list(set(src+dst))

        #Add edges to edges list
        for n in neighbors:
            if (node,n) not in edges and (n,node) not in edges:
                edges.append((node,n))

            if n not in visited:
                visited.append(n)
                nodes.append(n)
                queue.extend([(n,depth+1)])
    
    
    nodes = torch.tensor(nodes)
    edges = torch.tensor(edges).t().contiguous()   

    org_to_new_mapping, new_to_org_mapping, edges = node_index_mapping(nodes,edges)

    #find feature vectors for nodes
    node_features = data.x[nodes]

    #create data object for subgraph using node_fetures and edges
    data = Data(x=node_features, edge_index=edges)

    return data, org_to_new_mapping, new_to_org_mapping


def node_index_mapping(nodes,edges):

    # Create a mapping from original node indices to new indices
    node_index_mapping = {(old_idx.item(), new_idx) for new_idx, old_idx in enumerate(nodes)}

    # Create a dictionary mapping original indices to new indices
    org_to_new_mapping = {old_idx: new_idx for old_idx, new_idx in node_index_mapping}

    # Create a dictionary mapping new indices to original indices
    new_to_org_mapping = {new_idx: old_idx for old_idx, new_idx in node_index_mapping}

    # Update the edge indices to use the new node indices
    for old_idx, new_idx in node_index_mapping:
        edges[edges == old_idx] = new_idx

    return org_to_new_mapping, new_to_org_mapping, edges