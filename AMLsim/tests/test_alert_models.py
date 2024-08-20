import networkx as nx
TYPES = ["fan_in", "fan_out", "bipartite", "stack", "cycle", "scatter_gather", "gather_scatter"]
            
def perform_alert_test(alert_models, 
                       EXPECTED_NO_OF_MODELS, 
                       EXPECTED_SCHEDULE_ID, 
                       MIN_ACCTS, 
                       MAX_ACCTS, 
                       MIN_PERIOD, 
                       MAX_PERIOD, 
                       SOURCE_TYPE):
    
    for i, type in enumerate(TYPES):
        assert len(alert_models[type]) == EXPECTED_NO_OF_MODELS
        
        for nm in alert_models[type]:
            assert nm.graph.get("scheduleID") == EXPECTED_SCHEDULE_ID
            assert len(nm.nodes()) >= MIN_ACCTS
            assert len(nm.nodes()) <= MAX_ACCTS
            assert nm.graph.get("source_type") == SOURCE_TYPE[i]
            
            # ensure all transactions are within the expected range
            for node, neighbors in nm.adjacency():
                for neighbor, edge_data in neighbors.items():
                    assert edge_data["date"] >= MIN_PERIOD
                    assert edge_data["date"] <= MAX_PERIOD
            
            if type in ["cycle"]:
                # Ensure there is a cycle in the graph
                assert len(list(nx.simple_cycles(nm))) > 0
            else:
                # Ensure there is no cycle in the graph
                assert len(list(nx.simple_cycles(nm))) == 0
            
            # Ensure incoming transactions are done before outgoing in each layer
            if type in ["stack"]:
                for node in nm.nodes():
                    largest_in_date = -1
                    smallest_out_date = 1e9

                    for pred in nm.predecessors(node):
                        pred_value = nm.get_edge_data(pred, node).get("date")
                        largest_in_date = max(largest_in_date, pred_value)
            
                    for succ in nm.successors(node):
                        succ_value = nm.get_edge_data(node, succ).get("date")
                        smallest_out_date = min(smallest_out_date, succ_value)
                    
                    if largest_in_date != -1 and smallest_out_date != 1e9:
                        assert largest_in_date < smallest_out_date

def test_alert_small_graph(small_graph):
    txg = small_graph
    alert_models = dict()
    for type in TYPES:
        alert_models[type] = [nm for nm in txg.alert_groups.values() if nm.graph["reason"] == type]

    EXPECTED_NO_OF_MODELS = 1
    EXPECTED_SCHEDULE_ID = 2
    MIN_ACCTS = 3
    MAX_ACCTS = 4
    MIN_PERIOD = 1
    MAX_PERIOD = 20
    SOURCE_TYPE = ["TRANSFER", "TRANSFER", "CASH", "CASH", "TRANSFER", "TRANSFER", "TRANSFER"]
    
    perform_alert_test(alert_models, 
                       EXPECTED_NO_OF_MODELS, 
                       EXPECTED_SCHEDULE_ID, 
                       MIN_ACCTS, 
                       MAX_ACCTS, 
                       MIN_PERIOD, 
                       MAX_PERIOD, 
                       SOURCE_TYPE)

def test_alert_large_graph(large_graph):
    txg = large_graph
    alert_models = dict()
    for type in TYPES:
        alert_models[type] = [nm for nm in txg.alert_groups.values() if nm.graph["reason"] == type]

    EXPECTED_NO_OF_MODELS = 10
    EXPECTED_SCHEDULE_ID = 1
    MIN_ACCTS = 10
    MAX_ACCTS = 20
    MIN_PERIOD = 1
    MAX_PERIOD = 100
    SOURCE_TYPE = ["TRANSFER", "TRANSFER", "TRANSFER", "TRANSFER", "TRANSFER", "TRANSFER", "TRANSFER"]
    
    perform_alert_test(alert_models,
                        EXPECTED_NO_OF_MODELS, 
                        EXPECTED_SCHEDULE_ID, 
                        MIN_ACCTS, 
                        MAX_ACCTS, 
                        MIN_PERIOD, 
                        MAX_PERIOD, 
                        SOURCE_TYPE)

if __name__ == "__main__":
    # Run specific test functions
    import pytest
    pytest.main(["-v", "./tests"])
    