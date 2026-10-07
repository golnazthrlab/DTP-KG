import networkx as nx
import ast
import pandas as pd

class BiologicalNetwork:
    def __init__(self, ddi_path, dpi_path, ppi_path):

        # Load dataframes
        self.ddi_data = pd.read_csv(ddi_path)
        self.dpi_data = pd.read_csv(dpi_path)
        self.ppi_data = pd.read_csv(ppi_path)

        # Normalize IDs
        self.ddi_data["drug1"] = self.ddi_data["drug1"].astype(str)
        self.ddi_data["drug2"] = self.ddi_data["drug2"].astype(str)
        self.dpi_data["db_id"] = self.dpi_data["db_id"].astype(str)

        # Parse lists
        self.dpi_data["targets_flat"] = self.dpi_data["targets"].apply(self._safe_parse)
        self.ppi_data["interactions"] = self.ppi_data["interactions"].apply(self._safe_parse)

        # Create graph
        self.graph = nx.Graph()
        self._construct_network()
        self._remove_unconnected_nodes()

        print("Network constructed successfully.")

    def _safe_parse(self, x):
        """Parse a possibly nested string/list structure into a flat list of strings."""
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
            except:
                return []
        else:
            parsed = x

        # Now flatten it
        flat = []

        def _flatten(item):
            if isinstance(item, list):
                for sub in item:
                    _flatten(sub)
            elif isinstance(item, str):
                flat.append(item)
            else:
                # ignore unexpected types
                pass

        _flatten(parsed)
        return flat

    # -------------------------------------------------------------------
    #                      NETWORK CONSTRUCTION
    # -------------------------------------------------------------------

    def _construct_network(self):
        G = self.graph

        # Collect full sets
        all_drugs = set(self.dpi_data["db_id"])
        all_targets = set(t for arr in self.dpi_data["targets_flat"] for t in arr)
        all_proteins = set(self.ppi_data["uniprot_id"])

        # 1. Add all drug nodes
        for d in all_drugs:
            G.add_node(d, type="drug")

        # 2. Add target nodes
        for t in all_targets:
            G.add_node(t, type="target")

        # 3. Add all PPI protein nodes (only those not already targets)
        for p in all_proteins:
            if p not in all_targets:
                G.add_node(p, type="protein")

        # 4. Add Drug–Target edges
        for _, row in self.dpi_data.iterrows():
            d = row["db_id"]
            for t in row["targets_flat"]:
                if t in G:
                    G.add_edge(d, t, weight=1.0, type="drug-target")

        # 5. Add Drug–Drug edges (DDI)
        for _, row in self.ddi_data.iterrows():
            d1, d2 = row["drug1"], row["drug2"]
            if d1 not in G:
                G.add_node(d1, type="drug")
            if d2 not in G:
                G.add_node(d2, type="drug")
            G.add_edge(d1, d2, weight=1.0, type="drug-drug")

        # 6. Add Protein–Protein edges (including target-target, protein-target)
        for _, row in self.ppi_data.iterrows():
            u = row["uniprot_id"]
            for v in row["interactions"]:
                if u == v:
                    continue
                if v not in G:
                    # if v is not a drug/target, then it's a protein
                    G.add_node(v, type="protein")

                # determine edge type
                ut = G.nodes[u]["type"]
                vt = G.nodes[v]["type"]
                etype = f"{ut}-{vt}"

                # add edge
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=1.0, type=etype)

    def _remove_unconnected_nodes(self):
        G = self.graph
        to_remove = [n for n in G if G.degree(n) == 0]
        for n in to_remove:
            G.remove_node(n)

    def save_graph(self, path):
        nx.write_graphml(self.graph, path)
