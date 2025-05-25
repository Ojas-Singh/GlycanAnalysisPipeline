from pathlib import Path
from typing import List, Tuple, Any
import logging

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from lib import pdb, graph, dihedral
import lib.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_mol2_bonds(file_path: Path) -> List[Tuple[int, int]]:
    """Parse bond information from MOL2 file and ensure graph connectivity.
    
    Args:
        file_path: Path to MOL2 file
        
    Returns:
        List of tuples containing bonded atom pairs with missing connections added
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Parse atom coordinates
        atom_start = -1
        atom_end = -1
        for i, line in enumerate(lines):
            if line.strip() == "@<TRIPOS>ATOM":
                atom_start = i + 1
            elif atom_start != -1 and line.strip().startswith("@<TRIPOS>"):
                atom_end = i
                break

        atom_coords = {}
        if atom_start != -1:
            if atom_end == -1:
                atom_end = len(lines)
            
            for line in lines[atom_start:atom_end]:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        atom_id = int(parts[0])
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        atom_coords[atom_id] = (x, y, z)
                except (ValueError, IndexError):
                    continue

        # Parse bonds
        bond_start = -1
        bond_end = -1
        for i, line in enumerate(lines):
            if line.strip() == "@<TRIPOS>BOND":
                bond_start = i + 1
            elif bond_start != -1 and line.strip().startswith("@<TRIPOS>"):
                bond_end = i
                break

        if bond_start == -1:
            raise ValueError("No @<TRIPOS>BOND section found in the mol2 file.")

        if bond_end == -1:
            bond_end = len(lines)

        bonded_atoms = []
        for line in lines[bond_start:bond_end]:
            try:
                _, atom1, atom2, _ = line.strip().split()
                bonded_atoms.append((int(atom1), int(atom2)))
                
            except ValueError:
                logger.warning(f"Skipping invalid bond line: {line.strip()}")
                continue
        
        # Check connectivity and add missing bonds if needed
        G = nx.Graph()
        G.add_edges_from(bonded_atoms)
        
        # Find all connected components
        components = list(nx.connected_components(G))
        
        if len(components) > 1:
            logger.warning(f"Graph not fully connected. Found {len(components)} components. Adding missing bonds.")
            
            # Connect components by adding bonds between closest atoms by coordinate distance
            while len(components) > 1:
                min_distance = float('inf')
                best_connection = None
                
                for i in range(len(components)):
                    for j in range(i + 1, len(components)):
                        comp1_nodes = list(components[i])
                        comp2_nodes = list(components[j])
                        
                        # Find closest atoms between components by 3D distance
                        for node1 in comp1_nodes:
                            for node2 in comp2_nodes:
                                if node1 in atom_coords and node2 in atom_coords:
                                    coord1 = atom_coords[node1]
                                    coord2 = atom_coords[node2]
                                    distance = np.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(coord1, coord2)))
                                    if distance < min_distance:
                                        min_distance = distance
                                        best_connection = (node1, node2)
                
                if best_connection:
                    node1, node2 = best_connection
                    bonded_atoms.append((node1, node2))
                    logger.info(f"Added missing bond: {node1} - {node2} (distance: {min_distance:.3f})")
                    
                    # Update the graph with the new bond and recalculate components
                    G.add_edge(node1, node2)
                    components = list(nx.connected_components(G))
                else:
                    logger.warning("Could not find atoms with coordinates to connect components")
                    break

        return bonded_atoms
    except Exception as e:
        logger.error(f"Failed to parse MOL2 file {file_path}: {str(e)}")
        raise

def torsionspairs(pdbdata: Any, name: str) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Calculate torsion pairs for a molecule.
    
    Args:
        pdbdata: PDB structure data
        name: Molecule name
        
    Returns:
        Tuple of (all pairs, external torsions, internal torsions)
    """
    try:
        mol2_path = Path(config.data_dir) / name / f"{name}.mol2"
        df = pdb.to_DF(pdbdata)
        connect = parse_mol2_bonds(mol2_path)

        G = nx.Graph()
        G.add_edges_from(connect)
        
        # Remove hydrogens
        H_list = df.loc[df['Element']=="H", 'Number']
        G.remove_nodes_from(H_list)

        # Find cycles
        cycle = [atom for c in nx.cycle_basis(G) for atom in c]
        
        # Identify red nodes (degree 2, not in cycle)
        red = [node for node in G if G.degree(node)==2 and node not in cycle]
        
        # Create color map for visualization
        color_map = ['red' if node in red else 'orange' for node in G.nodes()]
        
        # Plot network graph
        plot_network_graph(G, color_map, name)
        
        external = []
        internal = []

        # Process red nodes
        for node in red:
            j, k = list(G.neighbors(node))
            
            if j in red or k in red:
                _process_red_neighbors(df, G, node, j, k, red, external)
            else:
                _process_normal_nodes(df, G, node, j, k, cycle, external, internal)

        # Save results
        pairs = external + internal
        _save_torsion_parts(pdbdata, name, pairs, connect)

        logger.info(f"Processed {len(pairs)} torsion pairs for {name}")
        return pairs, external, internal

    except Exception as e:
        logger.error(f"Failed to process torsions for {name}: {str(e)}")
        raise

def _process_red_neighbors(df: pd.DataFrame, G: nx.Graph, node: int, j: int, k: int, 
                         red: List[int], external: List[List[int]]) -> None:
    """Process torsions for nodes with red neighbors."""
    if df.loc[df['Number']==node, 'Name'].iloc[0] != "O6":
        return
        
    for neighbor, other in [(j,k), (k,j)]:
        if neighbor not in red:
            continue
            
        l = [x for x in G.neighbors(neighbor) if x != node]
        m = [x for x in G.neighbors(other) if x != node]
        
        aa = df.loc[df['Number']==m[0], 'Name'].iloc[0]
        external.append([l[0], neighbor, node, other])
        
        if df.loc[df['Number']==other, 'Name'].iloc[0] in ["C1", "C2"]:
            external.append([m[0] if aa in ["O5","C1"] else m[1], other, node, neighbor])
            
        res_id = df.loc[df['Number']==l[0], 'ResId'].iloc[0]
        c4_num = df.loc[(df['ResId']==res_id) & (df['Name']=="C4"), 'Number'].iloc[0]
        external.append([c4_num, l[0], neighbor, node])

def _process_normal_nodes(df: pd.DataFrame, G: nx.Graph, node: int, j: int, k: int,
                        cycle: List[int], external: List[List[int]], internal: List[List[int]]) -> None:
    """Process torsions for nodes with normal neighbors."""
    for neighbor, other in [(j,k), (k,j)]:
        if G.degree(neighbor) <= 1:
            continue
            
        neighbors = [x for x in G.neighbors(neighbor) if x != node]
        
        if any(n in cycle for n in neighbors) and other in cycle:
            name = df.loc[df['Number']==neighbor, 'Name'].iloc[0]
            
            if name == "C1":
                n0_name = df.loc[df['Number']==neighbors[0], 'Name'].iloc[0]
                external.append([neighbors[0] if n0_name=="O5" else neighbors[1], 
                               neighbor, node, other])
            elif G.degree(neighbor) == 4:
                res_id = df.loc[df['Number']==neighbor, 'ResId'].iloc[0]
                c1_num = df.loc[(df['ResId']==res_id) & (df['Name']=="C1"), 'Number'].iloc[0]
                external.append([c1_num, neighbor, node, other])
            else:
                next_c = int(name.strip("C")) + 1
                res_id = df.loc[df['Number']==neighbor, 'ResId'].iloc[0]
                next_num = df.loc[(df['ResId']==res_id) & 
                                (df['Name']==f"C{next_c}"), 'Number'].iloc[0]
                external.append([other, node, neighbor, next_num])
        else:
            internal.append([neighbors[0], neighbor, node, other])

def _save_torsion_parts(pdbdata: Any, name: str, pairs: List[List[int]], 
                       connect: List[Tuple[int, int]]) -> None:
    """Save torsion parts data."""
    anotherlist = []
    for pair in pairs:
        G = nx.Graph()
        G.add_edges_from(connect)
        node1, node2 = pair[1], pair[2]
        G.remove_edge(node1, node2)
        
        arr = np.ones(len(pdbdata[0]), dtype=bool)
        for node in nx.node_connected_component(G, node1):
            arr[node-1] = False
        anotherlist.append(arr)
        
    output_path = Path(config.data_dir) / name / "output" / "torparts"
    np.savez_compressed(output_path, a=pairs, b=anotherlist, allow_pickle=True)
    logger.info(f"Saved torsion parts to {output_path}")

def plot_network_graph(G: nx.Graph, color_map: List[str], name: str) -> None:
    """Plot molecular network graph.
    
    Args:
        G: NetworkX graph
        color_map: Node colors
        name: Molecule name
    """
    try:
        plt.figure(figsize=(10, 10))
        nx.draw(
            G,
            pos=nx.kamada_kawai_layout(G),
            node_size=800,
            node_color=color_map,
            alpha=0.8,
            with_labels=True,
            width=2,
            edge_color='grey',
            font_size=18
        )
        
        output_path = Path(config.data_dir) / name / "output" / "network_graph.png"
        plt.savefig(output_path, dpi=450, bbox_inches='tight')
        plt.close()
        logger.info(f"Network graph saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to plot network graph: {str(e)}")
