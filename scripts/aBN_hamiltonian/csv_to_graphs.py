from hghe import ElementGraph, OrbitalGraph
from hghe.enhancements import ChemEnhanceElementGraph, EdgeEnhanceElementGraph
from utils import number_to_6bit_list
import pandas as pd
import ast

def main(file_path,nr_atoms, radius):


    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    # Display the first few rows of the DataFrame
    print(df.head())

    graphs=[]

    print(df['nr_atoms'])
    for index, row in df.iterrows():
        if row['nr_atoms']>= nr_atoms and row['nr_atoms']<=nr_atoms+10 :
            print(row['nr_atoms'])
            # Build the atom graph
            atoms =ast.literal_eval(row["atoms_tipe"])
            coordinates= ast.literal_eval(row["atoms_xyz"])
            lattice_vectors =ast.literal_eval(row["lattice_vectors"])

            hmat=ast.literal_eval(row["hmat"])
            smat=ast.literal_eval(row["smat"])
            print(f"\nnr atoms:{len(atoms)}\nlattice_vectors:{lattice_vectors}\natoms:{atoms}\ncoordinates:{coordinates}")

            atomic_graph = ElementGraph(atoms, coordinates, lattice_vectors, radius)
            #atomic_graph.display_graph()
            print("atomic_graph:", atomic_graph)
            print("atomic_graph.data:", atomic_graph.data)
            print("atomic_graph.node_descriptor:", atomic_graph.node_descriptor)

            # Enhance the atomic graph
            # Node enhancement with chemical data
            chem_enh = ChemEnhanceElementGraph()
            chem_enh_atomic_graph = chem_enh.enhance_descriptor(atomic_graph)

            print("chem_enh_atomic_graph:", chem_enh_atomic_graph)
            print("chem_enh_atomic_graph.data:", chem_enh_atomic_graph.data)
            print("chem_enh_atomic_graph.node_descriptor", chem_enh_atomic_graph.node_descriptor)

            # Node edge enhancement:
            edge_enh = EdgeEnhanceElementGraph()
            enh_graph = edge_enh.enhance_descriptor(chem_enh_atomic_graph)

            print("enh_atomic_graph:", enh_graph)
            print("enh_atomic_graph.data:", enh_graph.data)
            print("enh_atomic_graph.node_descriptor", enh_graph.node_descriptor)
            print("enh_atomic_graph.edge_descriptor", enh_graph.edge_descriptor)

            # Let´s make it equivariant and remouve the x,y, z info from nodes
            enh_graph.edge_descriptor = enh_graph.edge_descriptor[3:]
            enh_graph.data.x = enh_graph.data.x[:, 3:]
            print("enh_atomic_graph.data:", enh_graph.data)
            print("enh_atomic_graph.node_descriptor", enh_graph.node_descriptor)



            # Extend to orbitals
            orbital_map = {"B": [f"o{o}" for o in range(13)],
                           "N": [f"o{o}" for o in range(13)],
                           "H":[f"o{o}" for o in range(13)],
                           "C":[f"o{o}" for o in range(13)]}
            orbital_encode = {}
            for o in range(13):
                orbital_encode[f"o{o}"]=number_to_6bit_list(0)

            orbital_graph = OrbitalGraph(atomic_graph, orbital_map, orbital_encode)
            orbital_graph.display_graph()

            print("atomic_graph:", atomic_graph)
            print("atomic_graph.data:", atomic_graph.data)
            print("orbital_graph:", orbital_graph)
            print("orbital_graph.data:", orbital_graph.data)
            print("orbitals:", orbital_graph.orbitals)

            graphs.append(orbital_graph)




if __name__ == "__main__":
    example_file_path = "DATA/DFT/aBN_DFT_CSV/DFT.csv"
    nr_atoms = 60
    main(example_file_path, nr_atoms, radius=7)
