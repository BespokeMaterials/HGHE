"""
This is the code to preapre the DFT data and extract the key elements to a csv file.

From etch dft experiment we extract the following:
- atoms
- atomic postion
- neighbours
- lattice vectors
- H matrix
- S matrix
"""

"""
Convert the data from the DFT to more friendly and readable format.
Is a parser for aiida.fdf files
"""

import numpy as np
from sisl import get_sile
from utils import (list_subdirectories,
                   create_directory_if_not_exists,
                   save_dict_to_json,
                   generate_heatmap)
from tqdm import tqdm
import csv

def parse_atomic_file(file_path):
    """
    Parse the given file to extract atomic coordinates, atomic symbols, atomic types, and mesh cutoff.

    :param file_path: Path to the input file
    :return: A dictionary with atomic coordinates, symbols, types, and mesh cutoff
    """
    atomic_data = {
        "atomic_coordinates": [],
        "atomic_symbols": [],
        "atomic_types": [],
        "mesh_cutoff": None
    }

    with open(file_path, 'r') as file:
        lines = file.readlines()

    in_species_block = False
    in_lattice_vectors_block = False
    in_atomic_coords_block = False

    species_dict = {}

    for line in lines:
        line = line.strip()

        if line.startswith("meshcutoff"):
            atomic_data["mesh_cutoff"] = line.split()[1] + " " + line.split()[2]

        if line == "%block chemicalspecieslabel":
            in_species_block = True
            continue

        if line == "%endblock chemicalspecieslabel":
            in_species_block = False
            continue

        if line == "%block lattice-vectors":
            in_lattice_vectors_block = True
            continue

        if line == "%endblock lattice-vectors":
            in_lattice_vectors_block = False
            continue

        if line == "%block atomiccoordinatesandatomicspecies":
            in_atomic_coords_block = True
            continue

        if line == "%endblock atomiccoordinatesandatomicspecies":
            in_atomic_coords_block = False
            continue

        if in_species_block:
            parts = line.split()
            species_dict[int(parts[0])] = parts[2]

        if in_atomic_coords_block:
            parts = line.split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            atom_type = int(parts[3])
            atomic_symbol = species_dict[atom_type]
            atomic_data["atomic_coordinates"].append((x, y, z))
            atomic_data["atomic_symbols"].append(atomic_symbol)
            atomic_data["atomic_types"].append(atom_type)

    return atomic_data


def construct_json_from_fdf(fdf_path):
    """
    :param fdf_path: Path to the input file aida.fdf
    :return: A dictionary with structure information.
    """

    fdf = get_sile(fdf_path)
    h = fdf.read_hamiltonian()
    hmat = h.Hk([0, 0, 0]).todense()
    smat = h.Sk([0, 0, 0]).todense()
    geometry = fdf.read_geometry()
    lattice_vectors = geometry.cell
    atomic_symbols = geometry.atoms

    atomic_info = parse_atomic_file(fdf_path)
    atoms = []
    for i, simbol in enumerate(atomic_info["atomic_symbols"]):
        atom = {
            "simbol": str(simbol),
            "xyz": atomic_info['atomic_coordinates'][i],
            "nr_orbitals": len(atomic_symbols[atomic_info['atomic_types'][i] - 1]),
        }
        atoms.append(atom)

    data = {
        "structure": {
            "lattice vectors": lattice_vectors.tolist(),
            "atoms": atoms
        },
        "hmat": hmat.tolist(),
        "smat": smat.tolist()
    }
    return data


def parse_dft(dft_path, json_path):
    create_directory_if_not_exists(json_path)
    create_directory_if_not_exists(f"{json_path}_img")

    dft_reg = list_subdirectories(dft_path)
    csv_rows=[]
    for k,sample in enumerate(tqdm(dft_reg)):
        path = f"{dft_path}/{sample}/aiida.fdf"
        json_dc = construct_json_from_fdf(path)



        #reshpae to csv:
        csv_row={
            "id":k,
            "filename":sample,
            "nr_atoms":len(json_dc["structure"]["atoms"]),
            "lattice_vectors":json_dc["structure"]["lattice vectors"],
             "atoms_tipe":[a["simbol"] for a in  json_dc["structure"]["atoms"]],
             "atoms_xyz":[a["xyz"] for a in  json_dc["structure"]["atoms"]],
             "atoms_nr_orbitals":[a["nr_orbitals"] for a in  json_dc["structure"]["atoms"]],
             "hmat":json_dc["hmat"],
             "smat": json_dc["hmat"],
             }
        csv_rows.append(csv_row)

        #save_dict_to_json(json_dc, f"{json_path}/{sample}.json")
        hmat = np.array(json_dc["hmat"])
        filename = (f"{json_path}_img/{sample}_hmat.png")
        generate_heatmap(hmat, filename, grid1_step=1, grid2_step=13)
        smat = np.array(json_dc["smat"])
        filename = f"{json_path}_img/{sample}_smat.png"
        generate_heatmap(smat, filename, grid1_step=1, grid2_step=13)
    # Specify the CSV column names
    csv_columns = [k for k in csv_rows[0].keys()]
    # Writing to CSV file
    try:
        with open(f"{path_json}/DFT.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_rows:
                writer.writerow(data)
    except IOError:
        print("I/O error")

def main(dft_path, json_path):
    parse_dft(dft_path, json_path)

    print("Files processed successfully!")
    return 0


if __name__ == "__main__":
    path_to_dft_files = "/home/ICN2/atomut/HGHE/Data/DFT/BN_DFT"
    path_json = "DATA/DFT/aBN_DFT_CSV"
    main(path_to_dft_files, path_json)

