"""
Add node properties.
"""
from jarvis.core.specie import Specie
import torch


class ChemEnhanceElementGraph:
    def __init__(self):
        """
        This class is used to add additional node proprieties to a graph.
        The proprieties are generated using only the atomic element in that node thru "cfid" and jarvis

        """

        self.chem_element_descriptor = [
            "is_halogen",
            "row",
            "GV",
            "nfunfill",
            "C-9",
            "C-8",
            "C-7",
            "C-6",
            "C-5",
            "C-4",
            "C-3",
            "C-2",
            "C-1",
            "C-0",
            "me1",
            "me3",
            "me2",
            "max_oxid_s",
            "npvalence",
            "mp",
            "first_ion_en",
            "ndunfill",
            "op_eg",
            "jv_enp",
            "nfvalence",
            "polzbl",
            "oq_bg",
            "atom_rad",
            "atom_mass",
            "is_alkali",
            "C-13",
            "C-12",
            "C-11",
            "C-10",
            "C-17",
            "C-16",
            "C-15",
            "C-14",
            "C-19",
            "C-18",
            "voro_coord",
            "is_noble_gas",
            "e1",
            "e3",
            "e2",
            "is_lanthanoid",
            "ndvalence",
            "KV",
            "min_oxid_s",
            "nsunfill",
            "C-26",
            "X",
            "is_actinoid",
            "C-28",
            "C-29",
            "C-27",
            "C-24",
            "C-25",
            "C-22",
            "C-23",
            "C-20",
            "C-21",
            "avg_ion_rad",
            "nsvalence",
            "is_metalloid",
            "elec_aff",
            "coulmn",
            "mol_vol",
            "bp",
            "C-31",
            "C-30",
            "C-33",
            "C-32",
            "C-35",
            "C-34",
            "is_transition_metal",
            "block",
            "therm_cond",
            "Z",
            "is_alkaline",
            "npunfill",
            "oq_enp",
            "mop_eg",
            "hfus",
        ]

    def enhance_descriptor(self, element_graph):

        elements_descriptors = []

        # TODO:This is a place for speed up:
        for node_nr, element in enumerate(element_graph.elements):
            descriptor = []
            al_info = Specie(symbol=element, source="cfid")
            for prop in self.chem_element_descriptor:
                descriptor.append(float(al_info.element_property(key=prop)))
            elements_descriptors.append(descriptor)

        elements_descriptors = torch.tensor(elements_descriptors)
        elements_descriptors = torch.tensor(elements_descriptors)
        # Insert the descriptor to the node graph
        element_graph.data.x = torch.cat((element_graph.data.x, elements_descriptors), dim=1)
        element_graph.node_descriptor.extend(self.chem_element_descriptor)

        return element_graph
