"""
Produces an object with the following structure:
FF.lj['ALA-CA'] = (sigma, epsilon)
"""
import xml.etree.ElementTree as ET


class FF():
    
    def __init__(self):
        tree = ET.parse('faspr.xml')
        self._root = tree.getroot()
        
        self.residues = {} 
        self.lj = {}
        
        for res in self._root.find('Residues'):
            resname = res.attrib['name']
            self.residues[resname] = set() 
            
            for atom in list(res):
                atmname = atom.attrib['name']
                self.residues[resname].add( atom.attrib['name'] )
                
                for node in self._root.find('NonbondedForce'):
                    if node.attrib['type'] == atmname:
                        sigma = node.attrib['sigma']
                        epsilon = node.attrib['epsilon']
                        self.lj[f'{resname}-{atmname}'] = (sigma, epsilon)

    def calc_energy(self, prot):
        for resi, resn in prot.res_names.items():
            print(resi, resn)
            atm_indices = prot.res_atm_indices[resi]
            for atmi in atm_indices:
                print(f'\t{prot.atm_names[int(atmi)]}')



