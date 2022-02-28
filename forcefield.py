"""
Produces an object with the following structure:
FF.lj['ALA-CA'] = (sigma, epsilon)
"""
import xml.etree.ElementTree as ET
import sys
import geom, energy

class FF():
    
    def __init__(self):
#         import ipdb; ipdb.set_trace(context=5)
        tree = ET.parse('faspr.xml')
        self._root = tree.getroot()
        
        self.residues = {} 
        self.lj = {}
        
        for res in self._root.find('Residues'):
            resname = res.attrib['name']
            self.residues[resname] = set() 
            
            for atom in list(res):
                atmname = atom.attrib['name']
                atmtype = atom.attrib['type']
                self.residues[resname].add( atom.attrib['name'] )
                
                for node in self._root.find('NonbondedForce'):
                    if node.attrib['type'] == atmtype:
                        sigma = float(node.attrib['sigma'])
                        epsilon = float(node.attrib['epsilon'])
                        self.lj[f'{resname}-{atmname}'] = (sigma, epsilon)
    

    def eval_prot(self, prot):
        missing_res_types = []
        missing_atm_types = []
        
        for i, resi in enumerate(prot.res_indices):
        
            resn = prot.res_names[resi]
            
            if not resn in self.residues:
                if not resn in missing_res_types:
                    missing_res_types.append(resn)
                continue
            
            atm_names = prot.atm_names[resi]
            for atm_name in atm_names:
                if not f'{resn}-{atm_name}' in self.lj:
                    missing_atm_types.append(f'{resn}-{atm_name}')
        
        if missing_res_types:
            print(f'WARNING: missing residue type(s): {" ".join(missing_res_types)}')
        if missing_atm_types:
            print(f'WARNING: missing atom type(s): {" ".join(missing_atm_types)}')
            
    def _inter_res_energy(self, resi, resj, prot):
        res_name_i = prot.res_names[resi]
        res_name_j = prot.res_names[resj]
        
        res_energy = 0.0
        
        for atmi, atm_name_i in enumerate(prot.atm_names[resi]):
            for atmj, atm_name_j in enumerate(prot.atm_names[resj]):
        
                sig_i, eps_i = self.lj[f'{res_name_i}-{atm_name_i}']
                sig_j, eps_j = self.lj[f'{res_name_j}-{atm_name_j}']

                coord1 = prot.coords[resi][atmi]
                coord2 = prot.coords[resj][atmj]
                
                r = geom.dist(coord1, coord2)
                res_energy += energy.faspr_vdw(sig_i, eps_i, sig_j, eps_j, r)
        
        return res_energy
                
    def calc_energy(self, prot):
        tot_energy = 0.0
        
        for i, resi in enumerate(prot.res_indices):
            resni = prot.res_names[resi]
            
            for j, resj in enumerate(prot.res_indices):
                resnj = prot.res_names[resj]
                
                if j <= i:
                    continue
                
                if resni in self.residues and resnj in self.residues:
                    tot_energy += self._inter_res_energy(resi, resj, prot)
                    
        return tot_energy

