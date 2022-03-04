import numpy as np
import geom
import copy
from collections import defaultdict

rot_lib_file = 'SimpleOpt2-5_ALL.bbdep.rotamers.lib'
# 0  1    2    3        4  5  6  7  8         9       10      11      12        13      14      15      16
# T  Phi  Psi  Count    r1 r2 r3 r4 Probabil  chi1Val chi2Val chi3Val chi4Val   chi1Sig chi2Sig chi3Sig chi4Sig

modeled_aas = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

default_rotameres = [
    [0, 'ALA', 'A', 0, ['CB'], [[1.53, 110.5, -122.5]], [[2, 0, 1, 4]]], 
    [1, 'ARG', 'R', 4, ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'], [[1.53, 110.5, -122.5], [1.52, 114.1, 181.0], [1.52, 111.5, 181.0], [1.461, 112.0, 181.0], [1.33, 124.5, 181.0], [1.326, 120.0, 0.0], [1.326, 120.0, 180.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [9, 7, 8, 10]]], 
    [2, 'ASN', 'N', 2, ['CB', 'CG', 'OD1', 'ND2'], [[1.53, 110.5, -122.5], [1.516, 112.7, 181.0], [1.231, 120.8, 181.0], [1.328, 116.5, 180.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6], [6, 4, 5, 7]]], 
    [3, 'ASP', 'D', 2, ['CB', 'CG', 'OD1', 'OD2'], [[1.53, 110.5, -122.5], [1.516, 112.7, 181.0], [1.25, 118.5, 181.0], [1.25, 118.5, 180.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6], [6, 4, 5, 7]]], 
    [4, 'CYS', 'C', 1, ['CB', 'SG'], [[1.53, 110.5, -122.5], [1.807, 114.0, 181.0]], [[2, 0, 1, 4], [0, 1, 4, 5]]], 
    [5, 'GLN', 'Q', 3, ['CB', 'CG', 'CD', 'OE1', 'NE2'], [[1.53, 110.5, -122.5], [1.52, 114.1, 181.0], [1.516, 112.7, 181.0], [1.231, 120.8, 181.0], [1.328, 116.5, 180.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7], [7, 5, 6, 8]]], 
    [6, 'GLU', 'E', 3, ['CB', 'CG', 'CD', 'OE1', 'OE2'], [[1.53, 110.5, -122.5], [1.52, 114.1, 181.0], [1.516, 112.7, 181.0], [1.25, 118.5, 181.0], [1.25, 118.5, 180.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7], [7, 5, 6, 8]]], 
    [7, 'HIS', 'H', 2, ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'], [[1.53, 110.5, -122.5], [1.5, 113.8, 181.0], [1.378, 122.7, 181.0], [1.354, 131.0, 180.0], [1.32, 109.2, 180.0], [1.374, 107.2, 180.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6], [6, 4, 5, 7], [4, 5, 6, 8], [4, 5, 7, 9]]], 
    [8, 'ILE', 'I', 2, ['CB', 'CG1', 'CG2', 'CD1'], [[1.546, 111.5, -122.5], [1.53, 110.3, 181.0], [1.521, 110.5, -122.6], [1.516, 114.0, 181.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [5, 1, 4, 6], [1, 4, 5, 7]]], 
    [9, 'LEU', 'L', 2, ['CB', 'CG', 'CD1', 'CD2'], [[1.53, 110.5, -122.5], [1.53, 116.3, 181.0], [1.521, 110.5, 181.0], [1.521, 110.5, 122.6]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6], [6, 4, 5, 7]]], 
    [10, 'LYS', 'K', 4, ['CB', 'CG', 'CD', 'CE', 'NZ'], [[1.53, 110.5, -122.5], [1.52, 114.1, 181.0], [1.52, 111.5, 181.0], [1.52, 111.5, 181.0], [1.489, 112.0, 181.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]], 
    [11, 'MET', 'M', 3, ['CB', 'CG', 'SD', 'CE'], [[1.53, 110.5, -122.5], [1.52, 114.1, 181.0], [1.807, 112.7, 181.0], [1.789, 100.8, 181.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7]]], 
    [12, 'PHE', 'F', 2, ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], [[1.53, 110.5, -122.5], [1.5, 113.8, 181.0], [1.391, 120.7, 181.0], [1.391, 120.7, 180.0], [1.393, 120.7, 180.0], [1.393, 120.7, 180.0], [1.39, 120.0, 0.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6], [6, 4, 5, 7], [4, 5, 6, 8], [4, 5, 7, 9], [5, 6, 8, 10]]], 
    [13, 'PRO', 'P', 2, ['CB', 'CG', 'CD'], [[1.53, 103.2, -120.0], [1.495, 104.5, 181.0], [1.507, 105.5, 181.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6]]], 
    [14, 'SER', 'S', 1, ['CB', 'OG'], [[1.53, 110.5, -122.5], [1.417, 110.8, 181.0]], [[2, 0, 1, 4], [0, 1, 4, 5]]], 
    [15, 'THR', 'T', 1, ['CB', 'OG1', 'CG2'], [[1.542, 111.5, -122.0], [1.433, 109.5, 181.0], [1.521, 110.5, -120.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [5, 1, 4, 6]]], 
    [16, 'TRP', 'W', 2, ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'], [[1.53, 110.5, -122.5], [1.5, 113.8, 181.0], [1.365, 126.9, 181.0], [1.433, 126.7, 180.0], [1.375, 110.2, 180.0], [1.413, 107.2, 180.0], [1.4, 133.9, 0.0], [1.399, 122.4, 180.0], [1.392, 118.7, 180.0], [1.372, 117.5, 0.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6], [6, 4, 5, 7], [4, 5, 6, 8], [4, 5, 7, 9], [4, 5, 7, 10], [5, 7, 9, 11], [5, 7, 10, 12], [7, 9, 11, 13]]], 
    [17, 'TYR', 'Y', 2, ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'], [[1.53, 110.5, -122.5], [1.511, 113.8, 181.0], [1.394, 120.8, 181.0], [1.394, 120.8, 180.0], [1.392, 121.1, 180.0], [1.392, 121.1, 180.0], [1.385, 119.5, 0.0], [1.376, 119.7, 180.0]], [[2, 0, 1, 4], [0, 1, 4, 5], [1, 4, 5, 6], [6, 4, 5, 7], [4, 5, 6, 8], [4, 5, 7, 9], [5, 6, 8, 10], [6, 8, 10, 11]]], 
    [18, 'VAL', 'V', 1, ['CB', 'CG1', 'CG2'], [[1.546, 111.5, -122.5], [1.521, 110.5, 181.0], [1.521, 110.5, 122.6]], [[2, 0, 1, 4], [0, 1, 4, 5], [5, 1, 4, 6]]]
]


class RotamereLib():
    
    def __init__(self, aa):
        aa_line = None
        for line in default_rotameres:
            if line[1] == aa:
                aa_line = line
        
        if not aa_line:
            return None
        
        self.num = aa_line[0]
        self.type = aa_line[1]
        self.num_chis = aa_line[3]
        self.atom_names = aa_line[4]
        self.default_internals = []
        
        for ins in np.array(aa_line[5]):
            ins_rad = [ins[0], np.radians(ins[1]), np.radians(ins[2])]
            self.default_internals.append(np.array(ins_rad))
        
        if self.num_chis == 0:
            self.internals = [self.default_internals]
            self.prob = 1
        else:
            self.internals = {}
            self.prob = {}
            with open(rot_lib_file) as file:
                for line in file:
                    if line.startswith('#'):
                        continue
                    elif not line.startswith(self.type):
                        continue
                    else:
                        lf = line.split()
                        bin1 = int( float(lf[1]) /10)
                        bin2 = int( float(lf[2]) /10)
                        chis = lf[9:14]
                        prob = float(lf[8])
                        chis = chis[:self.num_chis]
                        
                        if prob < 0.01:
                            continue
                        
                        new_internals = copy.deepcopy(self.default_internals)
                        for i in range(self.num_chis):
                            new_internals[i+1][2] = np.radians(float(chis[i]))
                        
                        if bin1 in self.prob:
                            if bin2 in self.prob[bin1]:
                                if sum( self.prob[bin1][bin2] ) > 0.98:
                                    continue
                                self.prob[bin1][bin2].append(prob)
                                self.internals[bin1][bin2].append(new_internals)
                            else:
                                self.prob[bin1][bin2] = [prob]
                                self.internals[bin1][bin2] = [new_internals]
                                
                        else:
                            self.prob[bin1] = {bin2: [prob]}
                            self.internals[bin1] = {bin2: [new_internals]}
                            
            for k1 in self.prob:
                for k2, probs in self.prob[k1].items():
                    print(k1, k2, len(probs), sum(probs))
                    break
                break
        
        self.indices = np.array(aa_line[6])    
            
    def build_rot(self, backbone, internals, indices):
        new_coords = backbone.copy()
        print( new_coords )
        
        for i, int_coords in enumerate(internals):
            int_indic = indices[i][:3]

            p3 = new_coords[int_indic[0]]
            p2 = new_coords[int_indic[1]]
            p1 = new_coords[int_indic[2]]

            r = int_coords[0]
            theta = int_coords[1]
            phi = int_coords[2]

            new_coords.append(geom.nerf(p1, p2, p3, r, theta, phi))

        return np.stack(new_coords), ['N', 'CA', 'C', 'O'] + self.atom_names
            
    def build_lib(self, backbone, bb_phi, bb_psi):
        if self.num_chis == 0:
            new_coords, new_atm_names = self.build_rot(backbone, self.internals[0], self.indices)
            return [[new_coords, new_atm_names]]
        else:
            bin1 = int((np.degrees( bb_phi ) + 5) / 10)
            bin2 = int((np.degrees( bb_psi ) + 5) / 10)
            
            if bin1 > 18:
                bin1 -= 36
            if bin2 > 18:
                bin2 -= 36
            if bin1 < -18:
                bin1 += 36
            if bin2 < -18:
                bin2 += 36
            
            build_rotameres = []
            internals_array = self.internals[bin1][bin2]
            
            for internals in internals_array:
                new_coords, new_atm_names = self.build_rot(backbone, internals, self.indices)
                build_rotameres.append([new_coords, new_atm_names])
            
            return build_rotameres

    
class RotamereBuilder(): 
    
    def __init__(self):
        self.residues = {}
        for aa in modeled_aas:
            self.residues[aa] = RotamereLib(aa)
        
    def build_rotameres(self, aa, coords, atm_names, phi, psi):
        backbone = []
        for n in ['N', 'CA', 'C', 'O']:
            for i, atm_name in enumerate(atm_names):
                if n == atm_name:
                    backbone.append(coords[i])
                    
        return self.residues[aa].build_lib(backbone, phi, psi)
    
        