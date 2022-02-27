#!/usr/bin/env python
import numpy as np
from collections import defaultdict

#ATOM      1  N   ALA X   1     -19.370  22.945  14.343  0.00  0.00      XP1  N 
# indexes:
#  0     1       2     3     4        5        6     7         8
# atom, index, space, name, index, resname, space, resindex, random,
# 9 10 11   12         13   14        15       16
# x, y, z, occupancy, beta, segment, element, randomstuff
# lengths of pdbfile columns
pdb_col_lengths = [6, 5, 1, 4, 1, 4, 1, 4, 4, 8, 8, 8, 6, 6, 4, 2, 100]

class Protein():

    def __init__(self, file):
        self.coords = []
        self.lead = []
        self.atm_names = []
        self.element = []
        self.res_atm_indices = defaultdict(list)
        self.res_names = {} 

        with open(file, "r") as f:
            # Number of atoms and comment
            coord_index = 1
            for line in f:
                if not line.startswith("ATOM") and not readHET:
                    continue
        
                col_i = 0
                lf = []
                for l in pdb_col_lengths:
                    lf.append(line[col_i:col_i+l])
                    col_i += l
                
                self.coords.append(np.array(list(map(float, (lf[9], lf[10], lf[11])))))
                self.lead.append(lf[0])
                self.atm_names.append(lf[3].strip())
                self.element.append(lf[15].strip())
                self.res_atm_indices[int(lf[7].strip())].append(coord_index)
                self.res_names[int(lf[7].strip())] = lf[5].strip()
                coord_index += 1

        for key in self.res_atm_indices:
            self.res_atm_indices[key] = np.array(self.res_atm_indices[key])
        self.coords = np.stack(self.coords)

    @property
    def num_residues(self):
        return len(self.res_atm_indices)

    @property
    def num_atoms(self):
        return len(self.coords)


if __name__ == '__main__':
    s = Protein('sample.pdb')
    print(s.num_residues)
    i = 42
    print(f'info about residue {i}')
    for atom in s.res_atm_indices[i]:
        print(atom, s.atm_names[atom])
    atom_types = sorted(list(set(s.atm_names)))
    for at in atom_types:
        print(f'\t\'{at}\':\t(),')


