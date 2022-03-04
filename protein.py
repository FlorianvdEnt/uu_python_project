#!/usr/bin/env python
import numpy as np
from collections import defaultdict
import geom

#ATOM      1  N   ALA X   1     -19.370  22.945  14.343  0.00  0.00      XP1  N 
# indexes:
#  0     1       2     3     4        5        6     7         8
# atom, index, space, name, index, resname, space, resindex, random,
# 9 10 11   12         13    14   15       16       17
# x, y, z, occupancy, beta, gap, segment, element, randomstuff
# lengths of pdbfile columns
pdb_col_lengths = [6, 5, 1, 4, 1, 4, 1, 4, 4, 8, 8, 8, 6, 6, 6, 4, 2, 100]

class Protein():

    def __init__(self, file):
        temp = defaultdict(list)
        self.lead = []
        self.atm_names = defaultdict(list)
        self.elements = defaultdict(list)
        self.res_names = {} 

        with open(file, "r") as f:
            coord_index = 1
            for line in f:
#                 line = line.strip('\n')
                if not line.startswith("ATOM") and not readHET:
                    continue
        
                col_i = 0
                lf = []
                for l in pdb_col_lengths:
                    lf.append(line[col_i:col_i+l])
                    col_i += l
                
                resi = int(lf[7].strip())
                temp[resi].append(np.array(list(map(float, (lf[9], lf[10], lf[11])))))
                self.lead.append(lf[0])
                if lf[3].strip() == 'OXT':
                    self.atm_names[resi].append('O')
                else:
                    self.atm_names[resi].append(lf[3].strip())
                self.elements[resi].append(lf[16].strip())
                self.res_names[resi] = lf[5].strip()
        
        self.res_indices = []
        self.coords = {}
        
        for resi, coords in temp.items():
            self.res_indices.append(resi)
            self.coords[resi] = np.stack(coords)
        
    @property
    def num_residues(self):
        return len(self.res_names.keys())

    @property
    def num_atoms(self):
        num = 0
        for res in self.coords.values():
            num += len(res)
        return num
    
    def write_xyz(self, file):
        atoms = []
        for res in self.res_indices:
            for i, coord in enumerate(self.coords[res]):
                atoms.append(f'{self.elements[res][i]} {coord[0]} {coord[1]} {coord[2]}\n')
        
        with open(file, 'w') as f:
            f.write(str(len(atoms)))
            f.write('\n')
            f.writelines(atoms)
    
    @property
    def pdb_str(self):
        atom_count = 1
        c_formatter = "{:.3f}".format
        pdb_str = []

        for res in self.res_indices:
            for atm, atm_name in enumerate(self.atm_names[res]):
                res_name = self.res_names[res]
                coords = self.coords[res][atm]
                element = self.elements[res][atm]

                line =  f'{"ATOM".ljust(6)} ' # ATOM
                line += f'{str(atom_count).rjust(4)}' # atom index

                # atom name
                if len(atm_name) == 4:
                    line += f' {atm_name.ljust(4)}'
                else:
                    line += f'  {atm_name.ljust(3)}'

                line += f'{res_name.rjust(4)} {"A"}{str(res).rjust(4)}{" "*4}' # residue name, chain index and residue index
                line += "".join([c_formatter(x).rjust(8) for x in coords]) # coordinates
                line += f'{"0.00".rjust(6)}{"0.00".rjust(6)}' # beta and temp factors
                line += f'{" "*6}{" "*4}{element.rjust(2)}' # spacing and element

                pdb_str.append(line)

                atom_count += 1
        return pdb_str
    
    def write_pdb(self, file):
        with open(file, 'w') as pdb_file:
            for line in self.pdb_str:
                pdb_file.write(line + '\n')
    
    def __str__(self):
        return f"Protein({self.num_residues} residues, {self.num_atoms} atoms)"
    
    def phi(self, index):
        if index == self.res_indices[0]:
            return np.radians(-60)
        
        C_1 = self.atm_names[index-1].index('C')       
        N = self.atm_names[index].index('N')
        CA = self.atm_names[index].index('CA')
        C = self.atm_names[index].index('C')
        
        C_1 = self.coords[index-1][C_1]   
        N = self.coords[index][N]
        CA = self.coords[index][CA]
        C = self.coords[index][C]
        
        return geom.dihedral_angle(C_1, N, CA, C)
    
    def psi(self, index):
        if index == self.res_indices[0]:
            return np.radians(-60)
              
        N = self.atm_names[index].index('N')
        CA = self.atm_names[index].index('CA')
        C = self.atm_names[index].index('C')
        C_1 = self.atm_names[index-1].index('C') 
        
        C_1 = self.coords[index-1][C_1]   
        N = self.coords[index][N]
        CA = self.coords[index][CA]
        C = self.coords[index][C]
        
        return geom.dihedral_angle(C_1, N, CA, C)
            

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

