#!/usr/bin/env python
from protein import Protein
from forcefield import FF
from rotamere import RotamereBuilder
import numpy as np
import copy
import sys


args = sys.argv

if len(args) != 4:
    print('Usage: mutate.py target.pdb residue_index target_residue')

s = Protein(args[1])
outfile = args[1].strip('.pdb') + f'_{args[2]}{args[3]}.pdb'
ff = FF()

to_mutate = int(args[2]) 
mutate_to = args[3]

phi = s.phi(to_mutate)
psi = s.psi(to_mutate)

coords = s.coords[to_mutate]
atm_names = s.atm_names[to_mutate]

rlib = RotamereBuilder()

new_residues = rlib.build_rotameres(mutate_to, coords, atm_names, phi, psi)
scores = []
for i, res in enumerate(new_residues):
    s.res_names[to_mutate] = mutate_to
    s.coords[to_mutate] = res[0]
    s.atm_names[to_mutate] = res[1]
    s.elements[to_mutate] = [a[0] for a in s.atm_names[to_mutate]]
    scores.append(ff.calc_energy(s))

best = np.min(scores)

s.res_names[to_mutate] = mutate_to
s.coords[to_mutate] = new_residues[i][0]
s.atm_names[to_mutate] = new_residues[i][1]
s.elements[to_mutate] = [a[0] for a in s.atm_names[to_mutate]]

s.write_pdb(outfile)

