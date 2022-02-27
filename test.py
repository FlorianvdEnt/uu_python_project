from protein import Protein
from forcefield import FF


s = Protein('sample.pdb')
ff = FF()

ff.calc_energy(s)


