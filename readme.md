The plan is to make a script called mutate.py that takes as input a pdb file, a residue position and a target amino acid residue. The script will then output a pdb file with the optimal rotamere of that amino acid for that position.

Rotameres from the Dunbreck rotamere library can be used as candidate rotameres and the scoring function used in the faspr program will be used to select the most suitable rotamere.

Dunbreck rotamere library:
Shapovalov, Maxim V., and Roland L. Dunbrack Jr. "A smoothed backbone-dependent rotamer library for proteins derived from adaptive kernel density estimates and regressions." Structure 19.6 (2011): 844-858.

Faspr:
Xiaoqiang Huang, Robin Pearce, Yang Zhang. FASPR: an open-source tool for fast and accurate protein side-chain packing. Bioinformatics (2020) 36: 3758-3765. 
