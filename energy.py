import math


def faspr_vdw(sig_i, eps_i, sig_j, eps_j, r):
    sig_ij = sig_i + sig_j
    r_star = r / sig_ij
    
    if r_star >= 1.9:
        return 0
    elif 1.9 > r_star >= 1:
        r_6 = (1/r_star)**2
        eps_ij = math.sqrt(eps_i * eps_j)
        return eps_ij * (r_6**2 - r_6)
    elif 1 > r_star >= 0.015:
        return 10 * ((r_star - 1) / (0.015 - 1))
    else:
        return 10
    
