#!/usr/bin/env python
import numpy as np
import math

# Some general functions that where needed
def norm(vec):
    return math.sqrt(sum(vec*vec))

def z_rot_matrix(theta):
    return np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]])


# Cartesian to internal stuff
def dist(vec1, vec2):
    return norm(vec1-vec2)

def angle(vec1, vec2, vec3):
    vec2_1 = vec1 - vec2
    vec2_3 = vec3 - vec2
    
    dot = np.dot(vec2_1, vec2_3)
    prod = norm(vec2_3) * norm(vec2_1)
    
    return math.acos(dot/prod)

def dihedral_angle(vec1, vec2, vec3, vec4):
    """
    Calculates the dihedral angle given 4 points.
    Based on formula taken from: https://en.wikipedia.org/wiki/Dihedral_angle
    """
    
    u1 = vec2 - vec1
    u2 = vec3 - vec2
    u3 = vec4 - vec3
    
    A = np.dot(norm(u2) * u1, np.cross(u2, u3))
    B = np.dot(np.cross(u1, u2), np.cross(u2, u3))
       
    return math.atan2(A, B)

# Cartesian to internal (NeRF functions)
def calc_srf(r, theta, phi):
    x = r * math.cos(theta)
    y = r * math.sin(theta) * math.cos(phi)
    z = r * math.sin(theta) * math.sin(phi)
    return np.array([x, y, z])

def nerf_rot_matrix(a, b, c):
    ab = b-a
    bc = c-b
    bc_norm = bc / norm(bc)
    n_plane = np.cross(ab, bc_norm)
    n_plane = n_plane / norm(n_plane)
    col_1 = - bc_norm
    col_2 = np.cross(n_plane, bc_norm)
    col_3 = n_plane
    rot_matrix = np.column_stack([col_1, col_2, col_3])
    return rot_matrix

def nerf(p1, p2, p3, r, theta, phi):
    srf = calc_srf(r, theta, phi)
    rot_matrix = nerf_rot_matrix(p3, p2, p1)
    return p1 + np.matmul(rot_matrix, srf)
 

# Place initial atoms
def place_init(order, r1, r2, theta):
    coords = np.array([
            [0.0, 0.0, 0.0],
            [r1, 0.0, 0.0],
            [0.0, 0.0, 0.0]
            ])
    ab = coords[order[0]] - coords[order[1]]
    ab_norm = ab / norm(ab)
    rot_matrix = z_rot_matrix(np.pi - theta)
    c = np.matmul(rot_matrix, r2 * ab_norm) 
    coords[2] += (c + coords[order[0]] ) 
    return coords
    
