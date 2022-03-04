import geom
import numpy as np

rtol_def = 1e-4
atol_def = 1e-4

def test_rot_mat_z():
    theta = np.radians(45)
    expected = np.array([
        [0.7071,-0.7071,0.0   ],
        [0.7071, 0.7071,0.0   ],
        [0.0,    0.0,   1.0000]
        ])
    actual = geom.z_rot_matrix(theta)
    np.testing.assert_allclose(actual, expected, rtol=rtol_def, atol=atol_def, equal_nan=False)

def test_place_init():
    r1 = r2 = 0.9
    theta = np.radians(104.5)
    order = [1, 0]
    expected = np.array([
        [0.    , 0.    , 0.0],
        [0.9   , 0.    , 0.0],
        [1.1253, 0.8713, 0.0]
        ])
    
    actual = geom.place_init(order, r1, r2, theta)

    np.testing.assert_allclose(actual, expected, rtol=rtol_def, atol=atol_def, equal_nan=False)

def test_place_init_2():
    r1 = r2 = 1.089
    theta = np.radians(109.4710)
    order = [0, 1]
    expected = np.array([
        [0.     , 0.      , 0.0],
        [1.089  , 0.      , 0.0],
        [-0.363 , -1.02672, 0.0]
        ])
    
    actual = geom.place_init(order, r1, r2, theta)

    np.testing.assert_allclose(actual, expected, rtol=rtol_def, atol=atol_def, equal_nan=False)

def test_place_init_3():
    r1 = 0.9
    r2 = 1.4
    theta = np.radians(105.0)
    order = [1, 0]
    expected = np.array([
        [   0.0,   0.0,   0.0],
        [   0.9,   0.0,   0.0],
        [1.2623,1.3523,   0.0]
        ])

    actual = geom.place_init(order, r1, r2, theta)

    np.testing.assert_allclose(actual, expected, rtol=rtol_def, atol=atol_def, equal_nan=False)

def test_nerf():
    r = 1.089
    theta = np.radians(109.471)
    phi = np.radians(120.000)

    coords = np.array([
        [0.     , 0.      , 0.0],
        [1.089  , 0.      , 0.0],
        [-0.363 , -1.02672, 0.0]
        ])
    
    expected = np.array([-0.363000, 0.513360, 0.889165])
    actual = geom.nerf(coords[0], coords[1], coords[2], r, theta, phi)

    np.testing.assert_allclose(actual, expected, rtol=rtol_def, atol=atol_def, equal_nan=False)

def test_nerf_2():
    r = 1.089
    theta = np.radians(109.471)
    phi = np.radians(120.000)

    coords = np.array([
       [ 0.000000,   0.000000,   0.000000],
       [ 0.000000,   0.000000,   1.089000],
       [ 1.026719,   0.000000,  -0.363000]
        ])
    
    expected = np.array([-0.513360,  -0.889165,  -0.363000])
    actual = geom.nerf(coords[0], coords[1], coords[2], r, theta, phi)
    
    np.testing.assert_allclose(actual, expected, rtol=rtol_def, atol=atol_def, equal_nan=False)

def test_nerf_3():
    r = 1.089
    theta = np.radians(109.471)
    phi = np.radians(-120.000)

    coords = np.array([
       [ 0.000000,   0.000000,   0.000000],
       [ 0.000000,   0.000000,   1.089000],
       [ 1.026719,   0.000000,  -0.363000],
       [-0.513360,  -0.889165,  -0.363000]
        ])
    
    expected = np.array([-0.513360,  0.889165,  -0.363000])
    actual = geom.nerf(coords[0], coords[1], coords[2], r, theta, phi)
    
    np.testing.assert_allclose(actual, expected, rtol=rtol_def, atol=atol_def, equal_nan=False)
        
def test_nerf_4():
    r = 0.9
    theta = 105.0
    phi = 120.0
    coords = np.array([
        [   0.0,   0.0,   0.0],
        [   0.9,   0.0,   0.0],
        [1.2623,1.3523,   0.0]
        ])
    
    expected = np.array([-0.21686, -0.71117, 0.50715])
    actual = geom.nerf(coords[0], coords[1], coords[2], r, theta, phi)
    np.testing.assert_allclose(actual, expected, rtol=rtol_def, atol=atol_def, equal_nan=False)

def test_angle():
    p1 = np.array([0.,-1., 0.])
    p2 = np.array([0., 0., 0.])
    p3 = np.array([0., 0., 1.])
    
    expected = 90 # degree
    actual = np.degrees(geom.angle(p1, p2, p3))
    
    np.testing.assert_allclose(expected, actual, rtol=rtol_def, atol=atol_def, equal_nan=False)
    
def test_dihedral_angle_1():
    p1 = np.array([3.485, 62.378, -48.884])
    p2 = np.array([3.963, 62.400, -50.103])
    p3 = np.array([3.188, 62.567, -51.311])
    p4 = np.array([3.337, 61.226, -51.970])

    expected = -112.75 # degree
    actual = np.degrees(geom.dihedral_angle(p1, p2, p3, p4))
    
    print(expected, actual)
    
    np.testing.assert_allclose(expected, actual, rtol=rtol_def, atol=atol_def, equal_nan=False)
    
def test_dihedral_angle_2():
    p1 = np.array([0.,-1., 0.])
    p2 = np.array([0., 0., 0.])
    p3 = np.array([0., 0., 1.])
    p4 = np.array([1., 0., 1.])

    expected = 90 # degree
    actual = np.degrees(geom.dihedral_angle(p1, p2, p3, p4))
    
    print(expected, actual)
    
    np.testing.assert_allclose(expected, actual, rtol=rtol_def, atol=atol_def, equal_nan=False)
    
