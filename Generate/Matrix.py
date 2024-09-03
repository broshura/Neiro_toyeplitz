# Calculating geometry matrix M
import json

import numpy as np
from numpy import sqrt, cos, sin, pi
from scipy import integrate
from scipy import special
from tqdm import tqdm
from Ring_Class import Ring

np.set_printoptions(linewidth=np.inf)

K = special.ellipk  # Сomplete elliptic integral of the first kind
E = special.ellipe  # Сomplete elliptic integral of the second kind

mu_0 = 4 * pi * 10 ** -7

# Computing for parallel-oriented rings
def L_parallel(dx, dy, dz, r1, r2, width = 0):

    # Define function to integrate over first defined parameter

    def dl(alpha, dx, dy, dz, r_1, r_2):
        db = sqrt(dx ** 2 + dy ** 2)
        dp = sqrt(r_2 ** 2 + db ** 2 + 2 * r_2 * db * cos(alpha))
        kappa_sq = 4 * r_1 * dp / ((dp + r_1) ** 2 + dz ** 2)
        kappa = sqrt(kappa_sq)
        A = 1/(2*pi)*sqrt(r_1/dp) * ((2/kappa - kappa) * K(kappa_sq) - 2 * E(kappa_sq)/kappa)
        return A * r_2 * (r_2 + db * cos(alpha))/dp

    #Considering stripe width

    if r1 == r2 and width:
        R = r1 + width / 2
        r = r1 - width / 2
        L_1, err_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r, r))
        L_2, err_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r, R))
        L_3, err_3 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, R, R))
        L = (L_1 + 2*L_2 + L_3)/4
    elif width:
        id_r1 = r1 == min(r1, r2)
        id_r2 = r2 == min(r1, r2)
        L_1, err_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1 + id_r1 * width/2, r2 + id_r2*width/2))
        L_2, err_2 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1 - id_r1 * width/2, r2 - id_r2*width/2))
        return (L_1 + L_2)/2
    else:
        L, err = integrate.quad(dl, 0, 2 * pi, args= (dx, dy, dz, r1, r2))
    return L

# Computing for orthogonal-oriented rings

def L_orthogonal(dx, dy ,dz, r1, r2, width):
    def dl(alpha, dx, dy, dz, r_1, r_2):
        dp = sqrt((dx - r_2 * sin(alpha)) ** 2 + dy ** 2)
        kappa_sq = 4 * r_1 * dp / ((dp + r_1) ** 2 + (dz - r_2 * cos(alpha)) ** 2)
        kappa = sqrt(kappa_sq) + 10 ** -7
        A = 1 / (2 * pi) * sqrt(r_1 / (dp + 10 ** -7)) * ((2 / kappa - kappa) * K(kappa_sq) - 2 * E(kappa_sq) / kappa)
        return A * r_2 * dy * cos(alpha) / (dp + 10 ** -7)

    # Considering stripe width

    if r1 == r2 and width:
        R = r1 + width / 2
        r = r1 - width / 2
        L_1, err_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r, r))
        L_2, err_2 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r, R))
        L_3, err_3 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, R, R))
        L = (L_1 + 2 * L_2 + L_3) / 4
    elif width:
        id_r1 = r1 == min(r1, r2)
        id_r2 = r2 == min(r1, r2)
        L_1, err_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1 + id_r1 * width/2, r2 + id_r2*width/2))
        L_2, err_2 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1 - id_r1 * width/2, r2 - id_r2*width/2))
        return (L_1 + L_2)/2
    else:
        L, err = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1, r2))
    return L

# Computing for any pair

def Mnm(First_ring, Second_ring, Data = {}):

    dx = Second_ring.x - First_ring.x
    dy = Second_ring.y - First_ring.y
    dz = Second_ring.z - First_ring.z
    r1 = First_ring.r
    r2 = Second_ring.r
    w = First_ring.w * 0.7

    # To avoid calculating integrals with same params each time
    # there is a dictionary with all parameters and values

    id_1 = f"{dx} {dy} {dz} {r1} {r2} {First_ring.pos}{Second_ring.pos}"
    id_2 = f"{-dx} {-dy} {-dz} {r2} {r1} {Second_ring.pos}{First_ring.pos}"

    if id_1 in Data:
        return Data[id_1]
    elif id_2 in Data:
        return Data[id_2]

    
    elif dx == 0 and dy == 0 and dz == 0:
        Data[id_1] = 0
        return 0

    # Consider all types of parallel orientation and symmetry for x-z axes

    if First_ring.pos == Second_ring.pos:
        if First_ring.pos == "z":                           # Z-oriented rings
            l = L_parallel(dx, dy, dz, r1, r2, w)
        elif First_ring.pos == "y":                         # Y-oriented rings
            l = L_parallel(dx, -dz, dy, r1, r2, w)
        else:                                               # X-oriented rings
            l = L_parallel(-dz, dy, dx, r1, r2, w)

    # Consider all types of orthogonal orientation

    else:
        if First_ring.pos == "z":
            if Second_ring.pos == "y":                      # Z-Y oriented pair
                l = L_orthogonal(dx, dy, dz, r1, r2, w)
            else:                                           # Z-X oriented pair
                l = L_orthogonal((dy), (dx), dz, r1, r2, w)
        elif First_ring.pos == "y":
            if Second_ring.pos == "z":                      # Y-Z oriented pair
                l = L_orthogonal(dx, (dz), (dy), r1, r2, w)
            else:                                           # Y-X oriented pair
                l = L_orthogonal(-dz, (dx), (dy), r1, r2,  w)
        elif First_ring.pos == "x":
            if Second_ring.pos == "z":                      # X-Z oriented pair
                l = L_orthogonal(dy, (dz), (dx), r1, r2, w)
            else:                                           # X-Y oriented pair
                l = L_orthogonal((dz), dy, (dx), r1, r2, w)
    # Data[id_1], Data[id_2], Data[id_3], Data[id_4], Data[id_5], Data[id_6], Data[id_7], Data[id_8], Data[id_9], Data[id_10] = [l*mu_0] * 10
    Data[id_1], Data[id_2] = [l * mu_0] * 2
    return l * mu_0

# Calculating mutual inductance for each pair

def Z_diag(omega, R, L, C):
    return R - 1j * omega * L + 1j / (omega*C)

def MatrixStraight(rings, omega, R, L, C):
    M = np.zeros((len(rings), len(rings)))
    for i in range(len(rings)):
        for j in range(len(rings)):
            if i == j:
                M[i, j] = 0
            else:
                M[i, j] = Mnm(rings[i], rings[j])
    Z = np.diag([Z_diag(omega, R, L, C)] * len(rings)) + 1j * omega * M/5e-7
    return Z

def MatrixToep(rings, omega, R, L, C):
    M = np.zeros(len(rings), dtype = complex)
    for i in range(len(rings)):
        if i == 0:
            M[i] = Z_diag(omega, R, L, C)
        else:
            M[i] = 1j*omega*Mnm(rings[0], rings[i])/5e-7
    return M
        


# Generating random geometry for 1dimensional case
def generate(dim):
    # Random direction
    directions = ['nz', 'ny', 'nx']
    direction = np.random.choice(directions)

    # Random width, radius and step
    Radius = np.random.uniform(0.1, 1)
    width  = Radius * np.random.uniform(0.1, 0.9)
    a = 2*(Radius+width/2) + np.random.uniform(0.1, 3)
    rings = [Ring(i * a * (direction == 'nx'),
                  i * a * (direction == 'ny'),
                  i * a * (direction == 'nz'),
                  'z', Radius, 0, 0, 0, 0) for i in range(dim)]
    # Random resistance, inductance and capacitance
    R = np.random.uniform(0.01, 0.1)
    L = np.random.uniform(1, 10)
    C = np.random.uniform(1, 10)
    omega_0 = 1/sqrt(L*C)
    omega = omega_0 * np.random.uniform(0.9, 1.1) 
    M = MatrixToep(rings, omega, R, L, C)

    return M

import torch 

def generate_and_save(dim, number):
    M = torch.zeros((number, dim))
    C = torch.zeros((number, dim))
    for i in tqdm(range(number)):
        M[i] = torch.from_numpy(generate(dim).imag)
        # Делаем оптимальные циркулянты
        n_min_i = np.arange(dim)
        n_i = dim - n_min_i
        M_i_numpy = M[i].numpy()  # Преобразование тензора в numpy массив
        C[i] = torch.from_numpy((M_i_numpy * n_i + n_min_i * np.roll(M_i_numpy[::-1], 1)) / dim)
    torch.save(M, f"/Users/shuramakarenko/LocalDocs/Workspace/Neiro_toyeplitz/DATA/Matrix1024/Samples.pth")
    torch.save(C, f"/Users/shuramakarenko/LocalDocs/Workspace/Neiro_toyeplitz/DATA/Matrix1024/Targets.pth")


generate_and_save(1024, 10**4)