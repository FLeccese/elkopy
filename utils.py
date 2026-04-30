import numpy as np
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def read_xyz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} non trovato.")
    return path

def dipole_dipole_J(m1, m2, coords, idx, singlet=True):
    if not singlet:
        return 0.0

    mu_D = m1.td.transition_dipole()[idx]
    mu_A = m2.td.transition_dipole()[idx]
    R_vec = np.array(coords) / 0.529177
    R_mag = np.linalg.norm(R_vec)
    R_u = R_vec / R_mag
    j_dd = (3 * (np.dot(mu_D, R_u)*np.dot(mu_A, R_u)) - np.dot(mu_D, mu_A)) / (R_mag**3)
    return j_dd * 27.2114

def print_row(d, jc, jk, jp, jd, jtot):
    print(f"{d:8.2f} | {jc:10.4e} | {jk:10.4e} | {jp:10.4e} | {jd:10.4e} | {jtot:10.4e}")
