import numpy as np

def dipole_dipole_J(m1, m2, coords, idx, singlet=True):
    if not singlet:
        return 0.0
    
    mu_D = m1.td.transition_dipole()[idx]
    mu_A = m2.td.transition_dipole()[idx]

    if np.dot(mu_D, mu_A) > 0:
        mu_A = -mu_A

    R_vec = np.array(coords) / 0.529177
    R_mag = np.linalg.norm(R_vec)
    R_u = R_vec / R_mag
    j_dd = (3 * (np.dot(mu_D, R_u)*np.dot(mu_A, R_u)) - np.dot(mu_D, mu_A)) / (R_mag**3)
    
    return j_dd * 27.2114
   
