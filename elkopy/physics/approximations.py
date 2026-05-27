import numpy as np
from pyscf import gto, scf, tdscf

def dipole_dipole_J(m1, m2, coords, idx=[1, 1], singlet=True):
    if not singlet:
        return 0.0
    
    mu_D = m1.td.transition_dipole()[idx[0] - 1]
    mu_A = m2.td.transition_dipole()[idx[1] - 1]

    if np.dot(mu_D, mu_A) > 0:
        mu_A = -mu_A

    R_vec = np.array(coords) / 0.529177
    R_mag = np.linalg.norm(R_vec)
    R_u = R_vec / R_mag
    j_dd = (3 * (np.dot(mu_D, R_u)*np.dot(mu_A, R_u)) - np.dot(mu_D, mu_A)) / (R_mag**3)
    
    return j_dd * 27.2114
   
def full_QM(m1, m2_moved, singlet=True, xc=None):
    mol_dimer = gto.mole.conc_mol(m1.mol, m2_moved.mol)
    
    if xc is None:
        mf_dimer = scf.RHF(mol_dimer).run(verbose=0)
    else:
        from pyscf import dft
        mf_dimer = dft.RKS(mol_dimer)
        mf_dimer.xc = xc
        mf_dimer.run(verbose=0)
        
    td_dimer = tdscf.TDA(mf_dimer)
    td_dimer.singlet = singlet
    td_dimer.run(nstates=2, verbose=0)
    
    return (td_dimer.e[1] - td_dimer.e[0]) * 27.2114 / 2.0