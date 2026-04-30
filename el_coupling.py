import numpy as np
from pyscf import scf, gto
from pyscf.scf.jk import get_jk


class ElectronicCoupling:
	def __init__(self, m1, m2, rho_D, rho_A):
		self.m1 = m1
		self.m2 = m2
		self.mol_D = m1.mol
		self.mol_A = m2.mol
		self.rho_D = rho_D
		self.rho_A = rho_A
    
	def get_J(self, singlet=True):
		if not singlet:
			return 0.0

		vj = get_jk((self.mol_A, self.mol_A, self.mol_D, self.mol_D), self.rho_D, scripts='ijkl,ji->kl', aosym='s4')
		j_coul = 2* np.sum(vj * self.rho_A) * 27.2114 #eV
		
		return j_coul
		
	def get_K(self):
		vk = get_jk((self.mol_A, self.mol_D, self.mol_D, self.mol_A), self.rho_D, scripts='ijkl,jk->il', aosym='s1')
		j_exch = np.sum(vk * self.rho_A) * 27.2114 #eV
        
		return j_exch

	def get_P_term(self):
        
        # 1. Overlap integrals (Only between D and A) s_cross is a matrix
		s_cross = gto.mole.intor_cross('int1e_ovlp', self.mol_D, self.mol_A)
        
        # 2. HOMO (h) and LUMO (l) coefficients
		nocc = self.mol_D.nelectron // 2
		c_d_h = self.m1.mf.mo_coeff[:, nocc-1]
		c_d_l = self.m1.mf.mo_coeff[:, nocc]
		c_a_h = self.m2.mf.mo_coeff[:, nocc-1]
		c_a_l = self.m2.mf.mo_coeff[:, nocc]

        # 3. S_hh e S_ll (scalar numbers)
		s_ab = c_d_h @ s_cross @ c_a_h
		s_apbp = c_d_l @ s_cross @ c_a_l
		
		# 4. h term (Hamiltonian/Fock)
		#dm_core_d = self.m1.mf.make_rdm1() - np.outer(c_d_h, c_d_h) * 2 #core density matrix with all donor electrons except for HOMO
		dm_core_a = self.m2.mf.make_rdm1() - np.outer(c_a_h, c_a_h) #same for acceptor
		
		h_core_cross = gto.mole.intor_cross('int1e_kin', self.mol_D, self.mol_A) + gto.mole.intor_cross('int1e_nuc', self.mol_D, self.mol_A) #h_ij^sigma_pi
		
		vj_a, vk_a = get_jk((self.mol_D, self.mol_D, self.mol_A, self.mol_A), (dm_core_a,dm_core_a), scripts=('ijkl,kl->ij','ijkl,jk->il'))

		v_screen_a= (2* vj_a - vk_a)
		h_aa = c_d_h @ (h_core_cross + v_screen_a) @ c_d_h
		h_apap = c_d_l @ (h_core_cross + v_screen_a) @ c_d_l

		h_ab = c_d_h @ (h_core_cross + v_screen_a) @ c_a_h
		h_apbp = c_d_l @ (h_core_cross + v_screen_a) @ c_a_l
	
		# 5. Beta (beta_ab = h_ab - (s_ab * h_aa))
		beta_ab = h_ab - (s_ab * h_aa)
		beta_apbp = h_apbp - (s_apbp * h_apap)
		
		# 6. 2 electron integrals calculation via get_jk (in order to not save ERI in the RAM/disk)
        #    V2e = (L_D L_D | H_A H_A) 
		rho_ld = np.outer(c_d_l, c_d_l)
		rho_ha = np.outer(c_a_h, c_a_h)
		vj_v2e = get_jk((self.mol_A, self.mol_A, self.mol_D, self.mol_D), rho_ld, scripts='ijkl,ji->kl', aosym='s4')
		v_2e = np.sum(vj_v2e * rho_ha)
		
        # 7. J0 = (L_D H_D | H_D L_D) -> Simile a K_Exchange
		rho_t = np.outer(c_d_l, c_d_h)
		v_j0 = get_jk(self.mol_D, rho_t, scripts='ijkl,ji->kl', aosym='s8')
		j0 = np.sum(v_j0 * rho_t)

		p_term = (s_apbp * beta_ab) + (s_ab * beta_apbp) - (s_ab * s_apbp * (v_2e + j0))
        
		return p_term * 27.2114 # eV
	

		
