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

		symmetry = 's4' if self.mol_D.nao == self.mol_A.nao else 's1'

		vj = get_jk((self.mol_D, self.mol_D, self.mol_A, self.mol_A), self.rho_D, scripts='ijkl,ji->kl', aosym=symmetry)
		j_coul = np.sum(vj * self.rho_A) * 27.2114 #eV
		
		return j_coul
		
	def get_K(self):
		vk = get_jk((self.mol_A, self.mol_D, self.mol_D, self.mol_A), self.rho_D, scripts='ijkl,jk->il', aosym='s1')
		j_exch = -1/2*np.sum(vk * self.rho_A) * 27.2114 #eV
        
		return j_exch

	def _setup_supermolecule_and_core(self):

		mol_super = gto.mole.conc_mol(self.mol_D, self.mol_A)
		nD , nA = self.mol_D.nao , self.mol_A.nao
        
		#HOMO(h) and LUMO(l) coefficients assuming a closed shell system
		nocc_d = self.mol_D.nelectron // 2
		c_d_h = self.m1.mf.mo_coeff[:, nocc_d-1]
		c_d_l = self.m1.mf.mo_coeff[:, nocc_d]

		nocc_a = self.mol_A.nelectron // 2
		c_a_h = self.m2.mf.mo_coeff[:, nocc_a-1]
		c_a_l = self.m2.mf.mo_coeff[:, nocc_a]

		c_a_super = np.concatenate([c_d_h, np.zeros(nA)])
		c_ap_super = np.concatenate([c_d_l, np.zeros(nA)])
		c_b_super = np.concatenate([np.zeros(nD), c_a_h])
		c_bp_super = np.concatenate([np.zeros(nD), c_a_l])

        # Overlap integrals
		s_super = mol_super.intor('int1e_ovlp')
		s_ab = c_a_super @ s_super @ c_b_super
		s_apbp = c_ap_super @ s_super @ c_bp_super
		
		# Core density matrices
		dm_core_d = self.m1.mf.make_rdm1() - 2 * np.outer(c_d_h, c_d_h)
		dm_core_a = self.m2.mf.make_rdm1() - 2 * np.outer(c_a_h, c_a_h)
		dm_core_super = np.zeros((nD + nA, nD + nA))
		dm_core_super[:nD, :nD] = dm_core_d
		dm_core_super[nD:, nD:] = dm_core_a

		#Screened hamiltonian
		h_core_super = mol_super.intor('int1e_kin') + mol_super.intor('int1e_nuc')
		vj_core, vk_core = scf.hf.get_jk(mol_super, dm_core_super)
		h_eff = h_core_super + (vj_core - 0.5 * vk_core)

		h_data = {
            'h_aa': c_a_super @ h_eff @ c_a_super,
            'h_bb': c_b_super @ h_eff @ c_b_super,
            'h_apap': c_ap_super @ h_eff @ c_ap_super,
            'h_ab': c_a_super @ h_eff @ c_b_super,
            'h_apbp': c_ap_super @ h_eff @ c_bp_super,
			'h_bpbp': c_bp_super @ h_eff @ c_bp_super,
            's_ab': s_ab,
            's_apbp': s_apbp,
            'c_d_h': c_d_h, 'c_d_l': c_d_l,
            'c_a_h': c_a_h, 'c_a_l': c_a_l
        }
        
		return h_data
	
	def _get_phase_and_j0(self, h_data, singlet=True):
		c_d_h = h_data['c_d_h']
		c_d_l = h_data['c_d_l']
		c_a_h = h_data['c_a_h']
		c_a_l = h_data['c_a_l']

		#phase determination for the coupling sign (based on the overlap of the test transition density with the actual transition density)
		rho_test_D = np.outer(c_d_h, c_d_l)
		phase_D = np.sign(np.sum(rho_test_D * self.rho_D))
		rho_test_A = np.outer(c_a_h, c_a_l)
		phase_A = np.sign(np.sum(rho_test_A * self.rho_A))

		#J0 = (L_D H_D | H_D L_D) -> Similar to K_Exchange
		rho_t = np.outer(c_d_l, c_d_h)
		v_j0 = get_jk(self.mol_D, rho_t, scripts='ijkl,ji->kl', aosym='s8')
		j0 = np.sum(v_j0 * rho_t)
		j0_term = j0 if singlet else -j0

		return phase_D * phase_A, j0_term

	def get_P_term_homo(self, singlet=True):

		h_data = self._setup_supermolecule_and_core()
		global_phase, j0_term = self._get_phase_and_j0(h_data, singlet=singlet)
	
		# Beta (beta_ab = h_ab - (s_ab * h_aa))
		beta_ab = h_data['h_ab'] - (h_data['s_ab'] * h_data['h_aa'])
		beta_apbp = h_data['h_apbp'] - (h_data['s_apbp'] * h_data['h_apap'])
		
		# 2 electron integrals calculation via get_jk (in order to not save ERI in the RAM/disk)
        # V2e = (L_D L_D | H_A H_A) 
		rho_ap = np.outer(h_data['c_d_l'], h_data['c_d_l'])
		rho_b = np.outer(h_data['c_a_h'], h_data['c_a_h'])
		vj_v2e = get_jk((self.mol_A, self.mol_A, self.mol_D, self.mol_D), rho_ap, scripts='ijkl,lk->ij', aosym='s4')
		v_2e = np.sum(vj_v2e * rho_b)

		p_term = - global_phase * ((h_data['s_apbp'] * beta_ab) + (h_data['s_ab'] * beta_apbp) - (h_data['s_ab'] * h_data['s_apbp'] * (v_2e + j0_term)))

		#print('h_aa=', h_data['h_aa']*27.2114)
		#print('h_ab=', h_data['h_ab']*27.2114)
		#print('h_apap=', h_data['h_apap']*27.2114)
		#print('h_apbp=', h_data['h_apbp']*27.2114)
		#print('s_ab=', h_data['s_ab'])
		#print('s_apbp=', h_data['s_apbp'])
		#print('beta_ab=', beta_ab*27.2114)
		#print('beta_apbp=', beta_apbp*27.2114)
		
		return p_term *  27.2114 # eV

	def get_P_term_hetero(self, singlet=True):

		h_data = self._setup_supermolecule_and_core()
		global_phase, j0_term = self._get_phase_and_j0(h_data, singlet=singlet)

		rho_a = np.outer(h_data['c_d_h'], h_data['c_d_h'])     # (aa)
		rho_ap = np.outer(h_data['c_d_l'], h_data['c_d_l'])    # (a'a')
		rho_b = np.outer(h_data['c_a_h'], h_data['c_a_h'])     # (bb)
		rho_bp = np.outer(h_data['c_a_l'], h_data['c_a_l'])    # (b'b')

		v_ap_inter = get_jk((self.mol_A, self.mol_A, self.mol_D, self.mol_D), rho_ap, scripts='ijkl,lk->ij')
		v_bp_inter = get_jk((self.mol_D, self.mol_D, self.mol_A, self.mol_A), rho_bp, scripts='ijkl,lk->ij')

		eri_aa_bpbp = np.sum(v_bp_inter * rho_a)
		eri_bb_apap = np.sum(v_ap_inter * rho_b)

		# J0 terms for tha acceptor A
		rho_t_A = np.outer(h_data['c_a_l'], h_data['c_a_h'])   # (b'b)
		v_j0_A = get_jk(self.mol_A, rho_t_A, scripts='ijkl,ji->kl', aosym='s8')
		j0_A = np.sum(v_j0_A * rho_t_A)
		j0_term_A = j0_A if singlet else -j0_A

		s_ab = h_data['s_ab']
		s_apbp = h_data['s_apbp']

		p_term = - global_phase * (
			(s_apbp * h_data['h_ab']) + 
			(s_ab * h_data['h_apbp']) - 
			(0.5 * s_ab * s_apbp * (2*h_data['h_bb'] + h_data['h_apap'] + h_data['h_bpbp'] + eri_aa_bpbp + eri_bb_apap + j0_term + j0_term_A))
		)

		return p_term *  27.2114 # eV

	def get_indirect_term_homo(self, singlet=True):
		
		h_data = self._setup_supermolecule_and_core()
		global_phase, j0_term = self._get_phase_and_j0(h_data, singlet=singlet)

		s_ab = h_data['s_ab']
		s_apbp = h_data['s_apbp']

		beta_ab = h_data['h_ab'] - s_ab * h_data['h_aa']
		beta_apbp = h_data['h_apbp'] - s_apbp * h_data['h_apap']

		rho_a = np.outer(h_data['c_d_h'], h_data['c_d_h'])     # (aa)
		rho_ap = np.outer(h_data['c_d_l'], h_data['c_d_l'])    # (a'a')
		rho_bp = np.outer(h_data['c_a_l'], h_data['c_a_l'])    # (b'b')

		vj_intra = get_jk(self.mol_D, rho_a, scripts='ijkl,ji->kl', aosym='s8')
		eri_aa_apap = np.sum(vj_intra * rho_ap)

		vj_inter = get_jk((self.mol_A, self.mol_A, self.mol_D, self.mol_D), rho_a, scripts='ijkl,lk->ij', aosym='s4')
		eri_aa_bpbp = np.sum(vj_inter * rho_bp)

		if singlet:
			A_gap = eri_aa_apap - eri_aa_bpbp - 2.0 * j0_term
		else:
			A_gap = eri_aa_apap - eri_aa_bpbp

		#print('A_gap:', A_gap)
	
		if singlet:
			t12 = beta_apbp + 0.5 * s_apbp * A_gap
			t13 = - (beta_ab + 0.5 * s_ab * A_gap)
		else:
			j0_pure = -j0_term
			t12 = beta_apbp + 0.5 * s_apbp * A_gap + s_apbp * j0_pure
			t13 = -beta_ab - 0.5 * s_ab * A_gap - s_ab * j0_pure

		#print('t12:', t12)
		#print('t13:', t13)

		t_indirect = - global_phase * (2.0 * t12 * t13) / A_gap

		return t_indirect * 27.2114 # eV