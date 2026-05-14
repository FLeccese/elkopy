from pyscf import gto, scf, tdscf
import numpy as np
from copy import deepcopy

class Monomer:
	def __init__(self, xyz_file, basis):
		self.xyz_file = xyz_file
		self.basis = basis
		self.mol = self._build_mol(xyz_file, basis)
		self.mf = None
		self.td = None

	@staticmethod
	def _build_mol(source, basis):
		mol = gto.Mole()
		mol.fromfile(source)
		mol.basis = basis
		mol.unit = 'Angstrom'
		mol.build()
		return mol

	def optimize_geometry(self):
		from pyscf.geomopt.geometric_solver import optimize
		mf = scf.RHF(self.mol)
		self.mol = optimize(mf)
		return self.mol

	def run_calculations(self, nstates=5, singlet=True, xc=None):
		if xc is None:
			self.mf = scf.RHF(self.mol).run()
			self.td = tdscf.TDA(self.mf)
		else:
			from pyscf import dft
			self.mf=dft.RKS(self.mol)
			self.mf.xc = xc
			self.mf.run()
			self.td=tdscf.TDA(self.mf)
		
		self.td.singlet = singlet
		self.td.run(nstates=nstates)
		return self.td
		
	def get_trans_density(self, idx):
		nocc = self.mol.nelectron // 2
		c_occ = self.mf.mo_coeff[:, :nocc]
		c_virt = self.mf.mo_coeff[:, nocc:]
		a_ia = self.td.xy[idx][0]*np.sqrt(2)
		rho_mono = c_occ @ a_ia @ c_virt.T 
		return rho_mono

	def copy(self):
		return deepcopy(self)

	def move(self, vector):
		#Shift the monomer by a vector dx,dy,dz    
		coords = self.mol.atom_coords(unit='Angstrom')
		labels = [a[0] for a in self.mol._atom]
        
		new_atoms = []
		for i, label in enumerate(labels):
			new_pos = coords[i] + vector
			new_atoms.append((label, tuple(new_pos)))
        
		self.mol.atom = new_atoms
		self.mol.build()