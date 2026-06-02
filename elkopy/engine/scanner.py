import numpy as np
from elkopy.compchem.monomer import Monomer
from elkopy.physics.el_coupling import ElectronicCoupling
from elkopy.physics.approximations import dipole_dipole_J
from elkopy.utils import io

def run_distance_scan(xyz_1, xyz_2=None, state=[1, 1], spin='singlet', basis='3-21g', xc=None, axis='z', scan_range=[3.0, 13.0, 0.5], offset=[0.0, 0.0, 0.0], run_fullQM=False):
    
    mol_coord1 = io.read_xyz(xyz_1)

    xyz_2 = xyz_2 if xyz_2 else xyz_1
    is_homodimer = (xyz_2 == xyz_1)

    mol_coord2 = io.read_xyz(xyz_2)
    is_singlet = (spin == 'singlet')

    m1 = Monomer(mol_coord1, basis=basis)
    m2 = Monomer(mol_coord2, basis=basis)

    state_D, state_A = state[0], state[1] if len(state) > 1 else state[0]

    max_states = max(state_D, state_A)
    nstates = 3 if max_states < 3 else max_states
    m1.run_calculations(nstates=nstates, singlet=is_singlet, xc=xc)
    m2.run_calculations(nstates=nstates, singlet=is_singlet, xc=xc)

    m1_data = m1.get_exstates_data()
    m2_data = m2.get_exstates_data() if not is_homodimer else None
    
    rho1 = m1.get_trans_density(state_D - 1)
    rho2 = m2.get_trans_density(state_A - 1)
    if np.dot(m1.td.transition_dipole()[state_D - 1], m2.td.transition_dipole()[state_A - 1]) < 0:
        rho2 = -rho2

    calculate_fullQM = is_homodimer and run_fullQM

    yield {
        "status": "init_done",
        "m1_data": m1_data,
        "m2_data": m2_data,
        "calculate_fullQM": calculate_fullQM
    }
    
    start, stop, step = scan_range
    for dist in np.arange(start, stop + (step/10), step): # +(step/10) is just to allow the calculation at distance=stop
        
        trans_vector = io.create_translation_vector(offset, dist, axis)
        m2_new=m2.copy()
        m2_new.move(trans_vector)
        
        # Coupling calculation
        coup = ElectronicCoupling(m1, m2_new, rho1, rho2)
        terms_functions = {
            'jc': lambda: coup.get_J(singlet=is_singlet),
            'jk': lambda: coup.get_K(),
            'jp': lambda: coup.get_P_term_homo(singlet=is_singlet) if is_homodimer else coup.get_P_term_hetero(singlet=is_singlet),
            'jind': lambda: coup.get_indirect_term_homo(singlet=is_singlet) if is_homodimer else float('nan'),
            'jd': lambda: dipole_dipole_J(m1, m2, trans_vector, [state_D, state_A], singlet=is_singlet)
        }

        if calculate_fullQM:
            from elkopy.physics.approximations import full_QM
            terms_functions['jes'] = lambda: full_QM(m1, m2_new, singlet=is_singlet, xc=xc)

        vals = {}
        for name, func in terms_functions.items():
            try:
                vals[name] = func()
            except Exception:
                vals[name] = float('nan')
        
        if np.isnan([vals['jc'], vals['jk'], vals['jp'], vals['jind']]).any():
            j_total = float('nan')
        else:
            j_total = vals['jc'] + vals['jk'] + vals['jp'] + vals['jind']
        
        scan_data = [dist, vals['jc'], vals['jk'], vals['jp'], vals['jind'], vals['jd']]

        if calculate_fullQM:
            scan_data.append(vals['jes'])

        scan_data.append(j_total)

        yield {
            "status": "scan_point",
            "data": scan_data
        }