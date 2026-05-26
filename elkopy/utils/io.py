import numpy as np
import os

def read_xyz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} non trovato.")
    return path

def print_row_output(d, jc, jk, jp, jd, jtot):
    print(f"{d:8.2f} | {jc:11.4e} | {jk:11.4e} | {jp:11.4e} | {jd:11.4e} | {jtot:11.4e}")

def create_translation_vector(offset, dist, axis):
    trans_vector = np.array(offset, dtype=float)
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    trans_vector[axis_map[axis]] += dist
    return trans_vector

def parse_excited_states(state_arg):
    if len(state_arg) == 1:
        return state_arg[0], state_arg[0]
    elif len(state_arg) == 2:
        return state_arg[0], state_arg[1]
    else:
        raise ValueError("Invalid --state argument. Provide either one state (for both monomers) or two states (one for each monomer).")
    
def print_input_recap(xyz_1, xyz_2, basis, state_D, state_A, spin, axis, range, offset):
    file_A = xyz_2 if xyz_2 else xyz_1

    print(f"--- Electronic Coupling Scan ---")
    print(f"Basis set: {basis} | Spin Multiplicity: {spin}")
    print(f"DONOR    -> File: {xyz_1:<25} | State: {state_D}")
    print(f"ACCEPTOR -> File: {file_A:<25} | State: {state_A}")
    print(f"Scan Axis: {axis.upper()} | Range: {range[0]} to {range[1]} (step {range[2]}) | Base Offset: {offset}\n")

def print_td_table(states_data, xyz_file):
    print(f"\n--- Excited states analysis (Monomer: {os.path.basename(xyz_file)}) ---\n")
    print(f"{'State':>5} | {'Energy (eV)':>12} | {'f':>8} | {'Dominant Transition':>22} | {'Amplitude':>8}")
    print("-" * 69)

    for data in states_data:
        trans_str = f"MO {data['mo_occ']:>3} -> MO {data['mo_virt']:<3}"
        print(f"{data['state']:>5} | {data['energy']:>12.4f} | {data['f']:>8.4f} | {trans_str:>22} | {data['amplitude']:>8.4f}")
	
    print("-" * 69 + "\n")
