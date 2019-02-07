from spglib import get_symmetry, standardize_cell, get_spacegroup
import numpy as np


def gulp_opt(stru):
    gulp_inp = 'opt.gin'
    tp_inp = 'tp_inp'
    out = 'out'
    cif_out = 'final.cif'
    symprec = 0.01

    ccell = standardize_cell((stru.cell, stru.get_scaled_positions(), stru.numbers), symprec=symprec)
    if ccell:
        symmetry = get_symmetry(ccell, symprec=symprec)
        spacegroup = get_spacegroup(ccell, symprec=symprec)
    else:
        symmetry = get_symmetry((stru.cell, stru.get_scaled_positions(), stru.numbers), symprec=symprec)
        spacegroup = get_spacegroup((stru.cell, stru.get_scaled_positions(), stru.numbers), symprec=symprec)
    asymmetric_atoms = np.unique(symmetry.equivalent_atoms)
    pass