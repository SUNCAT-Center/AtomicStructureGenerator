'''
This module uses Crystal Structure Prototype Database (CSPD) to generate a
list of crystal structures for the system defined by user. These structures
could be used for machine learning, high-throughput calculations, and
structure prediction for materials design. They are also a very good
source of structures for fitting empirical potentials. The function
atomic_structure_generator could return a list of ASE
(https://wiki.fysik.dtu.dk/ase/) Atoms object. You could use any calculator
supported by ASE for further calculations, such as structure optimization.

This program is developed at SUNCAT Center for Interface Science and
Catalysis, SLAC National Accelerator Laboratory, Stanford University. This
work is funded by SUNCAT center Toyota Research Institute.

The methodology of the program is introduced in the article [Chuanxun Su,
et al. "Construction of crystal structure prototype database: methods and
applications." Journal of Physics: Condensed Matter 29.16 (2017): 165901].
Please cite it if you use the program for your research.

Feel free to contact the author (information is listed below) whenever you
run into bugs, or you want some features to be added to the program. Please
let the author know if you want to add the module to your project or make
changes to it.

Author:
Chuanxun Su, Ph.D. in Condensed Matter Physics
Postdoctoral Scholar
SUNCAT Center for Interface Science and Catalysis, SLAC, Stanford University
Email: scx@stanford.edu / suchuanxun@163.com
QQ: 812758366

November 7, 2018
'''

from ase.db import connect
from ase import Atoms
from ase.io import write
from ase.data import covalent_radii
from spglib import standardize_cell, get_spacegroup
import random
import math
import numpy as np
import os


def atomic_structure_generator(symbols, fu=None, ndensity=None, volume=None,
                               mindis=None, nstr=None, maxatomn=None,
                               cspd_file=None, lw=None, format=None, clean=None,
                               sgn=None, to_primitive=None):
    """
    This function will read crystal structure prototypes from CSPD.db file
    and return a list of ASE Atoms object for the symbols defined by users.
    It could also write the output structures into any file format supported
    by ASE.

    The symbols is the only parameter required to define. The function has
    very justified default values for the rest of the parameters. You could
    also customize them according to your own understanding of your system.

    :param symbols: str (formula) or list of str
        Can be a string formula, a list of symbols or a list of
        Atom objects.  Examples: 'H2O', 'COPt12', ['H', 'H', 'O'],
        [Atom('Ne', (x, y, z)), ...]. Same as the symbols in ASE Atoms class
        (https://wiki.fysik.dtu.dk/ase/ase/atoms.html). This parameter is
        passed to the Atoms class.
    :param fu: list of int
        Range of formula unit. The symbols is multiplied by every formula
        unit in this range. The length of this list has to be 2.
    :param ndensity: float
        Total number of atoms divided by volume. It controls how dense
        the atoms stack.
    :param volume: float
        Average volume of the structure per symbols. If ndensity is
        defined, volume will be ignored. I would strongly recommend to use
        ndensity rather than volume.
    :param mindis: list of lists of float
        Minimum inter-atomic distances. The dimension is nxn for the
        structure has number of n type of element. For example, we could
        define it as [[1.7, 1.4],[1.4, 1.2]] for binary compound.
        mindis[0][1] defines the minimum distance between element 1 with
        element 2.
    :param nstr: int
        This number of structures will be returned. Less of structures might
        be returned when there's not enough qualified structures in the
        database.
    :param maxatomn: int
        Maximum number of atoms in the structure.
    :param cspd_file: str
        Path and file name of CSPD.db.
    :param lw: logical
        Whether to write the structures into files. The structures are put in
        structure_folder.
    :param format: str
        Used to specify the file-format. Same as the format in ase.io.write()
        function. Check out the supported file-format at their website
        (https://wiki.fysik.dtu.dk/ase/ase/io/io.html).
    :param sgn: list of int
        Range of space group sequential number as given in the International
        Tables for Crystallography.
    :return: list of ASE Atoms object
    """
    random.seed(a=27173)
    strulist = []
    newstrulist = []
    locmindis = {}
    rdr = 0.61
    scale_ndensity = 2.22
    structure_folder = 'structure_folder'
    lwf = False

    if fu == None:
        fu = [2, 8]
    if nstr == None:
        nstr = 1600
    if maxatomn == None:
        maxatomn = 60
    if cspd_file == None:
        cspd_file = '~/CSPD.db'
    if lw == None:
        lw = False
    if format == None:
        format = 'cif'
    if clean == None:
        clean = True
    if sgn == None:
        sgn = [1, 230]
    if to_primitive == None:
        to_primitive = False

    tmpstru = Atoms(symbols)
    intctype = count_atoms(tmpstru.numbers)
    atomnn = unify_an(tmpstru.numbers)
    nele, strctype, gcd = nele_ctype_fu(intctype)
    # intctype=[int(float(i)/gcd+0.5) for i in intctype]
    fulist = [i * gcd for i in range(fu[0], fu[1] + 1)]
    if format == 'db':
        dbw = connect(structure_folder + '/' + tmpstru.get_chemical_formula() + '.db', append=False)
    if volume:
        volume = float(volume)
    if ndensity:
        ndensity = float(ndensity)
    if mindis:
        for i, an in enumerate(atomnn.keys()):
            for j, an2 in enumerate(atomnn.keys()):
                locmindis[str(an) + '_' + str(an2)] = mindis[i][j]
    elif not (ndensity or volume):
        for an in atomnn.keys():
            for an2 in atomnn.keys():
                locmindis[str(an) + '_' + str(an2)] = (covalent_radii[an] + covalent_radii[an2]) * rdr
    if not ndensity:
        if volume:
            ndensity = len(tmpstru.numbers) / volume
        elif mindis:
            ndensity = scale_ndensity * sum(atomnn.values()) / (4 / 3.0 * math.pi / ((2 * rdr) ** 3) / 0.34 * sum(
                [atomnn[sym] * mindis[i][i] ** 3 for i, sym in enumerate(atomnn.keys())]))
        else:
            ndensity = scale_ndensity * sum(atomnn.values()) / (
                    4 / 3.0 * math.pi / 0.34 * sum([atomnn[sym] * covalent_radii[sym] ** 3 for sym in atomnn.keys()]))
    if (ndensity or volume) and (not mindis):
        for an in atomnn.keys():
            for an2 in atomnn.keys():
                locmindis[str(an) + '_' + str(an2)] = (covalent_radii[an] + covalent_radii[an2]) * rdr
        # Need to improve!!! calculate locmindis according to ndensity

    db = connect(cspd_file)
    for row in db.select('ctype=_' + strctype):
        if row.lfocp and sgn[0] <= row.sgn <= sgn[1]:
            tempstru = row.toatoms()
            if to_primitive:
                pcell = standardize_cell(
                    (tempstru.cell, tempstru.get_scaled_positions(), tempstru.numbers),
                    to_primitive=True, symprec=0.01)
                # sgsn=get_spacegroup((strulist[struid].cell,strulist[struid].get_scaled_positions(),strulist[struid].numbers),symprec=0.01)
                # sgsn2=get_spacegroup(pcell,symprec=0.01)
                if pcell:
                    tempstru = Atoms(cell=pcell[0], scaled_positions=pcell[1], numbers=pcell[2], pbc=True)
            natoms = len(tempstru.numbers)
            if natoms <= maxatomn:
                if to_primitive:
                    final_fu = int(row.fu / row.natoms * natoms + 0.5)
                else:
                    final_fu = row.fu
                for i in fulist:
                    if final_fu == i:
                        strulist.append(tempstru)
                        strulist[-1].sgn = row.sgn
                        strulist[-1].dname = row.dname
                        strulist[-1].oid = row.oid
                        # Or construct a list of rows and convert part of them to atoms object.
                        break
    if lw:
        if os.path.exists(structure_folder):
            if clean:
                for tpfn in os.listdir(structure_folder):
                    path_file = os.path.join(structure_folder, tpfn)
                    if os.path.isfile(path_file):
                        os.remove(path_file)
        else:
            os.makedirs(structure_folder)
    isucc = 0
    ifail = 0
    for i in range(len(strulist)):
        struid = int(random.random() * len(strulist))
        tempstru = strulist[struid]
        tempstru.set_cell(
            tempstru.cell * (len(tempstru.numbers) / tempstru.get_volume() / ndensity) ** (1.0 / 3),
            scale_atoms=True)
        tempstru.set_atomic_numbers(subst_ele(tempstru.numbers, atomnn))
        # Add break points will change the random number!!! Wired!!!
        if checkdis(tempstru, locmindis):
            newstrulist.append(tempstru)
            isucc += 1
        else:
            if lwf:
                write(structure_folder + '/' + str(ifail + 1) + '_failed_' + tempstru.get_chemical_formula()
                      + '_{}'.format(tempstru.sgn) + '.cif'
                      , tempstru)
            del strulist[struid]
            ifail += 1
            continue
        if lw:
            suffix = ''
            if format == 'cif':
                suffix = '.cif'
            elif format == 'vasp':
                suffix = '.vasp'
            if (format == 'db'):
                dbw.write(newstrulist[-1], sgn=newstrulist[-1].sgn, dname=newstrulist[-1].dname,
                          oid=newstrulist[-1].oid)
            else:
                write(structure_folder + '/' + str(isucc) + '_' + newstrulist[-1].get_chemical_formula()
                      + '_{}'.format(newstrulist[-1].sgn) + suffix
                      , newstrulist[-1], format=format)
        print('Chemical Formula: {:9} Space Group: {:4d}'.format(newstrulist[-1].get_chemical_formula(),
                                                                 newstrulist[-1].sgn))
        if isucc == nstr:
            break
        del strulist[struid]
    print('{} structures generated\n{} physically unjustified structures are filtered out'.format(isucc, ifail))
    return newstrulist


def variable_stoichiometry_generator(symbols, stoichiometry, clean=False, fu=None,
                                     mindis=None, nstr=None, maxatomn=None,
                                     cspd_file=None, lw=None, format=None,
                                     sgn=None, to_primitive=None):
    tpstru = Atoms(symbols)
    smbl = []
    for smb in tpstru.get_chemical_symbols():
        if not (smb in smbl):
            smbl.append(smb)
    for stc in stoichiometry:
        cc = ''
        for i, n in enumerate(stc):
            cc += smbl[i] + '{}'.format(n)
        atomic_structure_generator(
            symbols=cc, fu=fu, mindis=mindis, nstr=nstr, maxatomn=maxatomn,
            cspd_file=cspd_file, lw=lw, format=format, clean=clean, sgn=sgn, to_primitive=to_primitive)
    pass


def checkdis(atoms, dis):
    '''
    This function checks whether the inter-atomic distances is justified or
    not, and it's very time consuming for large structure!!! There's no
    documentation for the rest of the code. So, good luck and enjoy.

    :param atoms:
    :param dis:
    :return:
    '''
    squdis = {}
    for key in dis.keys():
        squdis[key] = dis[key] ** 2
    irange = cell_range(atoms.cell, max(dis.values()))
    natoms = len(atoms.numbers)
    strnum = [str(i) for i in atoms.numbers]
    transa = -irange[0] * atoms.cell[0]
    for ia in range(-irange[0] + 1, irange[0] + 1):
        transa = np.row_stack((transa, ia * atoms.cell[0]))
    transb = -irange[1] * atoms.cell[1]
    for ib in range(-irange[1] + 1, irange[1] + 1):
        transb = np.row_stack((transb, ib * atoms.cell[1]))
    transc = -irange[2] * atoms.cell[2]
    for ic in range(-irange[2] + 1, irange[2] + 1):
        transc = np.row_stack((transc, ic * atoms.cell[2]))
    for i1 in range(natoms):
        for i2 in range(i1, natoms):
            vct = atoms.positions[i2] - atoms.positions[i1]
            tpsqudis = squdis[strnum[i2] + '_' + strnum[i1]]
            for ia in range(-irange[0], irange[0] + 1):
                for ib in range(-irange[1], irange[1] + 1):
                    for ic in range(-irange[2], irange[2] + 1):
                        if i1 == i2 and ia == 0 and ib == 0 and ic == 0:
                            continue
                        if np.sum(np.square(
                                vct - transc[ic + irange[2]] - transb[ib + irange[1]] - transa[
                                    ia + irange[0]])) < tpsqudis:
                            return False
    return True


def cell_range(cell, rcut):
    recipc_no2pi = Atoms(cell=cell).get_reciprocal_cell()
    return [int(rcut * ((np.sum(recipc_no2pi[i] ** 2)) ** 0.5)) + 1 for i in range(3)]


def subst_ele(numbers, atomnn):
    locatomnn = atomnn.copy()
    origatomnn = unify_an(numbers)
    for key in origatomnn.keys():
        for key2 in locatomnn.keys():
            if origatomnn[key] == locatomnn[key2]:
                origatomnn[key] = key2
                del locatomnn[key2]
                break
    tmpn = []
    for key in numbers:
        tmpn.append(origatomnn[key])
    return tmpn


def count_atoms(numbers):
    ctype = {}
    for i in numbers:
        if i in ctype:
            ctype[i] += 1
        else:
            ctype[i] = 1
    return sorted(ctype.values())


def unify_an(numbers):
    '''
    Stores the composition type in a dictionary.
    :param numbers:
    :return:
    '''
    atmnn = {}
    for i in numbers:
        if i in atmnn:
            atmnn[i] += 1
        else:
            atmnn[i] = 1
    natom = sorted(atmnn.values())
    if len(natom) == 1:
        for symb in atmnn.keys():
            atmnn[symb] = 1
        return atmnn
    gcd = natom[0]
    for i in natom[1:]:
        n1 = gcd
        n2 = i
        while True:
            gcd = n2 % n1
            if gcd == 0:
                gcd = n1
                break
            elif gcd == 1:
                return atmnn
            else:
                n2 = n1
                n1 = gcd
    for symb in atmnn.keys():
        atmnn[symb] = int(float(atmnn[symb]) / gcd + 0.5)
    return atmnn


def nele_ctype_fu(natom):
    if len(natom) == 0:
        return 0, '0', 0
    elif natom[0] == 0:
        return 0, '0', 0
    gcd = natom[0]
    lctype = len(natom)
    if lctype == 1:
        return 1, '1', gcd
    for i in natom[1:]:
        n1 = gcd
        n2 = i
        while True:
            gcd = n2 % n1
            if gcd == 0:
                gcd = n1
                break
            elif gcd == 1:
                return lctype, strctype(natom), 1
            else:
                n2 = n1
                n1 = gcd
    return lctype, strctype([int(float(i) / gcd + 0.5) for i in natom]), gcd


def strctype(ctype):
    sctype = str(ctype[0])
    if len(ctype) == 1:
        return sctype
    for i in ctype[1:]:
        sctype = sctype + '_' + str(i)
    return sctype
