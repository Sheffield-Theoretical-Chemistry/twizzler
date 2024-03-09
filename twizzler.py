#!/usr/bin/env python
"""Python program to distort a molecule/chemical system along normal modes. Based on an Orca output file"""

from cclib.io import ccopen
import numpy as np
from os.path import splitext
import sys
import argparse

# Orca uses DA to signify dummy atoms
periodic_symbols = ["DA", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
                    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", 
                    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
                    "Ga", "Ge", "As", "Se", "Br", "Kr", 
                    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", 
                    "In", "Sn", "Sb", "Te", "I", "Xe", 
                    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", 
                    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", 
                    "Tl", "Pb", "Bi", "Po", "At", "Rn", 
                    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", 
                    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", 
                    "Nh", "Fl", "Mc", "Lv", "Ts", "Og" ]

def twizzle(structure, norm_mode, scaler=1.0, verbose=False):
    """Distorts a structure along the normal mode by a factor of scaler"""
    if verbose:
        print("Using the sacle factor:", scaler)
    distorted_structure = structure.copy()
    distorted_structure = distorted_structure + (scaler*norm_mode)

    return distorted_structure.reshape(-1, 3)
    
def dump_structure(outfile, no_atoms, atomic_numbers, coords, verbose=False):
    """Writes out an XYZ format file to outfile"""
    new_xyz = '{}\nTwizzled structure\n'.format(no_atoms)
    for label, coord in zip(atomic_numbers, coords):
        new_xyz += '{} {:.6f} {:.6f} {:.6f}\n'.format(periodic_symbols[label], *coord)
    if verbose:
        print("\nThe distorted structure is:")
        print(new_xyz)
    
    with open(outfile, 'w') as file:
        if verbose:
            print("Writing distorted structure to file:", outfile)
        file.write(new_xyz)
        
    return
    
def new_filename(old_filename, extension='-twizzle.xyz'):
    """Generates a new filename based upon the old one plus an optional extension"""
    file_tuple = splitext(old_filename)
    new_xyz_file = file_tuple[0]+extension

    return new_xyz_file

def imag_freq_check(freqs, verbose=False):
    """Checks for imaginary frequencies and returns a list of which normal modes these correspond to"""
    num_imag_freqs = 0
    imag_modes = []
    for f_num, freq in enumerate(freqs):
        if freq < 0:
            if verbose:
                print("Found imaginary frequency:",freq)
            num_imag_freqs += 1
            imag_modes.append(f_num)
    
    if num_imag_freqs == 0:
        print("No imaginary modes found, stopping.")
        sys.exit()
    elif verbose:
        print("Total number of imaginary frequencies is:", num_imag_freqs)
        
    return imag_modes

def parse_orca(orca_file, args):
    """Grabs the required information from an ORCA output file"""
    file = ccopen(orca_file)
    try:
        mol = file.parse()
    except:
        print("Error in cclib attempting to parse the Orca output.")
        
    #Attempt to extract the XYZ coordinates
    try:
        coords = np.asarray(mol.atomcoords[-1]).reshape(-1, 3)
    except AttributeError as error:
        print("Atomic coordinates not found in file, make sure their printing hasn't been turned off.")
        print("The error is: ",error)
    atomic_numbers = mol.atomnos

    freqs = np.asarray(mol.vibfreqs)
    imag_modes = imag_freq_check(freqs, verbose=args.verbose)
    normal_modes = np.asarray(mol.vibdisps)
    displacement_modes = []
    for mode in imag_modes:
        displacement_modes.append(normal_modes[mode])
    displacement_modes = np.array(displacement_modes)

    return coords, atomic_numbers, freqs, imag_modes, displacement_modes

def read_and_distort(args):
    """Read in the data from the output file and distort the structure"""
    filename = args.orca_file
    new_files = new_filename(filename)
        
    coords, atomic_numbers, freqs, imag_modes, displacement_modes = parse_orca(filename, args)
    no_atoms = len(atomic_numbers)

    for mode in displacement_modes:
        coords = twizzle(coords, mode, scaler=args.scale, verbose=args.verbose)
    dump_structure(new_files, no_atoms, atomic_numbers, coords, verbose=args.verbose)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distort a chemical system along a normal mode")
    parser.add_argument("orca_file", help="The ORCA output file to be processed")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-s", "--scale", help="scale factor to use in distortion, default value is 1.0",
                        type=float, default=1.0)
    args = parser.parse_args()
    
    read_and_distort(args)
        
