#!/usr/bin/env python
"""Python program to distort a molecule/chemical system along normal modes. Based on an Orca output file"""

import argparse
import sys
from os.path import splitext

import numpy as np
from cclib.io import ccopen

# Orca uses DA to signify dummy atoms
periodic_symbols = [
    "DA",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

element_masses = [
    # from IUPAC 2021 Table https://iupac.qmul.ac.uk/AtWt/
    0,
    1.00784,
    4.0026,
    6.94,
    9.01218,
    10.81,
    12.011,
    14.007,
    15.999,
    18.9984,
    20.1797,
    22.9898,
    24.305,
    26.9815,
    28.085,
    30.9738,
    32.06,
    35.45,
    39.95,
    39.0983,
    40.078,
    44.9559,
    47.867,
    50.9415,
    51.9961,
    54.938,
    55.845,
    58.9332,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.63,
    74.9216,
    78.971,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.9058,
    91.224,
    92.9064,
    95.95,
    97,
    101.07,
    102.9055,
    106.42,
    107.8682,
    112.414,
    114.818,
    118.71,
    121.76,
    127.6,
    126.9045,
    131.293,
    132.9055,
    137.327,
    138.9055,
    140.116,
    140.9077,
    144.242,
    145,
    150.36,
    151.964,
    157.25,
    158.9253,
    162.5,
    164.9303,
    167.259,
    168.9342,
    173.045,
    174.9668,
    178.486,
    180.9479,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.9666,
    200.592,
    204.38,
    207.2,
    208.9804,
    209,
    210,
    222,
    223,
    226,
    227,
    232.0377,
    231.0358,
    238.0289,
    237,
    244,
    243,
    247,
    247,
    251,
    252,
    257,
    258,
    259,
    262,
    267,
    270,
    269,
    270,
    270,
    278,
    281,
    281,
    285,
    286,
    289,
    289,
    293,
    293,
    294,
]


def weight_mode(structure, norm_mode, atomic_numbers, freq):
    """Converts the normal mode to mass-weighted force constants"""
    masses = np.asarray([element_masses[i] for i in atomic_numbers]).repeat(3)
    # Flatten the normal mode
    mode = np.asarray(norm_mode).reshape(-1, structure.shape[0] * 3)
    fc = abs(freq) * abs(freq) * (mode * mode * masses) / 16.9744 / 100
    fc = fc.reshape(-1, 3)

    return fc


def twizzle(
    structure,
    norm_mode,
    atomic_numbers,
    scaler=1.0,
    verbose=False,
    weight=False,
    freq=None,
):
    """Distorts a structure along the normal mode by a factor of scaler"""
    if verbose:
        print("Using the sacle factor:", scaler)
        if weight:
            print("Using mass-weighted force constants")
    distorted_structure = structure.copy()
    if weight:
        fc = weight_mode(structure, norm_mode, atomic_numbers, freq)
        distorted_structure = distorted_structure + (scaler * fc)
    else:
        distorted_structure = distorted_structure + (scaler * norm_mode)

    return distorted_structure.reshape(-1, 3)


def dump_structure(
    outfile, no_atoms, atomic_numbers, coords, verbose=False, dryrun=False
):
    """Writes out an XYZ format file to outfile"""
    new_xyz = "{}\nTwizzled structure\n".format(no_atoms)
    for label, coord in zip(atomic_numbers, coords):
        new_xyz += "{:2} {:9.6f} {:9.6f} {:9.6f}\n".format(
            periodic_symbols[label], *coord
        )
    if verbose:
        print("\nThe distorted structure is:")
        print(new_xyz)

    if not dryrun:
        with open(outfile, "w") as file:
            if verbose:
                print("Writing distorted structure to file:", outfile)
            file.write(new_xyz)

    return


def new_filename(old_filename, extension="twizzle.xyz"):
    """Generates a new filename based upon the old one plus an optional extension"""
    file_tuple = splitext(old_filename)
    new_xyz_file = file_tuple[0] + "-" + extension

    return new_xyz_file


def imag_freq_check(freqs, verbose=False):
    """Checks for imaginary frequencies and returns a list of which normal modes these correspond to"""
    num_imag_freqs = 0
    imag_modes = []
    first_print = True
    for f_num, freq in enumerate(freqs):
        if freq < 0:
            if first_print:
                first_print = False
                if verbose:
                    print("Summary of imaginary frequencies found")
                    print("======================================")
            num_imag_freqs += 1
            if verbose:
                counter = str(num_imag_freqs) + "."
                print(counter, "imaginary frequency:", freq)
            imag_modes.append(f_num)

    if num_imag_freqs == 0:
        print("No imaginary modes found, stopping.")
        sys.exit()
    elif verbose:
        print("\nTotal number of imaginary frequencies is:", num_imag_freqs, "\n")

    return imag_modes


def parse_orca(orca_file, args):
    """Grabs the required information from an ORCA output file"""
    file = ccopen(orca_file)
    try:
        mol = file.parse()
    except:
        print("Error in cclib attempting to parse the Orca output.")

    # Attempt to extract the XYZ coordinates
    try:
        coords = np.asarray(mol.atomcoords[-1]).reshape(-1, 3)
    except AttributeError as error:
        print(
            "Atomic coordinates not found in file, make sure their printing hasn't been turned off."
        )
        print("The error is: ", error)
    atomic_numbers = mol.atomnos

    if args.modes is None:
        selected_modes = "all"
    else:
        selected_modes = args.modes.split(",")

    freqs = np.asarray(mol.vibfreqs)
    imag_modes = imag_freq_check(freqs, verbose=args.verbose)
    normal_modes = np.asarray(mol.vibdisps)
    displacement_modes = []
    for mode in imag_modes:
        displacement_modes.append(normal_modes[mode])
    displacement_modes = np.array(displacement_modes)

    return coords, atomic_numbers, freqs, imag_modes, displacement_modes, selected_modes


def read_and_distort(args):
    """Read in the data from the output file and distort the structure"""
    filename = args.orca_file
    new_files = new_filename(filename, extension=args.output)

    coords, atomic_numbers, freqs, imag_modes, displacement_modes, selected_modes = (
        parse_orca(filename, args)
    )
    no_atoms = len(atomic_numbers)

    if selected_modes == "all":
        for count, mode in enumerate(displacement_modes):
            coords = twizzle(
                coords,
                mode,
                atomic_numbers,
                scaler=args.scale,
                verbose=args.verbose,
                weight=args.weight,
                freq=freqs[count],
            )
    else:
        print("Not all modes selected.\n")
        # TODO: pass the frequency here too.
        for mode in selected_modes:
            print("Displacing along selected mode", mode)
            actual_mode = int(mode) - 1
            if actual_mode not in imag_modes:
                print(
                    "Error, the selected mode",
                    actual_mode + 1,
                    "is not an imaginary mode.",
                )
                sys.exit()
            coords = twizzle(
                coords,
                displacement_modes[actual_mode],
                atomic_numbers,
                scaler=args.scale,
                verbose=args.verbose,
                weight=args.weight,
            )
    dump_structure(
        new_files,
        no_atoms,
        atomic_numbers,
        coords,
        verbose=args.verbose,
        dryrun=args.dryrun,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Distort a chemical system along a normal mode"
    )
    parser.add_argument("orca_file", help="The ORCA output file to be processed")
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    parser.add_argument(
        "-d",
        "--dryrun",
        help="prevents the writing of the output xyz file, should be used with --verbose",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--scale",
        help="scale factor to use in distortion, default value is 1.0",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="string to append to the filename for the output xyz file, default is twizzle.xyz",
        default="twizzle.xyz",
    )
    parser.add_argument(
        "-m",
        "--modes",
        help="optional comma separated list of imaginary modes to distort along",
    )
    parser.add_argument(
        "-w",
        "--weight",
        help="applies mass-weighting to the distortion",
        action="store_true",
    )

    args = parser.parse_args()

    read_and_distort(args)
