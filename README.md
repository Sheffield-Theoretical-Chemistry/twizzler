# Twizzler
A Python program that distorts a chemical system along the imagainary (vibrational) normal modes, based on an ORCA output file.

## Installation

Currently you can just download the script (or clone the repo) and run it. The main dependices are listed in the ``requirements.txt``.

Note that Twizzler requires cclib for the parsing of the ORCA output. The current cclib (1.8.1) errors when attempting to read ORCA files where minimal printing has been turned on, I've been able to bypass this by adapting the changes suggested in this [PR for aiida-orca](https://github.com/ezpzbz/aiida-orca/pull/67).

## Usage

The script uses Python's argparse module so that it can act as a command line utility. Running as:

    twizzler.py -h

will provide an overview of the command line flags available. The default operation is "silent" in that only a new xyz file is produced, with no other output to STDOUT. Use the ``-v`` flag for verbose output, which is handy if you want to select a subset of the imaginary modes to distort along.

The standard mode of operation applies ORCA's printed normal mode(s) to the system's XYZ coordinates, with an optional scaling value set with ``-s``. There is a somewhat experimental option to use mass-weighted force constants for the displacement with the command line option ``-w``. Again, this the optional scaling factor can be applied.

## Contributing

Contributions are welcome! Please raise Issues or Pull requests on this repo.

## Other notes

The normal modes output by ORCA are normalized Cartesian displacements. See an ORCA Freq output for more details.
