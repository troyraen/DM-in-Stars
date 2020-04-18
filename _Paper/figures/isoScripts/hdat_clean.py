######
#   This script is used to strip history.data files of unnecessary
#       columns for making isochrones
#   It takes 2 arguments
#       hdatold = path to original history.data file
#       hdatnew = path to write new history.data file to
#   Generates a new file hdatnew from hdatold keeping only columns in clist
#
######

import sys
import numpy as np

usepruned = True


def write_hdat_files_for_MIST(mesa_datadir, iso_datadir, iter_starModel_dirs):
    """Iterates over mesa_datadir to find history files.
    Writes new files with minimal number of columns to iso_datadir for MIST input.

    Args:
    mesa_datadir (Path): main MESA output dir (e.g. Path(plot_fncs.datadir))
    iso_datadir (Path): MIST input dir
                        (e.g. iso_datadir = Path('/home/tjr63/DMS/isomy/data/tracks/')
                        new history files will be placed in tracks/c{cb} directory
    iter_starModel_dirs (func): from data_proc module (data_proc.iter_starModel_dirs)
    """

    # create directories if needed
    for cb in range(7):
        d = iso_datadir / f'c{cb}'
        d.mkdir(parents=True, exist_ok=True)

    # clean and copy the history files to the iso_datadir
    for massdir, mass, cb in iter_starModel_dirs(mesa_datadir):
        hdat = 'history_pruned.data' if usepruned else 'history.data'
        hdatold = massdir / 'LOGS' / hdat
        hdatnew = iso_datadir / f'c{cb}' / f'{massdir.name}.data'
        if hdatold.is_file():
            write_single_hdat_for_MIST(hdatold, hdatnew)


def write_single_hdat_for_MIST(hdatold, hdatnew):
    clist = [ 'star_age','star_mass','log_LH','log_LHe','log_Teff','log_L','log_g',\
            'log_center_T','log_center_Rho','center_h1','center_he4','center_c12',\
            'center_gamma','surface_h1','he_core_mass','c_core_mass' ]

    try:
        # get list of column names that match names in clist
        colnames = np.genfromtxt(hdatold, dtype='str', skip_header=5, \
                    autostrip=True, max_rows=1).tolist()
        keepcols = [ i for i in range(len(colnames)) if colnames[i] in clist ] # list of column numbers to keep
        missingcols = [ i for i in clist if i not in colnames ]
        if len(missingcols) != 0:
            print('')
            print("WARNING, history.data FILE IS MISSING THE FOLLOWING COLUMNS")
            print(missingcols)
            print('')

        # get data from those columns
        # returns (n x m) = (numrows x len(keepcols)) array
        harray = np.genfromtxt(hdatold, dtype='str', skip_header=5, \
                    delimiter=41, usecols=tuple(keepcols))

        # get header data
        hdrtxt = np.genfromtxt(hdatold, dtype='str', delimiter=29, max_rows=3)
        colnums = np.genfromtxt(hdatold, dtype='str', skip_header=4, max_rows=1, \
                    delimiter=41, usecols=range(len(keepcols)))
        # create header
        hdr = '\n'.join([''.join(hdrtxt[0]), ''.join(hdrtxt[1]), \
                    ''.join(hdrtxt[2]), '', ''.join(colnums)])

        # write new file
        np.savetxt(hdatnew, harray, fmt='%s', delimiter='', comments='', header=hdr)
    except:
        print()
        print(hdatold, 'does not seem to exist')
        print('or there was a problem writing to', hdatnew)
        print('Skipping.\n')
