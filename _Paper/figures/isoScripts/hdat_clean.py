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

# cb=sys.argv[3]
# olddir=str(sys.argv[1])+'/LOGSc'+str(cb)
hdatold = str(sys.argv[1]) # olddir+'/history.data'
# newdir=str(sys.argv[2])
hdatnew = str(sys.argv[2]) # newdir+'/history.data'
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
