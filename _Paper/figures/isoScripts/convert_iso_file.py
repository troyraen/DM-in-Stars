import numpy as np


# converts isochrone output files (isochrone_cb.dat) to isochrones.csv for Glue
# can combine files from multple cboosts
def iso_to_csv(cboost=[0], append_cb=True, append_PEEPs=True, isodir=None):
    if isodir is None:
        isodir = '/Users/troyraen/Google_Drive/MESA/code/iso'
    eepinput = isodir+'/data/input.eep'
    fout = isodir+'/glue/isochrones.csv'
    hdr = ''

    for i, cb in enumerate(cboost):
        # print(i)
        fin = isodir+'/data/isochrones/isochrone_c'+str(cb)+'.dat'
        # get column names and make sure they're the same across files
        if i == 0:
            isoheader, cnt = get_columns(fin, cboost=append_cb, PrimaryEEPs=append_PEEPs)
            alldata = np.empty((1,cnt))
        else:
            hdr, __ = get_columns(fin, cboost=append_cb, PrimaryEEPs=append_PEEPs)
            if hdr != isoheader:
                print()
                print(fin, 'COLUMNS DO NOT MATCH isochrone_c0.dat')
                print()

        # get data
        newdata = np.genfromtxt(fin, comments='#')
        if append_cb:
            cbarr = cb*np.ones((newdata.shape[0],1))
            newdata = np.append(newdata, cbarr, axis=1)
        if append_PEEPs:
            newdata = append_PrimaryEEPs(isoheader, newdata, eepinput)
        alldata = np.concatenate((alldata, newdata), axis=0)

    # write new file
    np.savetxt(fout, alldata, delimiter=',', header=isoheader, comments='')


# returns column names as a list
# optionally adds cboost to the end
def get_columns(isofile, cboost=False, PrimaryEEPs=False):
    # get column names, ignore leading '#'
    isoheader = np.genfromtxt(isofile, dtype='str', \
            skip_header=10, comments=None, max_rows=1)[1:].tolist()
    if cboost:
        isoheader.append('cboost')
    if PrimaryEEPs:
        isoheader.append('PrimaryEEP')
    cnt = len(isoheader)
    isoheader = ','.join(isoheader[i] for i in range(len(isoheader)))
    return [isoheader, cnt]


# EEPs from Dotter 2016
def append_PrimaryEEPs(hdr, arrin, eepinput='/Users/troyraen/Google_Drive/MESA/code/iso/input.eep'):
    try:
        eep_col = np.where(np.array(hdr.split(",")) == 'EEP')[0][0]
    except IndexError:
        print("Data input file does not inlude EEP numbers. Cannot append Primary EEPs.")
        raise IndexError
    PEEPdict = get_PimaryEEPs(eepinput) # {EEP #: Primary EEP #}
    print('PEEPdict {EEP #: Primary EEP #}', PEEPdict)
    PEEP_names = ['PreMS', 'ZAMS', 'IAMS', 'TAMS', 'RGBTip', 'ZACHeB', 'TACHeB', 'TP-AGB', 'PostAGB', 'WDCS']
    print('PEEP_names', PEEP_names)
    num_rows = arrin.shape[0]
    arrout = np.append(arrin, -1*np.ones((num_rows,1)), axis=1) # default = -1 (EEP is not a primary)
    for i in range(num_rows):
        EEP = arrout[i][eep_col]
        if EEP in PEEPdict.keys(): arrout[i][-1] = PEEPdict[EEP]
    return arrout


def get_PimaryEEPs(eepinput):
    """Takes input.eep file as input.
    Returns dict of {EEP #: Primary EEP #}."""
    max_rows = int(np.genfromtxt(eepinput, skip_header=1, max_rows=1))
    arr = np.genfromtxt(eepinput, skip_header=2, max_rows=max_rows, dtype=int)
    run_EEP = 1
    PEEPdict = {}
    for i in range(max_rows):
        PEEPdict[run_EEP] = i
        run_EEP = run_EEP + arr[i][1] + 1
    return PEEPdict
