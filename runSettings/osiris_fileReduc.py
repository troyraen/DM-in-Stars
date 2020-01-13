"""
Meant to be run on Osiris to find large STD.out and history.data files
and generate reduced versions.
"""

import os
from os.path import join as pjoin

mesaruns = '/home/tjr63/DMS/mesaruns'
dr = mesaruns + '/RUNS_runSettings'


def check_for_reduc(dr=dr, run_key=['all'], max_fsize=500.0):
    """ Checks the following files in LOGSpath dir and creates a reduced
        version if file size > max_fsize [MB] (and does not already exist):
            STD.out
            history.data
    """

    smax = max_fsize*1024*1024 # [bytes]

    for cb in os.listdir(dr):
        if cb[0] != 'c': continue
        for mdir in os.listdir(pjoin(dr,cb)):
            rk = mdir.split('_',1)[-1]

            # use only dirs in run_key
            if (rk not in run_key) and (run_key!=['all']): continue
            # if rk.split('_')[0] == 'ow': continue # skip 'ow' dirs

            LOGSpath = pjoin(dr,cb,mdir,'LOGS')
            # m = float('.'.join(mdir.split('_')[0].strip('m').split('p')))
            # dfkey = (rk, int(cb[-1]), m)
            # print()
            # print(f'doing {dfkey}')

            # reduce STD and history file sizes if needed
            z = zip(['STD.out', 'history.data'],['STD_reduc.out', 'history_reduc.data'])
            for fil, rfil in z:
                typ = fil.split('.')[0]

                OGp, rp = pjoin(LOGSpath,fil), pjoin(LOGSpath,rfil)
                if os.path.exists(rp): continue

                if os.path.getsize(OGp) > smax:
                    print('Reducing {typ} at {OGp}'.format(typ,OGp))
                    if typ == 'STD':
                        os.system("tail -n 100 '{OGp}' > '{rp}'".format(OGp,rp))
                    elif typ == 'history':
                        os.system('../bash_scripts/data_reduc.sh {LOGSpath}'.format(LOGSpath))

    return None
