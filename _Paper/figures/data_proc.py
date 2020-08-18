"""
General functions for loading and processing MESA output data
"""
import os
import pandas as pd
import numpy as np
import datetime

###--- For generating descDF.csv ---###
from collections import OrderedDict as OD
from numpy import inf


# import plot_fncs as pf

priority_dict = {   'ZAMS': 99,
                    'IAMS': 98,
                    f'log h1$_c$ < -1': 97,
                    f'log h1$_c$ < -2': 96,
                    f'log h1$_c$ < -3': 95,
                    'TAMS': 94,
                    'He ignition': 93,
                    f'log he4$_c$ < -3': 92,
                }


# fs ###--- Rename a file ---###
def file_ow(fin, suffix='date'):
    """ Moves file fin.txt to fin_ow_%m%d%y_%H%M.txt
        Returns the new file name.
    """

    # split file name
    fsplit = str(fin).split('.')
    beg, end = '.'.join(fsplit[0:-1]), '.'+fsplit[-1]

    if suffix == 'date':
        # get date and time to use in new file name
        dtm = datetime.datetime.now()
        suffix = dtm.strftime("_ow_%y%m%d_%H%M")

    # create new file name
    fout = beg + suffix + end

    # move the file
    print('Moving existing file {0} to {1}'.format(fin, fout))
    os.rename(fin, fout)

    return fout
# fe ###--- Rename a file ---###


# fs ###--- Load MESA output files ---###

def iter_starModel_dirs(rootPath):
    """ Generator function that iterates over directories containing MESA star model data.
        e.g. rootPath = Path('/home/tjr63/DMS/mesaruns/RUNS_defDM')
    """
    for cbdir in rootPath.iterdir():
        if not cbdir.is_dir(): continue
        cb = int(cbdir.name[-1])

        for massdir in cbdir.iterdir():
            # This is the directory for a single star which contains the LOGS dir
            if not massdir.is_dir(): continue
            if len(massdir.name.split('_')) > 1: continue # skip dirs with a suffix
            mass = float('.'.join(massdir.name.split('p')).strip('m'))

            yield (massdir, mass, cb)

def load_profilesindex(fpidx):
    """ Loads a single profiles.index file """

    pidx_cols = ['model_number', 'priority', 'profile_number']
    pidf = pd.read_csv(fpidx, names=pidx_cols, skiprows=1, header=None, sep='\s+')
    return pidf

def load_history(hpath, masscb=None):
    hdf = pd.read_csv(hpath, header=4, sep='\s+')

    if masscb is not None:
        hdf['mass'], hdf['cb'] = masscb
        hdf.set_index(['mass','cb'], inplace=True)

    return hdf

# fe ###--- Load MESA output files ---###



# fs ###--- Prune history.data files ---###
def prune_hist(LOGSdir, skip_exists=True, ow_exists=False):
    """ Prunes the history.data file in a LOGSdir
        If skip_exists==True, will skip if LOGSdir/history_pruned.data exists (priority)
        If ow_exists==True, will rename LOGSdir/history_pruned.data if exists
    """

    fhist = LOGSdir / 'history.data'
    fnewhist = LOGSdir / 'history_pruned.data'
    fpidx = LOGSdir / 'profiles.index'

    if fnewhist.is_file():
        if skip_exists:
            print(f'{fnewhist} already exists.. skipping.')
            return
        elif ow_exists:
            file_ow(str(fnewhist))

    pidf = load_profilesindex(fpidx)
    modnums = pidf.model_number
    TAMSmod = pidf.loc[pidf.priority==priority_dict['TAMS'],'model_number'].values[0]
    care_cols = [   'star_mass','luminosity','log_Teff','mass_conv_core','log_LH',
                    'cno','pp','center_T','center_Rho','center_h1','center_he4']

    # read/write the file
    with open(fhist) as oldhist, open(fnewhist, 'w') as newhist:
        cols = []
        for l, line in enumerate(oldhist):
            write = False

            # write header
            if l<=5:
                write = True
                if l==5:
                    cols = line.split()

            # write selective history
            else:
                s = pd.Series(data=line.split(),index=cols)
                if l==6: slast = s.copy()

                # write line if model has a profile#.data file
                linemod = int(s.get('model_number'))
                if any(modnums.isin([linemod])):
                # if linemod<TAMSmod or any(modnums.isin([linemod])):
                    write = True

                # write line if fractional change > 0.1%
                else:
                    for c in care_cols:

                        # de-log the values
                        if c.split('_')[0] == 'log':
                            s[c] = 10**float(s[c])

                        # check fractional change
                        val, vallast = float(s.get(c)), float(slast.get(c))
                        if vallast==0: continue
                        if abs(vallast-val)/vallast > 0.001:
                            write = True
                            break

            # actually write the line
            if write:
                if l>5: slast = s.copy()
                newhist.write(line)

    print(f'Pruned history.data from {LOGSdir}')

# fe ###--- Prune history.data files ---###



# fs ###--- For gen_descDF_csv.py ---###
def get_desc(dic) -> "desc = [ mass, cb, other, #profiles] as strings":
    if 0 in dic.keys(): # dic is a dict of single star dicts
        desc = []
        for dkey, d in dic.items():
            numprofs = len(d['profiles'].keys()) if 'profiles' in d.keys() else 0
            desc.append([ 'stars['+str(dkey)+']', str(d['mass'])+' Msun', d['cb'], d['other'],\
                        str(numprofs)+' profiles'])
    else: # dic is a single star dict
        numprofs = len(dic['profiles'].keys()) if 'profiles' in dic.keys() else 0
        desc = [ str(dic['mass'])+' Msun', dic['cb'], dic['other'], \
                str(numprofs)+' profiles']
    return desc

# sort stars dicts by mass and cboost
def sort_stars_dicts(stars):
    stars_sorted = OD([])

    mlist = sorted(set([dic['mass'] for key,dic in stars.items()])) # get list of unique masses
    for m in mlist:
        mdicts = [dic for key, dic in stars.items() if dic['mass']==m] # get dicts of mass m
        cblist = sorted(set([dic['cb'] for dic in mdicts])) # list of unique cb's

        for cb in cblist:
            for dic in mdicts:
                l = len(stars_sorted)
                if dic['cb'] ==cb:
                    stars_sorted[l] = dic

    return stars_sorted

def make_csv(stars,othmap,list_Hcols,list_Pcols, clean=False, write=['hist','prof','desc'], \
            rt = '/Users/troyraen/Google_Drive/MESA/code/DATA'):
    dic_allhist = {}
    dic_allprof = {}
    dic_alldesc = {}
    MSDF = create_MS_DF(list(stars.values()), othmap) # for desc
    # clean = 'profiles'
    # clean = False
    print('\n')
    for skey, sdic in stars.items():
        if (('historyDF' not in sdic.keys()) or ('profilesDF' not in sdic.keys() and 'profiles' in sdic.keys())):
            # get history and profiles dataframes
            histprof_to_DF(sdic,othmap,list_Hcols,list_Pcols)
        if not clean: # clean = False
            histDF = sdic['historyDF']
            b = 'profilesDF' in sdic.keys()
            if b: b = 'all' in sdic['profilesDF'].keys() # if true, sdic['profilesDF']['all'] exists
            profDF = sdic['profilesDF']['all'] if b else [] # =[] if no profiles loaded
        else: # clean = 'all' or 'profiles' or 'history'
            histDF, profDF = clean_histprofDF(sdic, clean=clean) # profDF=[] if no profiles loaded
        dic_allhist[get_star_idx(sdic,othmap)] = histDF
        if type(profDF) == pd.DataFrame:
            dic_allprof[get_star_idx(sdic,othmap)] = profDF
        if 'descDF' not in sdic.keys():
            desc_to_DF(sdic, MSDF, othmap) # get description dataframe
        dic_alldesc[get_star_idx(sdic,othmap)] = sdic['descDF']
        print('Done with', get_desc(sdic))

    print('\n')
    glue_historyDF = pd.concat(dic_allhist, ignore_index=True) # make one big DF
    glue_historyDF.set_index('star_index', drop=True, inplace=True, verify_integrity=False)
    try:
        glue_profilesDF = pd.concat(dic_allprof, ignore_index=True)
    except: # if no profiles are loaded, create placeholder df so other things work
        columns = [ 'model_number', 'profile_number', 'star_index', 'mass', 'cb', 'other' ]
        row = np.zeros(shape=(1,len(columns)))
        glue_profilesDF = pd.DataFrame(row, columns=columns)
    glue_profilesDF.set_index('star_index', drop=True, inplace=True, verify_integrity=False)
    glue_descDF = pd.concat(dic_alldesc, ignore_index=False)
    glue_descDF.set_index('star_index', drop=True, inplace=True, verify_integrity=False)

    # save these to csv files
    if 'hist' in write:
        glue_historyDF.to_csv(rt+'/historyDF.csv', index=True, index_label='star_index')
    if 'prof' in write:
        glue_profilesDF.to_csv(rt+'/profilesDF.csv', index=True, index_label='star_index')
    if 'desc' in write:
        glue_descDF.to_csv(rt+'/descDF.csv', index=True, index_label='star_index')

def desc_to_DF(sdic, MStau_DF, othmap) -> "sdic['descDF'] = ":
    star_idx = get_star_idx(sdic,othmap)
    mass, cb, other, numprofs = get_desc(sdic)
    other = int(str(star_idx)[-1])# = 0 if other=='basic' else 1
    h = sdic['hist']

    zidx = findEV_enterMS(h)
    enterMSmodel = h.model_number[zidx]
    ZAMS_Teff = h.log_Teff[zidx]
    ZAMS_L = h.log_L[zidx]

    lidx = findEV_leaveMS(h)
    leaveMSmodel = h.model_number[lidx]
    lAMS_Teff = h.log_Teff[lidx]
    lAMS_L = h.log_L[lidx]

    tidx = findEV_TAMS(h)
    TAMSmodel = h.model_number[tidx]
    TAMS_Teff = h.log_Teff[tidx]
    TAMS_L = h.log_L[tidx]

    TACHeBmodel = h.model_number[findEV_TACHeB(h)]
    MStau = MStau_DF.MStau[star_idx]
    MStau_yrs = MStau_DF.MStau_yrs[star_idx]
    pp = MStau_DF.PPavg[star_idx]
    cno = MStau_DF.CNOavg[star_idx]
    masscc_avg = MStau_DF.masscc_avg[star_idx]
    masscc_ZAMS = MStau_DF.masscc_ZAMS[star_idx]
    Hburned = MStau_DF.Hburned[star_idx]

    cols=['star_index', 'mass', 'cboost', 'other', \
            'enterMS_model', 'ZAMS_Teff', 'ZAMS_L', \
            'leaveMS_model', 'lAMS_Teff', 'lAMS_L', \
            'TAMS_model', 'TAMS_Teff', 'TAMS_L', \
            'TACHeB_model', 'MStau_yrs', 'MStau', \
            'PPavg', 'CNOavg', 'masscc_avg', 'masscc_ZAMS', 'Hburned']
    data = [ int(star_idx), float(mass[:-5]), int(cb[1]), other, \
                enterMSmodel, ZAMS_Teff, ZAMS_L, \
                leaveMSmodel, lAMS_Teff, lAMS_L, \
                TAMSmodel, TAMS_Teff, TAMS_L, \
                TACHeBmodel, MStau_yrs, MStau, pp, cno, \
                masscc_avg, masscc_ZAMS, Hburned ]
    datadic = {1:data}
#     print(type(datadic[1]))
    df = pd.DataFrame.from_dict(datadic, orient='index', columns=cols)
    sdic['descDF'] = df

#### FIND EVENTS FUNCTIONS
# find h1_center < 0.71 (enter MS) event:
# returns h index of event, 0 if event not found
def findEV_enterMS(hist_data) -> 'h1idx':
    h = hist_data
    # tmp = np.where(h.center_h1<0.71)[0]
    tmp = np.where(h.center_h1 < (h.center_h1[0] - 0.0015))[0]
    h1idx = tmp[0] if len(tmp)!=0 else 0
    return h1idx

def findEV_leaveMS(hist_data) -> 'h1idx':
    """
    find h1_center < 0.001 (leave MS) event:
    returns h index of event, 0 if event not found
    """
    h = hist_data
    tmp = np.where(h.center_h1<0.001)[0]
    h1idx = tmp[0] if len(tmp)!=0 else 0
    return h1idx

def findEV_TAMS(hist_data) -> 'h1idx':
    """
    find h1_center < 10^-12 (terminal age MS) event:
    returns h index of event, 0 if event not found
    """
    h = hist_data
    tmp = np.where(h.center_h1<1e-12)[0]
    h1idx = tmp[0] if len(tmp)!=0 else 0
    return h1idx

def findEV_TACHeB(hist_data) -> 'h1idx':
    """
    find center_he4 < 10^-3 (terminal age He burning) event:
    returns h index of event, 0 if event not found

    I think MIST uses center_he4 < 10^-4, but 10^-3 is the profile I have saved, so I'm using it.
    """
    h = hist_data
    tmp = np.where(h.center_he4<1e-3)[0]
    h1idx = tmp[0] if len(tmp)!=0 else 0
    return h1idx

# function: create dataframe containing MS lifetimes and cboosts for plotting
# returns: dataframe with columns = ['star_index', 'mass', 'cb', 'MStau', 'PPavg', 'CNOavg']
def create_MS_DF(star_dicts, othmap) -> 'msdf':
    sdicts = star_dicts[:]
    listMSdicts = []

    while sdicts:
        idx = [] # list of indicies of processed dicts to remove from sdicts
        mass = sdicts[0]['mass']
        l = len(listMSdicts)
        listMSdicts.append({'mass':mass}) # create dict for this mass
        star_idx = []
        cb = []
        tau_yrs = []
        tau = []
        pp = []
        cno = []
        mass_conv_core = []
        mcc_ZAMS = []
        Hburned = [] #convert these to arrays at end

        # do this twice. 1st time, get c0. 2nd time, get all others.
        for k in [1,2]:
            for sdic in sdicts:
                if sdic['mass'] == mass:

                    if k==1 and sdic['cb']!='c0': continue
                    if k==2 and sdic['cb']=='c0': continue
                    if k==2 and len(cb)==0:
                        print('WARNING:', mass, 'DOES NOT CONTAIN C0 MODEL')
                        idx = idx + [sdicts.index(sdic)]
                        continue
                    idx = idx + [sdicts.index(sdic)]

                    # cboost:
                    cbtmp = int(sdic['cb'][1])
                    cb.append(cbtmp if sdic['other']=='basic' else cbtmp+0.5)

                    # MS lifetime:
                    h = sdic['hist']
                    eMS, lMS = findEV_enterMS(h), findEV_leaveMS(h)
                    enterMS = h.star_age[eMS]
                    leaveMS = h.star_age[lMS]
                    MStau = leaveMS - enterMS
                    tau_yrs.append(MStau)
                    MStau0 = MStau if k==1 else MStau0
                    tau.append(0. if k==1 else (MStau-MStau0)/MStau0)

                    # H burned during MS relative to c0 model
                    Hburn = h.total_mass_h1[eMS] - h.total_mass_h1[lMS]
                    Hburn0 = Hburn if k==1 else Hburn0
                    Hburned.append(0. if k==1 else (Hburn-Hburn0)/Hburn0)


                    # setup a DF for what follows
                    en = findEV_enterMS(h) - 1 # get 1 model before MS for delta_age (below)
                    ex = findEV_leaveMS(h)
                    df = pd.DataFrame(data={'star_age':h.star_age[en:ex],
                                            'pp':h.pp[en:ex],
                                            'cno':h.cno[en:ex],
                                            'mass_conv_core':h.mass_conv_core[en:ex]
                                            })
                    df['delta_age'] = df.star_age.diff()
                    df.drop(labels=0, axis=0, inplace=True) # drop the model before MS

                    # PP and CNO burning (average over MS)
                    pp.append(np.log10((10**(df.pp)* df.delta_age).sum() / MStau))
                    cno.append(np.log10((10**(df.cno)* df.delta_age).sum() / MStau))

                    # convective core (average over MS)
                    mass_conv_core.append((df.mass_conv_core* df.delta_age).sum() / MStau)
                    mcc_ZAMS.append(df.mass_conv_core.iloc[0])

                    # star_index
                    star_idx.append(get_star_idx(sdic,othmap))

        listMSdicts[l]['star_index'] = np.array(list(star_idx))
        listMSdicts[l]['cb'] = np.array(list(cb))
        listMSdicts[l]['MStau_yrs'] = np.array(list(tau_yrs))
        listMSdicts[l]['MStau'] = np.array(list(tau))
        listMSdicts[l]['mass'] = np.ones(len(cb))*mass
        listMSdicts[l]['PPavg'] = np.array(list(pp))
        listMSdicts[l]['CNOavg'] = np.array(list(cno))
        listMSdicts[l]['masscc_avg'] = np.array(list(mass_conv_core))
        listMSdicts[l]['masscc_ZAMS'] = np.array(list(mcc_ZAMS))
        listMSdicts[l]['Hburned'] = np.array(list(Hburned))
        sdicts = [s for s in sdicts if sdicts.index(s) not in idx]

    columns = ['star_index', 'mass', 'cb', 'MStau_yrs', 'MStau', 'PPavg', 'CNOavg', \
                'masscc_avg', 'masscc_ZAMS', 'Hburned']
    msdf = pd.concat([pd.DataFrame(listMSdicts[i], columns=columns) \
             for i in range(len(listMSdicts))], ignore_index=True)
    msdf.set_index('star_index', drop=False, inplace=True, verify_integrity=True)

    return msdf

def get_star_idx(dic, othmap={'basic':'0'}) -> "desc = MassCbSpinOther as 4-6 digit int (leading 0 stripped)":
#     oth = { 'basic':'0', 'real4':'4', 'real8':'8', 'real16':'9'}
    m = str(dic['mass'])[0]+str(dic['mass'])[2:]
    c = dic['cb'][1]
    s = '1' if dic['spin'] == 'Dep' else '0'
    o = othmap[dic['other']] if dic['other'] in othmap.keys() else '-1'
#     o = '0' if dic['other'] == 'basic' else '1'
    desc = int(m+c+s+o)
    return desc

# get hist data and profile data into dataframe
def histprof_to_DF(sdic,othmap, list_Hcols, list_Pcols) -> 'get hist data and profile data into dataframe':
    sidx = get_star_idx(sdic,othmap)

     # create DF from history data
    h = sdic['hist']
    hlen = len(h.model_number)
#     list_Hcols = list(h.bulk_names) # list of column names (strings)
    print(get_desc(sdic))
    hdata = [ h.data(i) for i in list_Hcols ] # list of column data (np arrays)
    bname = list_Hcols[:]

    # add column of Trho_degen_line
    degen = np.log10( Trho_degen_line(sdic) )
    degen[degen == -inf] = 0 # convert -inf to 0
    hdata.append(degen)
    bname.append('log_centerT_degen')
#     print(hlen, len(degen))

    # add column of MS boolean array (true if model is in MS)
    inms = [ is_in_MS(sdic, hidx=h) for h in range(len(sdic['hist'].model_number)) ] # get bools
    inms = np.asarray([ int(inms[i]==True) for i in range(len(inms))]) # turn True/False to 1/0
    hdata.append(inms)
    bname.append('inMS_bool')

    # add column of profile numbers, -1 if no profile is loaded
    prof_array = -1*np.ones(hlen)
    try:
        for pkey in sdic['profiles'].keys():
            prof_array[get_model_num(pkey, sdic)[1]] = int(pkey) #prof_array[hidx]=pkey
    except:
        # print('\nno profiles loaded, all profile numbers = -1\n')
        pass
    hdata.append(prof_array)
    bname.append('profile_number')

    # add column for star index
    sidx_array = np.ones(hlen)*sidx
    hdata.append(sidx_array)
    bname.append('star_index')

    # add mass
    mass = sdic['mass']
    mass_array = np.ones(hlen)*mass
    hdata.append(mass_array)
    bname.append('mass')

    # add cboost
    cb = int(sdic['cb'][1])
    cb_array = np.ones(hlen)*cb
    hdata.append(cb_array)
    bname.append('cb')

    # add other
    other = int(str(sidx)[-1])
    other_array = np.ones(hlen)*other
    hdata.append(other_array)
    bname.append('other')

    # create DF
    df = pd.DataFrame.from_dict(dict(zip(bname, hdata)))
    df.loc[:, (df != 0).any(axis=0)] # remove columns of all zeros
    df['profile_number'] = df['profile_number'].astype(int) # change to integer
    df['star_index'] = df['star_index'].astype(int)
    df['mass'] = df['mass'].astype(float)
    df['cb'] = df['cb'].astype(int)
    df['other'] = df['other'].astype(int)
#     df.set_index('star_index', inplace=True, drop=False)
    sdic['historyDF'] = df

     # create profilesDF from each profile data
    if (( 'profiles' in sdic.keys() ) and ( len(sdic['profiles']) > 0 )): # else no profiles are loaded
        sdic['profilesDF'] = {} # dict to hold df.s
        for p in sdic['profiles'].keys():
            pdat = sdic['profiles'][p]
            if p == '1001': p = '0'
            pdata = [ pdat.data(i) for i in list_Pcols ]
                    # load profile data as list of np arrays
            bname = list_Pcols[:]

            l = len(pdata[1])
            h = sdic['hist']
            lh = len(h.model_number)
            mod, hidx = get_model_num(p, sdic) if p!='0' else [h.model_number[-1], lh-1]
                    # get model number and hist index for profile p

            # for plotting degen line with T on y axis (T = degen_array at degeneracy)
            degen_array = np.log10( 1.21e5* pdat.mu**(-2/3)* pdat.free_e**(-5/3)* \
                        (10.**pdat.logRho)**(2/3) )
            pdata.append(degen_array)
            bname.append('log_T_degen')

            Tx_array = np.ones(l)*np.log10(h.wimp_temp[hidx]) # append array of model numbers
            pdata.append(Tx_array)
            bname.append('log_Tx')

            R = 10.**pdat.logR
            R = np.append(R,0.0)
            dr = [ R[i]-R[i+1] for i in range(len(R)-1) ] # zones go sfc to center
            pdata.append(np.asarray(dr))
            bname.append('dr')

            xL = pdat.extra_L # add dL_extra array
            dxL = [ xL[i]-xL[i+1] for i in range(len(xL)-1) ]
            dxL.append(xL[-1])
            pdata.append(np.asarray(dxL))
            bname.append('dxL')

            emom = calc_emoment_dm(pdat, h, hidx, norm=False) # add emoment array. plot with logT on x axis
            pdata.append(np.nan_to_num(emom, copy=False))
            bname.append('emoment_dm')
            emom = calc_emoment_dm(pdat, h, hidx, norm=True) # add normalized emoment array. plot with logT on x axis
            pdata.append(np.nan_to_num(emom, copy=False))
            bname.append('emoment_dm_norm')

            p_array = np.ones(l)*int(p) # append array of profile numbers
            pdata.append(p_array)
            bname.append('profile_number')

            mod_array = np.ones(l)*mod # append array of model numbers
            pdata.append(mod_array)
            bname.append('model_number')

            hidx_array = np.ones(l)*hidx# append array of hist index
            pdata.append(hidx_array)
            bname.append('history_index')

            sidx_array = np.ones(l)*sidx# append array of unique index for star with decimal point for profile
            pdata.append(sidx_array)
            bname.append('star_index')

            mass_array = np.ones(l)*mass # mass
            pdata.append(mass_array)
            bname.append('mass')

            cb_array = np.ones(l)*cb # cb
            pdata.append(cb_array)
            bname.append('cb')

            other_array = np.ones(l)*other # other
            pdata.append(other_array)
            bname.append('other')

            df = pd.DataFrame.from_dict(dict(zip(bname, pdata)))
            df.loc[:, (df != 0).any(axis=0)] # remove columns of all zeros
            df['profile_number'] = df['profile_number'].astype(int) # change to integer
            df['model_number'] = df['model_number'].astype(int)
            df['history_index'] = df['history_index'].astype(int)
#             df['star_index'] = df['star_index'].astype(int)
            df['mass'] = df['mass'].astype(float)
            df['cb'] = df['cb'].astype(int)
            df['other'] = df['other'].astype(int)
            df['log_T_degen'] = df['log_T_degen'].astype(float)
    #         df.set_index('star_index', inplace=True, drop=False)
            sdic['profilesDF'][p] = df # save profiles data as dataframe

         # combine profiles data into single dataframe
        sdic['profilesDF']['all'] = pd.concat(sdic['profilesDF'], ignore_index=False)

def Trho_degen_line(sdic) -> 'ydata = 1.21e5*mu**(-2/3)*free_e**(-5/3)*(rho**(2/3)); 0 if no suitable profiles':
    h = sdic['hist']
    rho = h.center_Rho
    # get earliest profile in MS and use central mu and free_e from that profile:
    p_in_ms = {}
    try:
        for pkey, pdic in sdic['profiles'].items():
            hidx = get_model_num(pkey, sdic)[1]
            if hidx == -1: continue # hidx no longer exists (thinned from history data)
            if is_in_MS(sdic, profile=pkey): p_in_ms[hidx] = pdic
    except:
        # print('\nno profiles loaded, skipping Trho_degen_line\n')
        pass
    if not bool(p_in_ms): # dictionary is empty
#         print(get_desc(sdic), ' has no loaded profile in MS. Returning 0.0 from Trho_degen_line.')
        ydata = rho*0.0
    else:
        pdat = p_in_ms[min(p_in_ms.keys())]
        m = 1.21e5*pdat.mu[-1]**(-2/3)*pdat.free_e[-1]**(-5/3)
        ydata = m*(rho**(2/3))
    return ydata

# returns true if model is in main sequence
def is_in_MS(sdic, hidx=None, model=None, profile=None) -> 'True if hidx or model is in MS':
    h = sdic['hist']
    if model is not None: hidx = findEV_model(h, model)
    if profile is not None: hidx = get_model_num(profile, sdic)[1]
    idxe = findEV_enterMS(h)
    idxl = findEV_leaveMS(h)
    in_MS = idxe <= hidx and hidx <= idxl
    return in_MS

# fe ###--- For generating descDF.csv ---###
