#!/Users/troyraen/anaconda/bin/python

# ---------------------------------- #
# ---------------------------------- #
# %matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mesa_reader as mr
import os
from pathlib import Path
#import collections as col
from collections import OrderedDict as OD
from scipy.interpolate import interp1d
#import matplotlib.gridspec as gridspec
import matplotlib.lines as lines
import pandas as pd
from matplotlib import colors as mcolors
import random
import inspect
from functools import reduce
from numpy import inf



# ---------------------------------- #
def calc_emoment_dm(prof, hist, hidx, norm=False):
    """prof = specific stars dict profile
    hist = stars dict history
    norm: whether to return normalized integral
    hidx: get using fnc get_model_num(profile_num, dic) -> '[ model_num, hidx ]':
    """
    # constants
    mstar = hist.star_mass[hidx]
    Tx = hist.wimp_temp[hidx]
    # maxT = 10.**(prof.logT.max())
	# Txhigh = maxT*1.2
	# Txlow = maxT/28.0
    Msun = 1.9892e33  # solar mass (g)
    kerg = 1.3806504e-16 # Boltzmann's constant (erg K^-1)
    gperGeV= 1.78266e-24 # grams to GeV/c^2 conversion
    mxGeV = 5. # 5 GeV WIMP
    mx = mxGeV* gperGeV # WIMP mass in grams
    mpGeV = 0.938272
    mp = 1.6726231e-24 # proton mass (g)
    T = 10.**prof.logT
    emom = []
    for Txx in np.nditer(T):
        efact = np.exp(-mx*prof.Vk/kerg/Txx) # array
        dm = prof.dq*mstar
        rho = 10.**(prof.logRho)
        if not norm:
            mfact = dm/4./np.pi/rho
            Tfact = np.sqrt((mpGeV*Txx+ mxGeV*T)/(mxGeV*mpGeV))* (T-Txx)
            emom.append(np.sum(prof.np*Tfact*efact*mfact))
        else:
            mfact = dm/Msun/rho
            Tfact = np.sqrt(Txx/1.e7+ mxGeV/mpGeV*T/1.e7)* (T-Txx)/1.e7
            emom.append(np.sum(prof.np/1.e25*Tfact*efact*mfact))
    return np.asarray(emom)
# ---------------------------------- #




# ---------------------------------- #
def make_csv(stars, othmap, list_Hcols,list_Pcols, clean=False, rt = '/Users/troyraen/Google_Drive/MESA/code/DATA'):
    dic_allhist = {}
    dic_allprof = {}
    dic_alldesc = {}
    MSDF = create_MS_DF(list(stars.values()), othmap) # for desc
    # clean = 'profiles'
    # clean = False
    print('\n')
    for skey, sdic in stars.items():
        histprof_to_DF(sdic,othmap,list_Hcols,list_Pcols) # get history and profiles dataframes
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
    maindir = rt+'/Glue'
    # maindir='./dftest'
    glue_historyDF.to_csv(maindir+'/historyDF.csv', index=True, index_label='star_index')
    glue_profilesDF.to_csv(maindir+'/profilesDF.csv', index=True, index_label='star_index')
    glue_descDF.to_csv(maindir+'/descDF.csv', index=True, index_label='star_index')

# ---------------------------------- #
# LOAD FROM SOURCE

# OD 'stars' holds everything
# each entry has key = order in which star was added (e.g. star[3]),
# value = dictionary holding:
#     mass: as float
#     spin: 'Dep' or 'Ind'
#     cb: c# (cboost)
#     other: other characteristics ( = 'basic' if no other characteristics)
#     hist: history.data file as mesa data object
#     profidx: profiles.index file as ?
#     profs: OD with key = model or prof number, value = profile#.data file as mesa data object


# root0 = './implicit_test/'
# impdirs = [ item for item in os.listdir(root0) if os.path.isdir(root0 + item) ]
# for idir in sorted(impdirs):
#     if idir != 'all': continue
#     root = root0+idir+'/'

# print("\n# ***** # LOADING DATA... \n")
def make_dicts(root, masses, load_prof=True):
    stars=OD([])
    massdirs = [ item for item in os.listdir(root) if os.path.isdir(root + item) ]
    for mdir in sorted(massdirs):
        if mdir[0] != 'm': continue
    #         if mdir != 'mass0p8': continue
        mass = float(mdir[4]+'.'+mdir[6:])
        if mass not in masses: continue
        mkcbdir_flat(stars, mass, 'Dep', root+mdir+'/', load_prof)
        print('finished directory '+ root+mdir +'\n')

    print('\n Data loaded to OD stars, SKIPPING find_events. \n')

    print('finished head directory'+ root)

    return stars


# ---------------------------------- #
def mkcbdir_flat(stars, mass, spin, root, load_prof=False):
    cbdirs = [ item for item in os.listdir(root) if \
                  ( os.path.isdir(root + item) and item[:5] == 'LOGSc' ) ]

    for c in sorted(cbdirs):
        rootc = root + c + '/'
        print("processing folder",rootc)

        sid = len(stars.items())
        stars[sid] = OD([])
        sdic = stars[sid]
        dic = sdic

        # mass, spin, cboost, other descriptors
        dic['mass'] = mass
        dic['cb'] = c[4:6]
        dic['spin'] = spin
        dic['other'] = c[7:] if len(c)>6 else 'basic'
        desc = get_desc(dic)
        print('Loading', desc)

        hist = rootc+'history.data'
        prof = rootc+'profiles.index'
        if not Path(hist).is_file() or not Path(prof).is_file():
            print(hist + ' or '+prof+ ' file not found. stars['\
                  +str(sid)+']: '+ ''.join(desc) + ' deleted')
            del sdic
            continue

        try:
            dic['hist'] = mr.MesaData(hist)
            dic['pidx'] = mr.MesaProfileIndex(prof)
            dic['l'] = mr.MesaLogDir(rootc)

        except:
            print('Problem loading data. stars['+str(sid)+']: '+''.join(desc) + ' deleted')
            del sdic
            continue

        if load_prof==True:
            load_profiles(dic, rootc)

#         try:
#             histprof_to_DF(sdic) # convert history and profiles to dataframes
#         except:
#             '*** unable to convert history and profiles to dataframes'
#         try:
#             pidx_to_DF(sdic) # convert profiles.index info to DF
#         except:
#             '*** unable to convert profiles.index info to dataframe'

    print('finished creating cdicts for '+ root)


# ---------------------------------- #
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

# ---------------------------------- #
# create description dataframe

def desc_to_DF(sdic, MStau_DF, othmap):
    star_idx = get_star_idx(sdic,othmap)
    mass, cb, other, numprofs = get_desc(sdic)
    other = int(str(star_idx)[-1])# = 0 if other=='basic' else 1
    h = sdic['hist']
    enterMSmodel = h.model_number[findEV_enterMS(h)]
    leaveMSmodel = h.model_number[findEV_leaveMS(h)]
    MStau = MStau_DF.MStau[star_idx]


    cols=['star_index', 'mass', 'cboost', 'other', 'enterMS_model', 'leaveMS_model', 'MStau']
    data = [ int(star_idx), float(mass[0:3]), int(cb[1]), other, enterMSmodel, leaveMSmodel, MStau ]
    datadic = {1:data}
#     print(type(datadic[1]))
    df = pd.DataFrame.from_dict(datadic, orient='index', columns=cols)
    sdic['descDF'] = df

# ---------------------------------- #
# get hist data and profile data into dataframe

def histprof_to_DF(sdic,othmap, list_Hcols, list_Pcols):
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

# ---------------------------------- #
# remove duplicate model numbers (currently none exist, so skipping for now)
    # and thin or remove data from post-MS

def clean_histprofDF(sdic, post_MS_action='remove', clean='all'):

    """Post-MS models thinned or removed from sdic[historyDF'] and sdic['profilesDF']['all']
        Returns second element as an empty list if there are no profiles loaded."""
    if post_MS_action!='remove':
        print('post_MS_action =', post_MS_action, 'has not been added to this function')
        return

    h = sdic['hist']
    msidx = findEV_leaveMS(h)
    leaveMSmod = h.model_number[msidx] if msidx !=0 else h.model_number[-1]
            # if leaveMS not found, use the last model
    if 'historyDF' not in sdic.keys(): histprof_to_DF(sdic)

    hdf = sdic['historyDF']
    hdf_clean = hdf[hdf.model_number <= leaveMSmod]

    b = 'profilesDF' in sdic.keys()
    if b: b = 'all' in sdic['profilesDF'].keys() # if true, sdic['profilesDF']['all'] exists
    if b:
        pdf = sdic['profilesDF']['all']
        pdf_clean = pdf[pdf.model_number <= leaveMSmod]
    else:
        pdf_clean = []


    if clean=='all':
        rtn = [ hdf_clean, pdf_clean ]
    elif clean=='profiles':
        rtn = [ sdic['historyDF'], pdf_clean ]
    elif clean=='history':
        rtn = [ hdf_clean, sdic['profilesDF']['all'] ]
    else:
        print('clean option', clean, 'not set up in clean_histprofDF')
        rtn = []

    return rtn

# ---------------------------------- #
# function: create dataframe containing MS lifetimes and cboosts for plotting
# returns: dataframe with columns = ['star_index', mass', 'cb', 'MStau']

def create_MS_DF(star_dicts, othmap):
    sdicts = star_dicts[:]
    listMSdicts = []

    while sdicts:
        idx = [] # list of indicies of processed dicts to remove from sdicts
        mass = sdicts[0]['mass']
        l = len(listMSdicts)
        listMSdicts.append({'mass':mass}) # create dict for this mass
        star_idx = []
        cb = []
        tau = [] #convert these to arrays at end

        # do this twice. 1st time, get c0. 2nd time, get all others.
        for k in [1,2]:
            for sdic in sdicts:
                if sdic['mass'] == mass:

                    if k==1 and sdic['cb']!='c0': continue
                    if k==2 and sdic['cb']=='c0': continue
                    if k==2 and len(cb)==0:
                        print('WARNING:', mass, 'DOES NOT CONTAIN C0 MODEL')
                    idx = idx + [sdicts.index(sdic)]

                    # cboost:
                    cbtmp = int(sdic['cb'][1])
                    cb.append(cbtmp if sdic['other']=='basic' else cbtmp+0.5)

                    # ms lifetime:
                    h = sdic['hist']
                    enterMS = h.star_age[findEV_enterMS(h)]
                    leaveMS = h.star_age[findEV_leaveMS(h)]
                    MStau = leaveMS - enterMS
                    MStau0 = MStau if k==1 else MStau0
                    tau.append(0. if k==1 else (MStau-MStau0)/MStau0)

                    # star_index
                    star_idx.append(get_star_idx(sdic,othmap))

        listMSdicts[l]['star_index'] = np.array(list(star_idx))
        listMSdicts[l]['cb'] = np.array(list(cb))
        listMSdicts[l]['MStau'] = np.array(list(tau))
        listMSdicts[l]['mass'] = np.ones(len(cb))*mass
        sdicts = [s for s in sdicts if sdicts.index(s) not in idx]

    columns = ['star_index', 'mass', 'cb', 'MStau']
    msdf = pd.concat([pd.DataFrame(listMSdicts[i], columns=columns) \
             for i in range(len(listMSdicts))], ignore_index=True)
    msdf.set_index('star_index', drop=False, inplace=True, verify_integrity=True)

    return msdf

# ---------------------------------- #
def pidx_to_DF(sdic):
    pnums = sdic['pidx'].profile_numbers
    bname = [ 'model_numbers', 'profile_numbers']
#     print(sdic['pidx'].priority)
    data = [ sdic['pidx'].data(nm) for nm in bname ]

    bname.append('hidx') # history index
    hidx = [ get_model_num(p, sdic)[1] for p in pnums ] # get hidx
    data.append(np.array(hidx))

    bname.append('index') # index in pidx data
    idx = [ np.where(pnums==p)[0][0] for p in pnums]
    data.append(np.array(idx))

    bname.append('available') # is the profile loaded in sdic
    avail = [ str(p) in sdic['profiles'].keys() for p in pnums ]
    data.append(np.array(avail))

    df = pd.DataFrame.from_dict(dict(zip(bname, data)))
#     df.set_index('index', inplace=True)
    sdic['pidxDF'] = df


# ---------------------------------- #
# load profile data

def load_profiles(dic, root):
    pfiles = [ item for item in os.listdir(root) \
                  if (item[0:7]=='profile' and item[-5:]=='.data') ]

    dic['profiles'] = {}
    for pfile in pfiles:
        try:
            p = pfile[7:-5]
            if p=='1001': p='0'
            dic['profiles'][p] = mr.MesaData(file_name=root+pfile)
        except:
            print('Problem loading profile', pfile)
            continue
    print('\t loaded', str(len(dic['profiles'])), 'profiles')

# ---------------------------------- #
# get descriptions of star dicts for different purposes

def get_desc(dic):
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

def get_desc_mcb(dic):
    if 0 in dic.keys(): # dic is a dict of single star dicts
        desc = []
        for dkey, d in dic.items():
            desc.append([ d['mass'], d['cb'] ])
    else: # dic is a single star dict
        desc = [ dic['mass'], dic['cb'] ]
    return desc

def get_star_idx(dic, othmap={'basic':'0'}):
#     oth = { 'basic':'0', 'real4':'4', 'real8':'8', 'real16':'9'}
    m = str(dic['mass'])[0]+str(dic['mass'])[2:]
    c = dic['cb'][1]
    s = '1' if dic['spin'] == 'Dep' else '0'
    o = othmap[dic['other']] if dic['other'] in othmap.keys() else '-1'
#     o = '0' if dic['other'] == 'basic' else '1'
    desc = int(m+c+s+o)
    return desc

# ---------------------------------- #
# returns [ model number, history index ]
def get_model_num(profile_num, dic):
    h = dic['hist']
    profile_num = np.int64(profile_num)
    if profile_num == 1001 or profile_num == 0:
        model_num = h.model_number[-1]
        hidx = len(h.model_number)-1
    else:
        idx = np.where(dic['pidx'].profile_numbers==profile_num)[0][0]
        model_num = dic['pidx'].model_numbers[idx]
        try:
            hidx = np.where(h.model_number==model_num)[0][0]
        except:
#             print( 'hidx has been thinned (no longer exists in data) for', get_desc(dic), \
#                   'profile',profile_num, 'model',model_num, '. \nReturning -1' )
            hidx = -1
    return [ model_num, hidx ]

# returns [ profile number, ? ]
def get_profile_num(model_num, dic):
    idx= np.where(dic['pidx'].model_numbers==model_num)[0][0]
    p= dic['pidx'].profile_numbers[idx]
    return p

# ---------------------------------- #
#### FIND EVENTS FUNCTIONS
def findEV_leaveMS(hist_data):
    """
    find h1_center < 0.01 (leave MS) event:
    returns h index of event, 0 if event not found
    """
    h = hist_data
    tmp = np.where(h.center_h1<0.01)[0]
    h1idx = tmp[0] if len(tmp)!=0 else 0
    return h1idx

# find h1_center < 0.71 (enter MS) event:
# returns h index of event, 0 if event not found

def findEV_enterMS(hist_data):
    h = hist_data
    tmp = np.where(h.center_h1<0.71)[0]
    h1idx = tmp[0] if len(tmp)!=0 else 0
    return h1idx


# find index of given model number
def findEV_model(hist_data, model_num):
    h = hist_data
    model_hidx = np.where(h.model_number==model_num)[0][0]
    return model_hidx


# returns true if model is in main sequence
def is_in_MS(sdic, hidx=None, model=None, profile=None):
    h = sdic['hist']
    if model is not None: hidx = findEV_model(h, model)
    if profile is not None: hidx = get_model_num(profile, sdic)[1]
    idxe = findEV_enterMS(h)
    idxl = findEV_leaveMS(h)
    in_MS = idxe <= hidx and hidx <= idxl
    return in_MS

# ---------------------------------- #
def Trho_degen_line(sdic):
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
