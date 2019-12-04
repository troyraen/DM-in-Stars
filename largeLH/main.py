%run largeLH_data_fncs

# fs----- plot luminosities v age -----#
h = hdf.loc[hdf.star_age>1e7,:]
age = h.star_age
L = h.luminosity
LH = 10**h.log_LH
LHe = 10**h.log_LHe
Lnuc = 10**h.log_Lnuc
Lneu = 10**h.log_Lneu
Lgrav = h.eps_grav_integral
extra_L = h.extra_L

dic = OD([
        ('age', age),
        ('L', (L, ':')),
        ('LH', (LH, '-.')),
        ('LHe', (LHe, ':')),
        ('extra_L', (extra_L, '-')),
        ('Lneu', (Lneu, ':')),
        ('Lnuc', (Lnuc, '-')),
        ('Lgrav', (Lgrav, '-')),
      ])
plot_lums_history(dic)

plot_lum_excess(age,L,Lnuc,Lgrav)

# fe----- plot luminosities v age -----#



# fs----- which timesteps have profiles? -----#
# using fnc and vars defined in 'plot luminosities v age' section
dic = OD([
        ('age', age),
        ('L', (L, ':')),
        ('Lnuc', (Lnuc, '-')),
        ('Lgrav', (Lgrav, '-')),
      ])
plot_lums_history(dic, profiles='all')

# profile numbers to load to df
pidf.loc[((pidf.star_age>3e8) & (pidf.star_age<4e8)),:]
pnums4df = [65,66,71,72,73,74,76,79] #***--- NEEDED TO PLOT LUMINOSITY PROFILES ---***#
plot_lums_history(dic, profiles=pnums4df)

# fe----- which timesteps have profiles? -----#



# fs----- plot luminosity profiles -----#
# load profiles to df
# get pnums4df from 'which timesteps have profiles?' section
dfs = []
for p in pnums4df:
    ppath = c6path+ f'/profile{p}.data'
    pdf = pd.read_csv(ppath, header=4, sep='\s+')
    pdf['profile_number'] = p
    dfs.append(pdf.set_index(['profile_number','zone']))
pdf = pd.concat(dfs, axis=0)

# plot
plot_lums_profiles(pdf)


# fe----- plot luminosity profiles -----#
