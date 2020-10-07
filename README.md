# DM-in-Stars #

Explores the effects of dark matter (DM) on stellar evolution. It adds a module to the code base Modules for Experiments in Stellar Astrophysics ([MESA](http://mesa.sourceforge.net)) which calculates the capture of DM by stars and the subsequent energy transport via scattering.


## Quick Start ##
1. Install MESA by following the instructions [here](http://mesa.sourceforge.net/prereqs.html).
2. Clone this repo `git clone git@github.com:troyraen/DM-in-Stars.git`
3. Change settings in the `inlist` as desired (optional)
4. Alter DM properties in `inlist_DM` as desired (optional, see below for details)
5. Run a MESA + DM model using the following:
```bash
cd DM-in-Stars
./clean
./mk
./rn
```

__Alternately__, if one has existing code running MESA models, this DM module can be incorporated by copying `inlist_DM` and the contents of the `src` directory from this repo.


## DM Parameters ##
The following are accessed through `inlist_DM`:

1.  DM mass (`mxGeV [GeV]`)
2.  DM-proton cross section (`sigmaxp [cm^2]`)
3.  Boost in DM capture rate relative to the solar environment (`cboost`, dimensionless). It encapsulates the environment's DM 1) density and 2) velocity dispersion.
4.  Spin dependent or independent scattering (`spindep`, Boolean).


## Functionality ##
The following is computed (`src/DM/DM_module.f90`) for each time step in a MESA run:
- Total number of DM particles captured.
- Number density of DM as a function of distance from the center.
- DM temperature.
- Amount of energy transported as a result of DM-nucleon scattering events, as a function of distance from the center of the star. This quantity is then passed to MESA via the `other_energy_implicit` hook (`src/run_star_extras.f`).
