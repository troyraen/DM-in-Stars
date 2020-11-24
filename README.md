# DM-in-Stars #

Explores the effects of dark matter (DM) on stellar evolution. Includes: (1) a module written to incorporate the effects of DM energy transport into the stellar evolution code MESA ([Modules for Experiments in Stellar Astrophysics](http://mesa.sourceforge.net)), and (2) supporting code to facilitate running MESA + DM models.

Code used for the paper _The Effects of Asymmetric Dark Matter on Stellar Evolution I: Spin-Dependent Scattering_ (Raen 2020) is in the [`Raen2020` branch](https://github.com/troyraen/DM-in-Stars/tree/Raen2020).

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
3.  Boost in DM capture rate relative to the solar environment (`cboost`, dimensionless). It encapsulates the environment's DM density and velocity dispersion. This variable is $\gamma_B$ in the Raen 2020 paper.
4.  Spin dependent or independent scattering (`spindep`, Boolean). Currently, only the spin dependent option is fully functional.


## Functionality ##
The following is computed (`src/DM/DM_module.f90`) for each time step in a MESA run:
- Total number of DM particles captured.
- Number density of DM as a function of distance from the center.
- DM temperature.
- Amount of energy transported as a result of DM-nucleon scattering events, as a function of distance from the center of the star. This quantity is then passed to MESA via the `other_energy_implicit` hook (`src/run_star_extras.f`).
