#### DM-in-Stars explores the effects of dark matter (DM) on stellar evolution. It adds a module to the code base Modules for Experiments in Stellar Astrophysics ([MESA](http://mesa.sourceforge.net)) which calculates the capture of DM by stars and the subsequent energy transport. The physical mechanism is the scattering of stellar nuclei with DM particles in the environment.

## Quick Start ##
1. Install MESA by following the instructions [here](http://mesa.sourceforge.net/prereqs.html).
2. Clone this repo and navigate into it.
3. Change settings in the `inlist` as desired (it is currently set to run a 1.0 Msun model).
4. Set the DM properties in `inlist_DM` as desired.
5. To start the model, execute `./clean`, `./mk`, and `./rn` from the command line (in the directory you cloned this repo to).

## DM Parameters ##
The following are accessed through `inlist_DM`:

1.  DM mass (`mxGeV [GeV]`)
2.  DM-proton cross section (`sigmaxp [cm^2]`)
3.  Boost in DM capture rate relative to the solar environment (`cboost`, dimensionless). It encapsulates the environment's DM 1) density and 2) velocity dispersion.
4.  Spin dependent or independent scattering (`spindep`, Boolean).

## Functionality ##
The following is computed for each time step in a MESA run:
- Total number of DM particles captured
- Number density of DM as a function of distance from the center
- DM temperature
- Amount of energy transported as a result of DM-nucleon scattering events, as a function of distance from the center of the star. This quantity is then passed to MESA via the built-in `extra_heat_implicit` hook.
