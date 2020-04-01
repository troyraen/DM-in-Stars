#### DM-in-Stars explores the effects of dark matter (DM) on stellar evolution. It adds a module to the code base Modules for Experiments in Stellar Astrophysics ([MESA](http://mesa.sourceforge.net)) which calculates the capture of DM by stars and the subsequent energy transport. The physical mechanism is the scattering of stellar nuclei with DM particles in the environment.

## Quick Start ##
1. Install MESA by following the instructions [here](http://mesa.sourceforge.net/prereqs.html).
2. Clone this repo and navigate into it.
3. Change settings in the `inlist` as desired (it is currently set to run a 1.0 Msun model).
4. Set the DM properties in `inlist_DM` as desired.
5. To start the model, execute `./clean`, `./mk`, and `./rn` from the command line (in the directory you cloned this repo to).

## DM Parameters ##
- DM-nucleon cross section (`sigmaxp` in [cm^2]), set in `src/DM/DM_module.f90`. (to do: configure this to be set in `inlist_DM`)
- DM mass (`mxGeV` in [GeV]), set in `src/DM/DM_module.f90`. (to do: configure this to be set in `inlist_DM`)
- Cboost factor (`cboost`, dimensionless), set in `inlist_DM`. This is a boost factor to the DM capture rate relative to the Sun's environment. It encapsulates the local DM density and its characteristic infall speed which are both unknown and degenerate with each other.
- Spin dependent or independent scattering (`spindep`, Boolean), set in `inlist_DM`. Spin dependence effectively means that only hydrogen is available for scattering.

## Functionality ##
The following is computed for each time step in a MESA run:
- Total number of DM particles captured
- Number density of DM as a function of distance from the center
- DM temperature
- Amount of energy transported as a result of DM-nucleon scattering events, as a function of distance from the center of the star. This quantity is then passed to MESA via the built-in `extra_heat_implicit` hook.
