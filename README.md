#### DM-in-Stars explores the effects of dark matter (DM) on stellar evolution. It adds a module to the code base Modules for Experiments in Stellar Astrophysics ([MESA](http://mesa.sourceforge.net)) which calculates the capture of DM by stars and the subsequent energy transport. The physical mechanism is the scattering of stellar nuclei with DM particles in the environment.

## Quick Start ##
1. Install MESA by following the instructions [here](http://mesa.sourceforge.net/prereqs.html).
2. Clone this repo ([git@github.com:troyraen/DM-in-Stars.git](git@github.com:troyraen/DM-in-Stars.git)) and navigate into it.
3. Change settings in the `inlist` as desired (it is currently set to run a 1.0 Msun model).
4. Set the DM properties in `inlist_DM` as desired.
5. To start the model, execute `./clean`, `./mk`, and `./rn` from the command line (in the directory you cloned this repo to).

## Parameters ##
- DM-nucleon cross section, set in `src/DM/DM_module.f90`.
- DM mass, set in `src/DM/DM_module.f90`.
- Cboost factor, set in `inlist_DM`. This is a boost factor to the DM capture rate relative to the Sun's environment. It encapsulates the local DM density and its characteristic infall speed which are both unknown and degenerate with each other.
- Spin dependent or independent scattering, set in `inlist_DM`. Spin dependence effectively means that only hydrogen is available for scattering.

## Function ##
The following is computed for each time step in a MESA run:
- Total number of DM particles captured
- Number density of WIMPs as a function of distance from the center
- Amount of energy transported as a result of DM-nucleon scattering events as a function of distance from the center of the star

## Output ##
The amount of energy transported by DM (as a function of 'zone') is passed to MESA via the built-in 'extra_heat' hook.
