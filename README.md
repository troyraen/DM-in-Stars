#### MESA_WIMPS adds modules to the base code Modules for Experiments in Stellar Astrophysics (MESA). These modules calculate the capture of WIMPs and the resulting energy transport. The mechanism for both is the scattering of WIMPs in the stellar environment with stellar nuclei.

# Physics ####
####   Since WIMPs have a very small cross section they have a large mean free path which means they can transport energy across large distances. At high enough densities the WIMPs can transport enough energy to affect the evolution of the star in an observable way, via stellar cluster isochrones on HR diagrams.

# MESA_WIMPS #####

## Parameters ##
####   Whether scattering is spin dependent or independent. Spin dependence means that only hydrogen is available for scattering.
####   Cboost factor. This is a boost factor to the WIMP capture rate relative to the Sun's environment. It encapsulates the local WIMP density and their characteristic infall speed which are both unknown and degenerate with each other.
####   WIMP cross section.

## Function ##
####   The following is computed for each time step (in a MESA run):
####       The total number of WIMPs captured
####       The number density of WIMPs as a function of distance from the center
####       The amount of energy transported as a result of WIMP-nucleon scattering events as a function of distance from the center

## Output ##
#### The amount of energy transported by WIMPs (as a function of 'zone') is passed to MESA via the built-in 'extra_heat' hook.
