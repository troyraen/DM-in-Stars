
# Results
<!-- fs -->

Paragraph 18. Standard stellar evolution. Qualitative difference between high and low mass stars.


## Low Mass
<!-- fs -->
Low mass stars (Mstar < 1.3 Msun)
- keep first paragraph?: low central temps -> powered by pp chain -> burning is less sensitive to tempreature and doesn't peak as strongly -> radiative cores.

- pp -> lower burning rate -> radiative
- DM reduces temp in center and increases in shell (new figure 1: 1Msun profiles burning and temp)
- burning rates change similarly
- since burning rate in shell increases, these stars burn more total fuel (need to check that this is true).

Put the following in the appendix?
- previous paper (Iocco, fig 3) saw signifiant oscillations in L, Teff and noted that they may be numerical artifacts.
- when we ran MESA models using old energy conservation scheme we saw similar oscillations. they were a result of the DM temperature falling below the central temp -> DM moves energy back to the center -> temp increases -> DM move energy away from the center. This resulted in large fluctuations in the burning rate profiles which propagated to the surface and manifested in oscillations in the size of the star (radius) -> Teff and L.
- DM temperature falling below the central temp turned out to be caused by numerical artifacts and resulted in poor energy conservation. Should I mention that the lifetimes were greatly reduced?
- We ran newer MESA version with improved energy conservation scheme and the oscillations disappeared.

<!-- fe ## Low Mass -->

## High Mass
<!-- fs -->
- keep paragraph 1? Move to a different section?
- keep paragraph 2: effects of DM on convection -> MS lifetimes
- walk a DM person through profiles figure
    - eps dm negative values indicate that DM is removing energy from this region
    - describe the _features_ in the plot that a DM person should identify (shell burning: eps_nuc zero at center and larger a little farther out)
    - annotate panels: 'zero age main sequence', 'intermediate MS'
    - in caption, panels are numbered, also include the y variable name

<!-- fe ## High Mass -->

## MS Lifetimes

- We find that MS lifetimes ... (delta Tau figure)
    - high mass stars
    - add mass scaling relations to the end. capture rate scales linearly with stellar mass, MS lifetimes scale as M^-2.5 (check exact number) -> effects drop off rapidly with increasing mass.
    - __low mass stars__ with gammaB = 10^2-4 live slightly longer. is this because burning rate in center is reduced?
    - __low mass stars__ with gammaB = 10^6 (not sure yet about 5) show almost no change in MS lifetime. why does the effect go away? maybe because temperature is so low the central burning shuts off earlier?


## Isochrones
- isochrones look older primarily because they are at a lower luminosity at fixed age
- then show plot of hottest Teff and L vs isochrone age

<!-- fe # Results -->


# Abstract
<!-- fs -->
### from Brett
search for DM. ADM yet to be ruled out by obs. we consider its effect on stellar evolution. because can scatter with baryons can be collected in stars, transport energy, may be observable. we use mesa to study effects of ADM on stars from 1-5 solar masses. write a module to do the DM energy transport which we make publicly avail (footnote). test several strengths of DM to simulated different galactic evirons. stars with radiative cores <1.3Msun have small effects in the core, not observable. however, in stars with convective cores, dm transports energy, shuts off convection, starving the core of a replenished fuel supply, reduces MS lifetime. decreases .


### first draft
Most of the DM search over last few decades has focused on WIMPs but the viable parameter space is quickly shrinking. ADM is an alternative DM theory that predicts masses slightly smaller than the standard WIMP and no present day annihilation (should also mention how cross sections compare to WIMPS). These properties mean that stars can capture and build up potentially large quantities of ADM over their lifetimes. Further, the captured ADM would orbit and transport energy through a significant volume of the star, potentially affecting stellar evolution in observable ways.

We investigate the effects of ADM energy transport on stellar structure and the resulting changes in main sequence (MS) lifetimes.

We use the MESA stellar evolution code to study stars with 0.8 <= Mstar/Msun <= 5.0 in varying DM environments. We wrote a module (footnote with repo link) that integrates with MESA and calculates the capture of DM and the subsequent energy transport within the star. We fix the DM mass to 5 GeV and cross section to 10^-37 cm^2, and we study varying environments by scaling the DM capture rate.

We find that in stars with Mstar <~ 1.3 Msun, the presence of ADM flattens the temperature and burning profiles in the core, but that the effect on MS lifetimes and observable properties is small. However, in stars with Mstar >~ 1.3 Msun, ADM energy transport shuts off convection in the core, limiting the fuel available and therefore shortening MS lifetimes by as much as ~50%. This translates to changes in the luminosity and effective temperature of the MS turnoff in stellar population isochrones.


### draft 2

Most of the DM search over last few decades has focused on WIMPs but the viable parameter space is quickly shrinking. ADM is a WIMP-like DM candidate with slightly smaller masses and no present day annihilation, meaning that stars can capture and build up large quantities of it. The captured ADM can transport energy through a significant volume of the star. We investigate the effects of spin-dependent ADM energy transport on stellar structure and evolution in stars with 0.8 <= Mstar/Msun <= 5.0 in varying DM environments.

We wrote a publicly available MESA module (footnote with repo link) that calculates the capture of DM and the subsequent energy transport within the star. We fix the DM mass to 5 GeV and cross section to 10^-37 cm^2, and we study varying environments by scaling the DM capture rate.

For stars with radiative cores (Mstar <~ 1.3 Msun), the presence of ADM flattens the temperature and burning profiles in the core, but the effect on MS lifetimes and observable properties is small. However, in stars with Mstar >~ 1.3 Msun, ADM energy transport shuts off convection in the core, limiting the fuel available and therefore shortening MS lifetimes by as much as ~50%. This translates to changes in the luminosity and effective temperature of the MS turnoff in stellar population isochrones.


<!-- fe # Abstract -->
