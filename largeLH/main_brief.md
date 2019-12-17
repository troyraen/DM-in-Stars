__Original run (1 M_sun, Gamma_B=10^6):__

This model's MS lifetime is much shorter than the Gamma_B=0 model.

Note L < LH and the oscillations:

<img src="lum_v_age_og.png" alt="lum_v_age_og"/>

Energy is not well conserved:

<img src="plots_r10398/rel_enery_error.png" alt="rel_enery_error.png"/>

---
__With the new "gold tolerances":__

L = LH and the oscillations are gone:

<img src="plots_r12115/dedt_gold/lum_v_age_c6_with_profile_nums.png" alt="lum_v_age_c6_with_profile_nums"/>

Energy conservation is much better:

<img src="plots_r12115/dedt_gold/rel_enery_error.png" alt="rel_enery_error"/>

---
The new models are taking a _long_ time to run. Gamma_B=10^6 originally took less than an hour, the new model has been running for about 60 hours when I made these plots. Despite having a higher luminosity than the Gamma_B=0 model, the central H1 fractions are very similar, so __I don't think the MS lifetimes are going to end up being much different__.

Comparison with Gamma_B=0 model:

<img src="plots_r12115/dedt_gold/lum_v_age.png" alt="lum_v_age"/>

<img src="plots_r12115/dedt_gold/center_h1.png" alt="center_h1"/>
