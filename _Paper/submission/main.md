- [Publication links](#links)
- [Reviewer Comments - Round 2](#revcomm2)
- [Reviewer Comments - Round 1](#revcomm1)
- [Prepare files for Zenodo](#zen)
---

<a name="links"></a>
# Publication links
<!-- fs -->
Submissions:
- 10/8/2020: Zenodo, arXiv, MNRAS
- 2/25/2021: arXiv, MNRAS

Zenodo: [https://doi.org/10.5281/zenodo.4064115](https://doi.org/10.5281/zenodo.4064115)

arXiv:
- Overleaf: [download the automatically generated files (.bbl)](https://www.overleaf.com/learn/how-to/How_do_I_download_the_automatically_generated_files_%28e.g._.bbl,_.aux,_.ind,_.gls%29_for_my_project%3F_My_publisher_asked_me_to_include_them_in_my_submission)
- temporary submission identifier is: submit/3394413
- You may update your submission at: https://arxiv.org/submit/3394413
- permanent arXiv identifier 2010.04184 and is available at: http://arxiv.org/abs/2010.04184
- https://mail.google.com/mail/u/0/#inbox/FMfcgxwKhqjGfxsQXDJcwGMnfHPJcqzr

Monthly Notices:
- Manuscript ID: MN-20-4148-MJ
- https://mc.manuscriptcentral.com/mnras
- email https://mail.google.com/mail/u/0/#inbox/FMfcgxwKhqddwPCWMJctRsxFRbHpCqKh
- __[Editor/Reviewer comments](#revcomm)__
    - reviewer email https://mail.google.com/mail/u/0/#search/monthly+notices/FMfcgxwKjKvblcRlpZRSzPcbLvFhMSnB

<!-- fe # Publication links -->
---

<a name="revcomm2"></a>
# Reviewer Comments - Round 2
<!-- fs -->
- - The interpretation of the statement in section 4.1, “The result is that the increase in MS lifetimes peaks at Gamma_B ∼ 10^3 after which MS lifetimes decrease with increasing environmental factor” is not straightforward. The wording is correct, and I leave it to the authors to decide if they want to keep it in the final version as it is now. However, I note that this sentence could still be interpreted as if for gamma_B>10**3 the MS lifetimes decrease compared to the no-DM scenario, which is not what the authors want to state. The correct interpretation is much clearer to the reader when looking at Figure 3.
    - __We have updated this statement to better characterize the decreasing MS lifetimes.__

- I interpret the statement in the caption of Figure 1, “The gamma_B=10^6 model reaches the Xc evolutionary markers at older ages, relative to the NoDM model…”, as saying that 1 Msun stars in Gamma_B=10^6 have their MS lifetimes increased compared to no DM. However, Figure 3 shows that 1 Msun stars in Gamma_B=10**6 have their MS lifetimes decreased by about 5% compared to no DM. The authors should either investigate if these are contradictory results or clarify why my interpretation of the statement in Figure 1 caption and Figure 3 is wrong and improve the clarity of how they present the results accordingly.
    - __This statement was incorrect and we have removed it__. See the ["1.0 Msun evolutionary ages" section of figures/main.md](../Paper/figures/main.md#m1p0-ages). The Gamma_B = 10**6 model generally reaches each evolutionary marker at a younger age (except for the X_c = 0.3 marker), however the difference in ages is generally small and the timesteps (and therefore the exact X_c values) are not exactly aligned. Therefore we simply removed the statement.
<!-- fe Reviewer Comments - Round 2 -->
---

<a name="revcomm1"></a>
# Reviewer Comments - Round 1
<!-- fs -->
- [ ]  Evaporation of DM particles outside the star:
    - The study concentrates on a hypothetical ADM particle with a mass of 5 GeV, which is just marginally above the lower limit for evaporation in the Sun as found by Gould [1] (see also more modern detailed studies of evaporation of DM in the Sun in Refs [2] and [3]). Considering that this limit is very sensitive to stellar parameters and that __evaporation could potentially invalidate the results presented in this paper, the authors should take evaporation into account__. That could be done by implementing the full treatment of evaporation in the code (which is fairly similar to the energy transport by scattering) or __by justifying in the text why evaporation can be neglected for the DM particles and stellar masses that they study__.

[Troy]: I think that the ADM simply does not get far enough out to evaporate. I could check for the radius at which the density goes to <_insert threshold_> and cite it as justification for ignoring evaporation. Also see 'Troy review [Ref [4]]' below. Add to 1st paragraph of section 2.

- [ ]  Literature: The presented results are not compared with the existing literature thoroughly enough. This is necessary not only to give the appropriate credit to previous works, but crucially to highlight which results are novel, which presently is not clear for the reader of the manuscript.

    - [x]  The __reduction of burning rate in the center of the star, and increase in a shell__, was already reported by Ref [4] for the Sun and Ref [5] for main sequence stars, among others. Although you already cite these works elsewhere, __their previous results should be mentioned when you present your findings in section 4.1, and your results should be compared to previous ones. Are they in agreement?__

        - _Troy review [Ref [4]](https://arxiv.org/pdf/1005.5711.pdf):_ studies several DM models, ADM is one (__results sec VI; Fig 2__),
            - __environment: sun.; m_X = 7 GeV, sigma_SD = [1,2,3]*10^-36__
            - their Fig 2, our Fig 1:
                - Temp results seem consistent, but not sure how carefully I need to check. (note: Their NoDM T_c ~1.55e7 K, ours is T_c ~1.41e7)
                - Their eps_X seems high, but they use a larger cross section (m_x = 5 GeV here).
            - "Lowering the DM mass goes in the direction of maximizing the transport effects, but also the evaporation rate. __Above mχ= 5 GeV, evaporation can be safely neglected__ and the mass of the particle acts to modify the radius and normalization of DM inside the star." evaporation further discussed in __appendix A 3__.

        - _Troy review [Ref [5]](https://arxiv.org/pdf/1201.5387.pdf):_
            - again results seem consistent, but not sure how closely I need to check.
            - __Fig 1__: "environmental DM density ρχ=10^3GeV/cm3 and ¯v=220 km/s adopted in this run"
            - __sec Stars in high ADM densities; Fig 2__: 1Msun star, vary DM density. m_X = 10 GeV, sigma_SD = 10^-37 cm^2.  stellar velocity through the DM halo ¯v=220 km/s. DarkStars code. They find oscillations which we have addressed. They find |eps_DM| > |eps_nuc|.. maybe I should check if any of our models meet that condition. that result may also have been a result of the numerical artifacts

    - [x]  __Suppression of core convection__ due to DM was already reported, among others, by Refs [6-8]. Although you already cite most of these works elsewhere, __a direct comparison of your results is due in section 4.2. Do you extend the existing results in any way?__
        - _Troy review [Ref [6]](http://articles.adsabs.harvard.edu/pdf/1987A%26A...171..121R):_ theoretical calculation, shows suppression of core convection. proton cross section 10^-36 cm^2, m_X = [4-60] GeV. Focus on HB stars. "net effect would be a drastic reduction of the HB lifetime, accompanied by a drastic increase in AGB lifetime. This can certainly be ruled out by stellar counts in globular clusters". Do not mention SD vs SI (perhaps the concept wasn't around yet?).
        - _Troy review [Ref [7]](https://arxiv.org/pdf/1212.2985.pdf):_  
            - "ADM modifies the central temperature and density of low-mass stars and suppresses the convective core expected in 1.1-1.3 M stars even for an environmental DM density as low as the expected in the solar neighborhood"
            - __Fig 1__: T and rho for for model of star KIC 8006161 (see table 1), range of m_X and sigma_SD
            - __Fig 2__: convective core for models of star HD 52265 (see table 1)
        - _Troy review [Ref [8]](https://arxiv.org/pdf/1505.01362.pdf):_
            - "Studying this star (KIC 2009505) we found that the asymmetric DM interpretation of the results in the CoGeNT experiment is incompatible with the confirmed presence of a small convective core in KIC 2009505."
            - mass range: 1.1 − 1.35 Msun; DM params: in table II (cogent m_X = 8 GeV, sigma_SD = 10^-33 cm^2); environment: "dispersion ¯vχ = 270 km s−1 and a local DM density around the star ρχ = 0.4 GeV cm−3, as estimated in the solar neighborhood"

    - [ ]  Ref [9] studied the impact of ADM SD scattering in low-mass stars, and reported __changes in MS lifetimes, suppression of core convection and showed isochrones with the absence of the convective hook__. Your results should be compared to their results. Up to what extent do they agree? What is new in your work?
        - _Troy review [Ref [9]](https://arxiv.org/pdf/1907.05785.pdf)_:
            - environment: nuclear star cluster (I assume this is gamma_B ~ 10^3?)
            - increase MS lifetimes, suppress core convection
            - mass range: 1-2 Msun; m_X = 4 GeV; sigma_SD = 10−37 cm2

- [ ]  Potentially observable signatures:
    - In the second paragraph of section 5, the authors speculate on how the reported signatures could be used to investigate the nature of dark matter or have an impact in the analysis of stellar populations. They discuss that it “requires very high-quality observations of a stellar population residing in an environment with a large ambient dark matter density”, like the Local Group dwarf galaxies that are mentioned elsewhere in the text. Considering the importance of a hypothetical comparison of the reported results with observations, the __conclusions would be much more complete with a short discussion on whether these “very high-quality observations” are within the reach of present and planned missions__.

- [ ]  Clarity: In the second paragraph of section 2.1, the authors state that “The result is that the __MS lifetime of the star decreases with increasing environmental factor for gamma_B>10\*\*3”__, whereas in the last sentence of the caption of Figure 1 they state that “The __gamma_B=10**6 model reaches the Xc evolutionary markers at older ages, relative to the NoDM model__, because the changes to the burning rates cause the central hydrogen fraction to decline more slowly.”. Putting these 2 sentences together, it seems that, for high gamma_B, the MS lifetime of the star decreases and at the same time the central hydrogen fraction declines slowlier. Either these results are contradictory or they are not clearly enough explained. In either case, the authors should review the results and the wording of the text.

[Troy]: see Fig 1, "reduces the burning rate in the center" not true for X_c=0.3.
- Plot X_c vs time for both models to see if last sentence in caption is true.
- Plot rho_c vs time for all gamma_B to see if the turnover in MS lifetimes corresponds to any changes in density, and to check that density is increasing significantly.
- Pols equ 6.47 and 6.51 eps_pp linear in density, temp power between 3.5 and 6 (cno also linear in density)

Additional minor points that the authors may consider addressing if they find it appropriate:

- [x]  In the second-last paragraph of __section 1, the authors briefly summarize the results of the paper__. While this is in general __not needed__ in the Introduction of an article, if the authors prefer to do it, I would recommend to better explain the different results. The present summary focuses on stellar lifetimes and ignores other interesting and potentially observable signatures in clusters.

- [ ]  In the second paragraph of section 2, when the authors comment on the recent results of the [PICO collaboration](https://arxiv.org/pdf/1902.04031.pdf), I note that it may be of interest of the reader to __explicitly state which are the limits set by PICO for mx=5GeV__.

- [-]  The parametrization used by the authors for the capture rate (Eq. 2) is indeed very useful due to the degeneracy and uncertainties in the full capture calculation. However, it may be worth to discuss the capture process more thoroughly. The authors may consider to implement the __full capture rate calculation__ (the code is publicly available, see [10]) instead of the approximation used in Eq. 1 in future works. The __discussion on the impact of DM velocity distribution__, relevant for dwarf galaxies, may also be enriched with the insights of a very recent work on the topic [11], while the __discussion on the impact of different DM particle properties could also consider studies on momentum and velocity dependent interactions [12]__.

- [x]  In the __1st paragraph of section 5, the absence of the convective hook__ is missing from the summary of conclusions.

- [-]  In Figures 4, 5 and 6, the __axes labels are very large__, in particular when compared with Figure 8, which shows more proportionate letter sizes.


[1] A. Gould, Evaporation of WIMPs with arbitrary cross sections, ApJ 356 (1990) 302–309.
[2] 1208.0834
[3] 1703.07784
[4] 1005.5711
[5] 1201.5387
[6] Renzini, A. 1987, A&A, 171, 121
[7] 1212.2985
[8] 1505.01362
[9] 1907.05785
[10] 0904.2395
[11] 2007.15927
[12] 1703.07784

<!-- fe Reviewer Comments -->

---
<a name="zen"></a>
# Prepare files for Zenodo
<!-- fs -->
```bash
# on Osiris
cd DMS/mesaruns/
mkdir zenodo_Raen2020
# copy inlist templates, and src dir to newdir/supporting_code
# copy model data from RUNS_FINAL to newdir/model_data

# on Osiris
tar -czvf supporting_code.tar.gz supporting_code
# on Roy
scp tjr63@osiris-inode01.phyast.pitt.edu:/home/tjr63/DMS/mesaruns/zenodo_Raen2020/supporting_code.tar.gz Downloads/zenodo_Raen2020/.

# on Osiris
# having trouble uploading the full tarball; exclude large files
tar -czvf model_data.tar.gz --exclude=model_data/1.0_Msun_data/log_GammaB_6/LOGS/history.data --exclude=model_data/1.0_Msun_data/log_GammaB_6/LOGS/STD.out --exclude=model_data/1.0_Msun_data/log_GammaB_4/LOGS/history.data --exclude=model_data/*/*/photos model_data
# on Roy
scp tjr63@osiris-inode01.phyast.pitt.edu:/home/tjr63/DMS/mesaruns/zenodo_Raen2020/model_data.tar.gz Downloads/zenodo_Raen2020/.

```
<!-- fe Prepare files for Zenodo -->


# Collaborators

- [x]  Andrew
- [x]  Carlos
- [x]  Travis
- [x]  Hector
- [x]  Rachel
