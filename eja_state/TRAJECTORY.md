# TRAJECTORY -- the spine, in order

events 321 | corrections 60 | promotions+axioms 42 | correction:promotion ratio 60:42

final boundary (open tensions): [12, 13, 14, 19, 20, 24, 25, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42]


## Boundary size over the run

0 1 1 2 5 3 5 4 1 2 5 5 6 7 7 6 7 10 11 10 11 11 10 10 7 7 7 7 7 7 8 8 11 11 11 12 14 16 17 18 18


## The correction substream (every push-back, in order)

[   1] (refusal) REFUSED alpha1_attribution: distance 0.0049 vs scale 0.0042 (ratio 1.17) -- gap exceeds the sub-percent gate; eps measured flat, resolution transfer unresolved

[   4] (gate_raise) ENGINE REFUSED: invariance not observed; refusing promotion

[  54] (arc) ARC leray-ladder node 0 FAULTED under freed-pin decisive run, both flanks measured (arc stays LIVE; revisit due)

[  55] (tension_resolve_refuted) TENSION #2 REFUTED: Freed-pin solve VERIFIED as instrument (G0 4e-11) but the closure FAILED: pinned family self-parallels the identity line (dcl/dTH +1.677 vs 2/WX 1.672, 99.7% cancellation) so corner data cannot be det

[  62] (tension_resolve_refuted) TENSION #8 REFUTED: Freed-pin branch run did not produce a resolution-stable alpha -- it produced no converged freed root at all (both seeds stalled). Ghost verdict

[  63] (tension_resolve_refuted) TENSION #9 REFUTED: malformed entry (agent misfire, text='list') -- no content

[  76] (refusal) REFUSED dilation_exact_discrete_null: distance 0.00028 vs scale 2.81e-12 (ratio 99644128.11) -- The wedge dilation y->lambda y is an EXACT continuum symmetry (generator dA=G1*LA A, dB=G1*LB2 B - B, dP=G1*LPmu P - 2P, dcl

[  83] (refusal) REFUSED dilation_breaks_outer_neumann: distance 8.21e-08 vs scale 0.0017252 (ratio 0.00) -- I predicted the wedge dilation would be broken by the outer Neumann row d_xi Pt = 0 (a domain-truncation effect, since dilation 

[  84] (refusal) REFUSED grading_dominates_smallest_singular_values_of_the_generator: distance 3e-07 vs scale 0.0039701 (ratio 0.00) -- PREMORTEM PREMISE REFUTED FOR THE REDUCED OPERATOR. After the index-2 DAE reduction (P slaved, pins/C

[  94] (refusal) REFUSED hutchinson_trace_eigcount: distance 1.53 vs scale 1 (ratio 1.53) -- Hutchinson 40-probe estimator of the argument-principle eigenvalue count returned 2.530 and 2.513 where exact dense trace gives 1.0000; absolute

[  96] (refusal) REFUSED physical_L2dy_norm_representable: distance 1e+45 vs scale 1e+12 (ratio 999999999999999945575230987042816.00) -- The physical L2(dy) norm of (Om,Th) on the wedge has weight span 1.00e45 across xi in [0,25] because

[ 103] (refusal) REFUSED full_rhp_winding_sweep_feasible: distance 141 vs scale 1 (ratio 141.00) -- RHP D-contour total variation of arg det measured at 2pi x 335.8 rad for n_finite=376, i.e. 0.89 x n_finite because arg(z-lam_j) advances

[ 108] (refusal) REFUSED grading_deflation_is_cosmetic: distance 0.1955 vs scale 7.6e-05 (ratio 2572.37) -- D2 argued the explicit grading deflation 'buys 7.6e-05' and is therefore cosmetic. That number is the z=0 sigma_min effect ONLY (

[ 112] (refusal) REFUSED projnorm_P_resolution_stable: distance 0.501 vs scale 0.05 (ratio 10.02) -- ||P|| = 1/sin(theta_min) measured 489.8969 at (16,40,12) and 735.3920 at (24,56,12), +50.1 percent under a 1.35x increase in n_f, while 

[ 113] (refusal) REFUSED eps_cross_resolution_stable: distance 0.131 vs scale 0.02 (ratio 6.55) -- eps_cross on the imaginary axis at z=i: 3.96318e-03 at (16,40,12) vs 3.44344e-03 at (24,56,12), 13.1 percent apart; relative to ||L|| it i

[ 117] (refusal) REFUSED G2_headline_pairing_z0: distance 0.85 vs scale 1e-08 (ratio 85000000.00) -- SPEC quotes 'sparse bordered solve vs explicit dense reduction agree to 3.603e-13' as the G2 PASS. spectrum.py reproduces the SAME pair 

[ 118] (refusal) REFUSED G1_clause2_literal: distance 0.999983 vs scale 1e-10 (ratio 9999830000.00) -- Clause 2 as written demands ||Pi v_g||/||v_g|| < 1e-10 under the projector actually used. Measured with the structural (ker Cg) projec

[ 136] (refusal) REFUSED lipschitz_grid_certifies_rhp: distance 0.004 vs scale 1.4 (ratio 0.00) -- I planned to certify 'no spectrum in Re z>0' by a Lipschitz grid: spacing h certifies sigma_min >= grid_min - h/sqrt(2). MEASURED sigma_mi

[ 139] (refusal) REFUSED max_Re_We_is_zero: distance 514097 vs scale 0 (ratio 514096999999999967193641775394258749236935883541860171946278103691382331775914186197948577766231527815877724312722620500965741868908873026666569663479453566660

[ 149] (refusal) REFUSED rightmost_eigenvalue_is_measurable: distance 0.21392 vs scale 0.0043723 (ratio 48.93) -- Q6 cross-resolution eigenvalue matching, 4758 vs 6438 eigenvalues. The rightmost eigenvalue MOVES 2.1392e-01 in the complex

[ 160] (refusal) REFUSED eps_star_is_resolution_converged: distance 0.0008465 vs scale 1.175e-09 (ratio 720425.53) -- eps* = min over the closed RHP of sigma_min(zI-L), the crossing level, measured on a dense imaginary-axis scan at two r

[ 163] (refusal) REFUSED eps_star_relative_under_resolution: distance 0.6444 vs scale 0.000312 (ratio 2065.38) -- eps*/||L|| is invariant under eps_b to 3.12e-04 (3.533485e-06 at eps_b=1e-4 vs 3.534589e-06 at 5e-5, same grid) which tempt

[ 167] (refusal) REFUSED margins_are_truncation_artifacts: distance 3.364e-05 vs scale 0.3173 (ratio 0.00) -- MY OWN PREDICTION, REFUTED BY MY OWN MEASUREMENT. Tension #26 predicted that both the spectral abscissa and eps* are manufactur

[ 169] (tension_resolve_refuted) TENSION #26 REFUTED: Closed by direct measurement, against the prediction. XMAX 18 -> 25 moves the spectral abscissa by 3.364e-05 (0.011 percent) and eps* by +1.4 percent (away from the axis). The margin is not a truncat

[ 177] (refusal) REFUSED kreiss_constant_is_resolution_converged: distance 0.6588 vs scale 0.05 (ratio 13.18) -- K = sup_(Re z>0) Re(z)||R(z)||, refined on a 209-point positive-real-axis scan plus the 2-D RHP grids: K >= 8.965919e+01 at 

[ 196] (tension_resolve_refuted) TENSION #27 REFUTED: spurious: minted by a malformed 'ej tension list' invocation; no content

[ 197] (refusal) REFUSED unconditional_linear_stability_of_corner_profile: distance 0.3173 vs scale 8.823e-06 (ratio 35962.82) -- ADJUDICATION: the campaign may NOT claim 'the Chen-Hou corner profile is linearly stable'. Three independen

[ 200] (refusal) REFUSED hand_deflation_of_grading: distance 0.196 vs scale 0.01 (ratio 19.60) -- deflating the grading (what I proposed) moves ||R(+0.50)|| by 19.6/17.1/20.2 percent -- its restricted shadow is an ordinary stiff directio

[ 206] (refusal) REFUSED note_omega_over_normL_is_exactly_one_half: distance 0.00484 vs scale 0.5 (ratio 0.01) -- NOTE sec 6.2 writes 'omega(L)/||L|| = 1/2 EXACTLY (0.49516, 0.49666, 0.49523)'. The three quoted measurements sit 0.67-0.97

[ 212] (refusal) REFUSED NOTE_s7_ghost_moves_away_from_EVERY_named_target: distance 0.009107 vs scale 0.0001 (ratio 91.07) -- NOTE.md s7 says the ghost exponent 'moved AWAY from every named target with growing steps'. Measured from hunt_

[ 214] (refusal) REFUSED NOTE_s7_ground_moved_2.3e-3_over_a_decade: distance 0.00042441 vs scale 2.3e-05 (ratio 18.45) -- NOTE.md s7 pairs the branch's 1.3e-5 eps-flatness 'across a decade of eps_b' against 'where the ground state moved 

[ 215] (refusal) REFUSED NOTE_s8_polar_spectrum_implements_section6: distance 6 vs scale 1 (ratio 6.00) -- NOTE.md s8: 'polar_spectrum.py implements Section 6 with its gates runnable in place.' Read the file: /Users/epagogellc/parzival/b

[ 217] (refusal) REFUSED NOTE_s8_cornerreg_PRINTS_h_id_at_every_convergence: distance 1 vs scale 0.05 (ratio 20.00) -- NOTE.md s8 and NOTE_CLAIMS M6 (graded ENFORCED) say polar_cornerreg.py 'prints h_id at every convergence'. The code RE

[ 218] (refusal) REFUSED NOTE_s7_corner_algebra_is_one_object: distance 2.386 vs scale 0.000276 (ratio 8644.93) -- NOTE.md uses 'corner algebra' for two different objects in the same document, and s7 lands both on the same state. s2.3: '

[ 226] (refusal) REFUSED REFERENCE_INSIDE_MODEL_SPREAD: distance 5.41e-05 vs scale 2.96e-06 (ratio 18.28) -- note 4.1 asserts the reference -0.34240009 lies inside the three-fit spread [-0.34245715,-0.34240305]; it lies 2.96e-6 ABOVE the

[ 231] (refusal) REFUSED THREE_FIT_FAMILIES_AICC_INDISTINGUISHABLE: distance 1 vs scale 0 (ratio 99999999999999990380306940742611396889821876611810314178983394957235655241172226419230565904001050952687299421724881919707014421606312553018

[ 232] (refusal) REFUSED TIGHTEN_LADDER_ABSCISSA: distance 2.5e-05 vs scale 3e-05 (ratio 0.83) -- final_ladder.py drives rung 3 at eps_b=2.5e-5 (loop tuple (24,2.5e-5)) but prints it with f'{eps:8.0e}', which renders 2.5e-5 as '3e-05' (v

[ 233] (refusal) REFUSED AXES_CLOSED_MEASURED_OFF_LADDER: distance 0.0104 vs scale 5e-05 (ratio 208.00) -- The axes-closed table presents its worst-effect numbers as closing axes behind C1's alpha, but two of the three closed rows were m

[ 237] (refusal) REFUSED SHARES_NO_MACHINERY_WITH_EITHER_EXISTING_COMPUTATION: distance 1.79819 vs scale 1.79819 (ratio 1.00) -- polar_cornerreg.py:80 WX_REF,THXX_REF = 1.19620314, 1.79819132, sourced from polar_gauge_gate.py which loadm

[ 238] (refusal) REFUSED CERTIFY_STABILITY_WITHOUT_COMPUTING_ANY_EIGENVALUE: distance 4758 vs scale 4758 (ratio 1.00) -- scipy 1.18.0 solve_continuous_lyapunov (q345_lyap.py:37) begins r,u = schur(a, output='real') -- it computes a real 

[ 240] (refusal) REFUSED NO_SHARED_CONSTANT_LEFT_STANDING_BEHIND_C1: distance 10 vs scale 2 (ratio 5.00) -- ej audit over the five ladder rungs of final_ladder.log returns TEN shared constants: Nb=36, THXX=1.79819132, WX=1.19620314, XMAX

[ 241] (refusal) REFUSED FREE_RESIDUAL_SEPARATES_REAL_FROM_FALSE: distance 1 vs scale 1 (ratio 1.00) -- N4's insufficiency half is valid from one counterexample. Its converse half -- that a free residual IS what separates real from false

[ 242] (refusal) REFUSED RIGHTMOST_EIGENVALUE_PROVEN_UNMEASURABLE: distance 2 vs scale 5 (ratio 0.40) -- ALPHA_RESULT s6.4 writes 'proven unmeasurable at these resolutions' and S9 states 'The rightmost eigenvalue is not measurable'. The 

[ 244] (refusal) REFUSED alpha4_independent_confirmation: distance 1 vs scale 1 (ratio 1.00) -- NOT independent: my geometric-gap prediction was built from THEIR alpha_0..alpha_3, and their lambda=1/(an+b)+1 form has lambda_inf=1 BY CONS

[ 258] (tension_resolve_refuted) TENSION #29 REFUTED: T-SCAN REFUTES t-mismatch. Pre-registered rule required a clean |h_id| minimum well below 2.39 at some t != t_ground. Measured: |h_id| is FLAT at 2.18-2.25 across the ENTIRE scan (t=1.17 -2.1835, 1.2

[ 259] (refusal) REFUSED leray_value_at_t117: distance 2.1835 vs scale 0.001 (ratio 2183.50) -- the t=1.17 root's cw/cl=-0.501072 sits 3 digits from the Leray -1/2, but its h_id is -2.18: it is an identity-violating artifact like every o

[ 261] (arc) ARC leray-ladder node 0 FAULTED under t-scan, six pinned corner-invariant values (arc stays LIVE; revisit due)

[ 270] (inoc_check) STAGE 1 NULLIFIED: Newton STALLED at ||F||=1.21e-5 (needed 1e-11) after 12 steps and sigma_min = 5.83e-10, an order below the 1e-9 kill line and 4300x worse than the standard syste

[ 271] (preflight) PREFLIGHT over 275 recorded entries: 3 collision, 2 support, 0 orphan, 2 neutral

[ 281] (tension_mint) TENSION #35 (open): COST CORRECTION on euler3d-homotopy stage 1, self-caught: I costed it '~2 minutes, one solve'. The SOLVE is 2 minutes; the IMPLEMENTATION is not. The four corrections must be rewritten in the divided 

[ 284] (tension_mint) TENSION #36 (open): REPLACEMENT READOUT for the 3D path: the measurable is d alpha / d eps at eps = 0, the first-order sensitivity of the Boussinesq exponent to genuine 3D structure, NOT alpha(tau=1). It is ONE linear so

[ 286] (refusal) REFUSED euler3d_homotopy_endpoint_is_alpha3d: distance 1 vs scale 0 (ratio 9999999999999999038030694074261139688982187661181031417898339495723565524117222641923056590400105095268729942172488191970701442160631255301862676

[ 305] (refusal) REFUSED p_mechanism_transport_transpose: distance 0.32 vs scale 0.04 (ratio 8.00) -- ADJUDICATED (adjoint-balance lens, computed): my mechanism paired C ~ rho/xi^2 with a transport-transpose adjoint w ~ rho^alpha. WRONG 

[ 307] (refusal) REFUSED coefficient_computable_from_inner_region: distance 2 vs scale 1 (ratio 2.00) -- ADJUDICATED (cutoff-validity lens, serious, computed): the drift AMPLITUDE is not computable from the inner region. (i) 73-80% of th

[ 308] (deduction) DEDUCTION timedomain_linear_approach_v2 [NOVEL] drift_time_exponent_halfwidth = 0.1

[ 311] (tension_mint) TENSION #41 (open): FARFIELD EVIDENCE REPAIR (farfield lens, not refuted, minor): kappa=1 is CONFIRMED pointwise (1.000 +- 0.001 on 30/35 betas, rms 5.7e-6 at the wall column) but two of MY evidentiary steps were defecti

[ 314] (refusal) REFUSED stage0_degree_screen_on_generic_fields: distance 1 vs scale 0.05 (ratio 20.00) -- ADJUDICATED (corner-degrees lens, serious): stage 0 certified C1 as +1 subleading by degree-counting on GENERIC fields. On the SOL

[ 315] (refusal) REFUSED raw_dalpha_deps_scalar: distance 1 vs scale 1e-11 (ratio 100000000000.00) -- ADJUDICATED (limits lens, serious): the raw unwindowed dalpha/deps scalar means nothing on this grid. Only 463/3132 nodes (14.78 percen

[ 319] (tension_mint) TENSION #42 (open): C1's corner action is a GAUGE RENORMALIZATION: WX -> WX - 6*tau*c at corner order 1, the order where alpha is first selected (order 0 closes as h_id = 0 identically and is alpha-blind). Consequences: 

[ 320] (inoc_check) STAGE 1 NULLIFIED: Triple kill, no solve run and none needed: (1) the endpoint was refused independently (alpha_3D = alpha_2D in the exact self-similar limit, so tau=1 solves a sna


## Open boundary, final state

  #12: TENSION #12 (open): Corner-circle pin and the two d1 gauge rows encode the SAME two continuum numbers (wx,thxx); the realization's admissible space is 2 dims SMALLER than the physi

  #13: TENSION #13 (open): Axis column (beta=pi/2-eps_b) pinned to seed data is an extra Dirichlet BC on the perturbation. Cost on alpha measured 3e-8; cost on the SPECTRUM unmeasured.

  #14: TENSION #14 (open): The corner pin deletes the continuum 2-D corner ODE (Wdot,Tdot) from the realization: the computed spectrum is that of the CORNER-CLAMPED dynamics.

  #19: TENSION #19 (open): The realization itself (E = identity on live transport rows, pencil (E,J) Hessenberg index-2) is a SHARED CONSTANT behind every spectral number measured this se

  #20: TENSION #20 (open): The rank-2 singular solve np.linalg.solve(Pi_c M, f) is ill-posed (measured rel spread 5.3e-1 .. 1.1e0 over 8 seeds) yet it is the route behind the SPEC's headl

  #24: TENSION #24 (open): The measured 0.33 s per grid point is memory-bandwidth bound: each sigma_min needs ~20 inverse-power iterations, each two ztrtrs sweeps over the 362 MB Schur fa

  #25: TENSION #25 (open): The Lyapunov decay rate 1/(2 lambda_max P) = 8.823e-06 is 3.6e4 times SMALLER than the Schur spectral abscissa 0.3173. The certificate proves Hurwitz but its ra

  #30: TENSION #30 (open): H2 DERIVATION (by hand): the next-order corner Taylor identity beyond cl=2THXX/WX; measure its transversality to the broken-dilation direction on the pinned gro

  #31: TENSION #31 (open): GROUND TIME-MARCH (dedalus_bsq, existing code): perturb the converged profile, march, compare decay against exp(tL) from the certified generator. Closes the E=I

  #33: TENSION #33 (open): REDIRECT after t-scan: Gate A passed, so alpha-Newton works and is unused. The failing component is deflated multistart at FROZEN alpha, which produces identity

  #34: TENSION #34 (open): 3D REFERENCE DATA GAP (surfaced by preflight, not by reasoning): the entire certification apparatus (h_id, the Chen-Hou validation gate) is anchored to hardcode

  #35: TENSION #35 (open): COST CORRECTION on euler3d-homotopy stage 1, self-caught: I costed it '~2 minutes, one solve'. The SOLVE is 2 minutes; the IMPLEMENTATION is not. The four corre

  #36: TENSION #36 (open): REPLACEMENT READOUT for the 3D path: the measurable is d alpha / d eps at eps = 0, the first-order sensitivity of the Boussinesq exponent to genuine 3D structur

  #38: TENSION #38 (open): THE REAL FRONTIER STATEMENT for 3D: the sensitivity is dominated by the far field, which is precisely where the near-wall map R = 1 - eps*y is INVALID. The eps^

  #39: TENSION #39 (open): The last decade before the boundary carries 0.908 of total |shell| mass vs 0.77 expected from a clean rho^0.64 density, a mild excess consistent with the sweep'

  #40: TENSION #40 (open): SIDE-DISCOVERY from the battery: refining the inner/mid panels (16,40)->(20,48) at eps_b=1e-4 moved h_id from -1.06e-3 to +1.56e-4: a 7x shrink AND a sign cross

  #41: TENSION #41 (open): FARFIELD EVIDENCE REPAIR (farfield lens, not refuted, minor): kappa=1 is CONFIRMED pointwise (1.000 +- 0.001 on 30/35 betas, rms 5.7e-6 at the wall column) but 

  #42: TENSION #42 (open): C1's corner action is a GAUGE RENORMALIZATION: WX -> WX - 6*tau*c at corner order 1, the order where alpha is first selected (order 0 closes as h_id = 0 identic
