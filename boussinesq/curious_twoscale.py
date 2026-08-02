#!/usr/bin/env python3
"""CURIOUS track K1 (UNGRADED -- see CURIOUS.md; nothing here enters
NOTE_CLAIMS.md without passing STANDARD.md).

Question: does the full 2D Boussinesq system admit a STRICT two-scale
power-law collapse -- different similarity exponents in the two directions --
and if so, what does the equation force?  Pure order counting on the ansatz.
No profile equation is solved, no boundary conditions imposed.

System (half-plane, x along the wall):
    w_t  + u w_x  + v w_y  = th_x        u = psi_y,  v = -psi_x
    th_t + u th_x + v th_y = 0           Lap psi = w

Ansatz (directions labelled by collapse speed, not by x/y):
    slow scale ~ (T-t)^cs   (cs > 0),   fast scale ~ (T-t)^(cs+Dl)  (Dl >= 0)
    w  = (T-t)^(-wexp) W(xi_slow, xi_fast, s),   s = -ln(T-t)
    th = (T-t)^(gexp)  H(xi_slow, xi_fast, s)

Assumptions (each a flag, not a footnote):
  A1  strict power laws -- log corrections / drifting exponents evade all of this
  A2  theta actively transported (its two leading terms genuinely balance)
  A3  Poisson inverted along the fast direction at leading order;
      corrections O((T-t)^(2 Dl)); the wall corner may break this locally
  A4  amplitudes tied to the sup-normalization used by features.py
"""

import sympy as sp

cs, Dl, wexp, gexp = sp.symbols('c_s Delta w g', real=True)
cf = cs + Dl

OUT = []
def say(t=''):
    OUT.append(t)

# ---- exponent table (powers of (T-t)) -------------------------------------
P    = 2*cf - wexp            # psi, from psi_ff ~ w at leading order (A3)
u_x  = P - cf                 # u = psi_y : fast-derivative, advects along SLOW
v_y  = P - cs                 # v = -psi_x: slow-derivative, advects along FAST
adv_w   = sp.simplify(u_x + (-wexp - cs))    # u w_x   (cross pairing)
adv_w2  = sp.simplify(v_y + (-wexp - cf))    # v w_y
dt_w    = -wexp - 1                          # includes the drift terms
adv_th  = sp.simplify(u_x + (gexp - cs))
adv_th2 = sp.simplify(v_y + (gexp - cf))
dt_th   = gexp - 1

# incompressibility crosses the pairings: both advection components share
# one exponent, in each equation.  Verify.
assert sp.simplify(adv_w - adv_w2) == 0
assert sp.simplify(adv_th - adv_th2) == 0

say('K1: strict two-scale power-law collapse vs the full Boussinesq system')
say('=' * 72)
say()
say('Exponent table (powers of (T-t); smaller = more dominant as t->T):')
say(f'  omega_t (incl. drift)   : {dt_w}')
say(f'  u.grad omega            : {adv_w}')
say(f'  theta_x  (buoy, slow)   : {gexp - cs}')
say(f'  theta_x  (buoy, fast)   : {gexp - cf}')
say(f'  theta_t                 : {dt_th}')
say(f'  u.grad theta            : {adv_th}')
say()

# ---- R1: the theta equation alone fixes w ---------------------------------
w_forced = sp.solve(sp.Eq(dt_th, adv_th), wexp)[0]
assert sp.simplify(w_forced - (1 + Dl)) == 0
say('R1  THE THETA-TRANSPORT RELATION.  theta has exactly two terms; if it')
say('    is actively transported (A2) they must balance, which forces')
say('        w = 1 + Delta,   Delta = c_fast - c_slow >= 0.')
say('    The anisotropy gap is paid for by a FASTER vorticity amplitude.')
say('    Delta = 0 recovers w = 1 (Chen-Hou).  [check: PASS]')
say()

# ---- R2: the omega equation then balances automatically -------------------
gap = sp.simplify((dt_w - adv_w).subs(wexp, w_forced))
assert gap == 0
say('R2  With w = 1 + Delta, omega_t and u.grad omega balance IDENTICALLY')
say('    (transport structure; not an extra condition).  [check: PASS]')
say()

# ---- R3: buoyancy placement fixes g ---------------------------------------
g_slow = sp.solve(sp.Eq(gexp - cs, dt_w.subs(wexp, w_forced)), gexp)[0]
g_fast = sp.solve(sp.Eq(gexp - cf, dt_w.subs(wexp, w_forced)), gexp)[0]
say('R3  Buoyancy in the leading balance fixes g:')
say(f'      gradient along slow direction: g = {sp.simplify(g_slow)}')
say(f'      gradient along fast direction: g = {sp.simplify(g_fast)}')
say('    (Delta = 0: both give g = c_s - 2, the isotropic value.  Buoyancy')
say('    subdominant is also admissible: an Euler-like transport collapse.)')
say()

# ---- R4: BKM is automatic -------------------------------------------------
say('R4  w = 1 + Delta >= 1 always: a strict two-scale collapse is')
say('    automatically BKM-supercritical.  The ansatz cannot manufacture a')
say('    sub-BKM fake blowup.  (Small, but it closes a checking chore.)')
say()

# ---- R5: confrontation with the banked numbers ----------------------------
# Aspect drift as measured by features.py:  |D| = dln(l_f/l_s)/dln|w| in
# magnitude = Delta / w.  With R1:  |D| = (w-1)/w  =>  w = 1/(1-|D|),
# admissible ONLY for |D| < 1.
say('R5  CONFRONTATION.  |D| := |dln(aspect)/dln|omega||  =  Delta/w.')
say('    R1 then demands   w = 1/(1 - |D|),  admissible only if |D| < 1.')
say()
say('      measured |D|          predicted w        verdict')
for Dv, label in [(0.0, 'corner profile (exact)'),
                  (0.16, 'engineered corner ICs'),
                  (1.01, 'generic seeded ICs')]:
    if Dv < 1:
        wpred = 1.0 / (1.0 - Dv)
        say(f'      {Dv:5.2f}  {label:24s}  w = {wpred:.4f}')
    else:
        say(f'      {Dv:5.2f}  {label:24s}  NO ADMISSIBLE w  (|D| >= 1)')
say()
say('    Fallout, stated at UNGRADED strength:')
say('    (a) GENERIC ICs: |D| ~ 1.01 sits at/above the admissibility')
say('        boundary.  If the per-run bar keeps |D| >= 1, generic data')
say('        CANNOT be a strict two-scale power-law collapse at all --')
say('        independent of any w measurement.  The two-exponent "settled')
say('        wandering" picture would be refuted by the equation itself;')
say('        what remains is asymptotically-isotropic-with-drift (log/')
say('        prefactor wandering) or genuinely non-power-law.')
say('    (b) ENGINEERED ICs: |D| = 0.16 with the banked w = 1 violates R1')
say('        by 0.16.  Either the 0.16 is transient drift on a truly')
say('        isotropic collapse, or w = 1.19 on those runs.  KILL TEST:')
say('        refit the existing T* logs with forced exponent -1.19 vs -1.00')
say('        and compare residuals (free-residual style, no new runs).')
say('    (c) M3 hypothesis space SHRINKS pending grading: "settles to a')
say('        two-exponent profile" is not on the menu; settling means the')
say('        isotropic corner class, or nothing.')
say()
say('Assumptions that bound all of the above: A1-A4 in the header.')
say('A1 is the load-bearing one -- the s-march (M1) is exactly the')
say('instrument that can see log corrections.  The tracks converge.')

text = '\n'.join(OUT) + '\n'
with open('CURIOUS_TWOSCALE.out', 'w') as f:
    f.write(text)
print(text)
