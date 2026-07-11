#!/usr/bin/env python3
"""
sph_hram.py — 2D SPH Hydraulic Ram (HRAM) simulation
Reference: Varas et al. (2009) Int J Impact Eng 36:678-686

Setup (default: Varas benchmark):
  Vessel 200×200 mm, fluid-filled, rigid walls.
  Rigid cylindrical projectile φ=25 mm, m=50 g, v0=900 m/s.
  Entry through pre-drilled hole (diameter = projectile diameter) in left wall.

Physics:
  - Tait EOS:  p = B[(ρ/ρ₀)^γ − 1],  B = ρ₀c₀²/γ,  γ=7
  - Cavitation cutoff:  p ← max(p, p_cav)   [tension not physical for liquids]
  - Physical viscosity (Shao 2012 Eq.6) + Monaghan 1992 AV  (α=1.0 for shocks)
  - Projectile BC: smooth penalty repulsion force (D=5c₀²/h) + soft push-out
  - Rigid walls: damped reflection on all 4 sides; left wall has entry hole
  - XSPH position correction (ε=0.5) + Leapfrog (Verlet) integration
  - Cell-linked list → O(N·k) neighbour search

Outputs (tagged by fluid/v0/dx):
  hram_pressure_probes_<tag>.dat  — t, p_entry, p_center, p_exit  [Pa]
  hram_wall_force_<tag>.dat       — t, F_exit[N], v_proj[m/s], x_proj[m]
  hram_<tag>.png                  — 4-panel time-history plot

Objectives:
  (c) python sph_hram.py --fluid water          ← pressure field + wall force
  (d) python sph_hram.py --fluid air            ← compare air vs water
  (e) python sph_hram.py --fluid water --plot   ← full validation run

Usage:
  python sph_hram.py [--dx 0.002] [--tend 5e-4] [--v0 900]
                     [--fluid water|air] [--alpha 1.0]
                     [--plot] [--gpu]
"""

import argparse, os, time as clk, math
import numpy as np
from numba import njit, prange

# ─── Vessel geometry ──────────────────────────────────────────────────────────
Lv  = 0.200    # length [m]
Hv  = 0.200    # height [m]

# ─── Fluid properties — module-level defaults (water) ─────────────────────────

RHO0  = 1000.0    # reference density [kg/m³]
C0    = 1480.0    # speed of sound [m/s]  (real water value)
MU    = 1.0e-3    # dynamic viscosity [Pa·s]
GAMMA = 7.0       # Tait polytropic index
P_CAV = -1.0e5    # cavitation tension cutoff [Pa]

# ─── Projectile (Varas 2009 benchmark) ────────────────────────────────────────
R_PROJ   = 0.0125    # radius [m]  (φ = 25 mm)
RHO_PROJ = 7800.0    # density [kg/m³]  (steel)
# 2D mass per unit depth:  M = ρ_proj · π · R²  (NOT the 3D total mass)
# For steel, φ=25mm:  M_2D = 7800 · π · 0.0125² = 3.827 kg/m
M_PROJ   = RHO_PROJ * math.pi * R_PROJ**2   # [kg/m depth]
V0_DEF   = 900.0     # default initial velocity [m/s]

# ─── SPH constants ────────────────────────────────────────────────────────────
EPS      = 0.5     # XSPH ε
E_BC     = 0.3     # wall restitution
ALPHA_AV = 1.0     # Monaghan AV coefficient (shock problems: 0.5–2.0)
pi       = math.pi

# ─── Pressure probe locations  (y = Hv/2 for all three) ──────────────────────
P1_X = 0.020       # near entry wall
P2_X = Lv / 2.0    # vessel centre
P3_X = Lv - 0.020  # near exit wall


# ══════════════════════════════════════════════════════════════════════════════
#  CUBIC SPLINE KERNEL  (identical to sph_sloshing.py)
# ══════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def kernel(r, dxij, dyij, h):
    sig = 10.0 / (7.0 * pi * h * h)
    q   = r / h
    if q <= 1.0:
        W  = sig * (1.0 - 1.5*q*q + 0.75*q*q*q)
        gq = sig * (-3.0*q + 2.25*q*q) / h
    elif q <= 2.0:
        W  = sig * 0.25 * (2.0 - q)**3
        gq = -sig * 0.75 * (2.0 - q)**2 / h
    else:
        return 0.0, 0.0, 0.0
    if r > 1.0e-12:
        f = gq / r
        return W, f * dxij, f * dyij
    return W, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  CELL-LINKED LIST  (identical to sph_sloshing.py)
# ══════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def build_cells(x, y, N, dc, nxc, nyc):
    head = -np.ones((nxc, nyc), dtype=np.int32)
    nxt  = -np.ones(N,          dtype=np.int32)
    for i in range(N):
        ix = min(max(int(x[i] / dc), 0), nxc - 1)
        iy = min(max(int(y[i] / dc), 0), nyc - 1)
        nxt[i]       = head[ix, iy]
        head[ix, iy] = i
    return head, nxt


# ══════════════════════════════════════════════════════════════════════════════
#  DENSITY SUMMATION
# ══════════════════════════════════════════════════════════════════════════════
@njit(parallel=True, cache=True)
def density_sum(x, y, m, N, h, dc, head, nxt, nxc, nyc, rho0_v):
    rho = np.empty(N, np.float64)
    for i in prange(N):
        acc = 0.0
        ix  = min(max(int(x[i] / dc), 0), nxc - 1)
        iy  = min(max(int(y[i] / dc), 0), nyc - 1)
        for icx in range(max(0, ix - 2), min(nxc, ix + 3)):
            for icy in range(max(0, iy - 2), min(nyc, iy + 3)):
                j = head[icx, icy]
                while j >= 0:
                    dxij = x[i] - x[j]
                    dyij = y[i] - y[j]
                    r    = np.sqrt(dxij*dxij + dyij*dyij)
                    if r < 2.0 * h:
                        W, _, _ = kernel(r, dxij, dyij, h)
                        acc += m * W
                    j = nxt[j]
        rho[i] = max(acc, 0.5 * rho0_v)
    return rho


# ══════════════════════════════════════════════════════════════════════════════
#  TAIT EOS  with cavitation tension cutoff
#    p = B[(ρ/ρ₀)^γ − 1],   B = ρ₀c₀²/γ
# ══════════════════════════════════════════════════════════════════════════════
@njit(parallel=True, cache=True)
def pressure_tait(rho, N, rho0_v, c0_v, p_cav_v):
    B = rho0_v * c0_v * c0_v / GAMMA
    p = np.empty(N, np.float64)
    for i in prange(N):
        v    = B * ((rho[i] / rho0_v)**GAMMA - 1.0)
        p[i] = v if v > p_cav_v else p_cav_v
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  ACCELERATIONS  (Shao 2012 Eq.6 viscosity + Monaghan 1992 AV)
#  No tank forcing — gravity only.
# ══════════════════════════════════════════════════════════════════════════════
@njit(parallel=True, cache=True)
def accel_hram(x, y, vx, vy, rho, p, m, N, h, dc, head, nxt, nxc, nyc,
               mu_v, alpha_v, c0_v):
    ax = np.zeros(N, np.float64)
    ay = np.full(N, -9.81, np.float64)
    for i in prange(N):
        ix = min(max(int(x[i] / dc), 0), nxc - 1)
        iy = min(max(int(y[i] / dc), 0), nyc - 1)
        for icx in range(max(0, ix - 2), min(nxc, ix + 3)):
            for icy in range(max(0, iy - 2), min(nyc, iy + 3)):
                j = head[icx, icy]
                while j >= 0:
                    if j != i:
                        dxij = x[i] - x[j]
                        dyij = y[i] - y[j]
                        r    = np.sqrt(dxij*dxij + dyij*dyij)
                        if 0.0 < r < 2.0 * h:
                            _, dWx, dWy = kernel(r, dxij, dyij, h)
                            # symmetric pressure gradient
                            pterm = m * (p[i] / rho[i]**2 + p[j] / rho[j]**2)
                            ax[i] -= pterm * dWx
                            ay[i] -= pterm * dWy
                            # Shao 2012 Eq.6 physical viscosity
                            xdW = dxij * dWx + dyij * dWy
                            vc  = (4.0 * m * (mu_v + mu_v) * xdW /
                                   ((rho[i] + rho[j])**2 * (r*r + 0.01*h*h)))
                            ax[i] += vc * (vx[i] - vx[j])
                            ay[i] += vc * (vy[i] - vy[j])
                            # Monaghan 1992 AV — approaching pairs only (shock capturing)
                            if alpha_v > 0.0:
                                vdotr = ((vx[i] - vx[j]) * dxij +
                                         (vy[i] - vy[j]) * dyij)
                                if vdotr < 0.0:
                                    mu_ij = h * vdotr / (r*r + 0.01*h*h)
                                    rho_b  = 0.5 * (rho[i] + rho[j])
                                    PI_ij  = -alpha_v * c0_v * mu_ij / rho_b
                                    ax[i] -= m * PI_ij * dWx
                                    ay[i] -= m * PI_ij * dWy
                    j = nxt[j]
    return ax, ay


# ══════════════════════════════════════════════════════════════════════════════
#  XSPH POSITION CORRECTION  (identical to sph_sloshing.py)
# ══════════════════════════════════════════════════════════════════════════════
@njit(parallel=True, cache=True)
def xsph_drift(x, y, vx, vy, rho, m, N, h, dc, head, nxt, nxc, nyc, dt):
    cx = np.zeros(N, np.float64)
    cy = np.zeros(N, np.float64)
    for i in prange(N):
        ix = min(max(int(x[i] / dc), 0), nxc - 1)
        iy = min(max(int(y[i] / dc), 0), nyc - 1)
        for icx in range(max(0, ix - 2), min(nxc, ix + 3)):
            for icy in range(max(0, iy - 2), min(nyc, iy + 3)):
                j = head[icx, icy]
                while j >= 0:
                    if j != i:
                        dxij = x[i] - x[j]
                        dyij = y[i] - y[j]
                        r    = np.sqrt(dxij*dxij + dyij*dyij)
                        if r < 2.0 * h:
                            W, _, _ = kernel(r, dxij, dyij, h)
                            rav = 0.5 * (rho[i] + rho[j])
                            cx[i] += (m / rav) * (vx[j] - vx[i]) * W
                            cy[i] += (m / rav) * (vy[j] - vy[i]) * W
                    j = nxt[j]
    xn = x + (vx + EPS * cx) * dt
    yn = y + (vy + EPS * cy) * dt
    return xn, yn


# ══════════════════════════════════════════════════════════════════════════════
#  WALL BC  — 4 rigid walls; left wall has circular entry hole
# ══════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def wall_bc(x, y, vx, vy, N, r_hole, dx_v):
    """
    Damp-reflect on all 4 walls.
    Left AND right walls have a circular hole of radius r_hole + dx at mid-height.
    This models the pre-drilled entry (left) and projectile-punched exit (right),
    allowing pressure relief once the wave reaches the exit wall — consistent
    with the Varas 2009 experimental setup.
    """
    for i in range(N):
        in_hole = abs(y[i] - Hv * 0.5) < r_hole + dx_v
        if x[i] < 0.0 and not in_hole:
            x[i]  = 0.0;  vx[i] =  E_BC * abs(vx[i])
        if x[i] > Lv and not in_hole:
            x[i]  = Lv;   vx[i] = -E_BC * abs(vx[i])
        if y[i] < 0.0:
            y[i]  = 0.0;  vy[i] =  E_BC * abs(vy[i])
        if y[i] > Hv:
            y[i]  = Hv;   vy[i] = -E_BC * abs(vy[i])
    return x, y, vx, vy


# ══════════════════════════════════════════════════════════════════════════════
#  PROJECTILE REPULSION  — smooth penalty force + soft push-out
#
#  Penalty force model (Monaghan & Kajtar 2009 inspired):
#    For fluid particles within distance (R_PROJ + h) of projectile centre:
#      d = dist_to_centre − R_PROJ  (signed: negative = inside projectile)
#      if d < h:
#        a_rep = D · (1 − max(d,0)/h)²  · n_hat   [m/s² on fluid particle]
#    D = 5·c₀²/h  — calibrated to decelerate a c₀-speed particle over one h.
#
#  Reaction force on projectile (Newton 3rd law):
#    F_proj = −Σ m_part · a_rep · n_hat   [N/m depth]
#  Projectile velocity updated in main loop via:
#    vxp += F_proj_x / M_PROJ · dt
#
#  Soft push-out: particles that penetrate inside are moved to the surface
#  (position correction only; the force already handles the velocity).
# ══════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def proj_force(x, y, m_part, N, xpc, ypc, h, c0_v):
    """
    Returns per-particle repulsion acceleration (ax_p, ay_p) and
    total reaction force on projectile (Fx, Fy) [N/m depth].
    Also corrects positions of any particle inside the projectile.
    """
    D    = 5.0 * c0_v * c0_v / h    # repulsion strength [m/s²]
    ax_p = np.zeros(N, np.float64)
    ay_p = np.zeros(N, np.float64)
    Fx   = 0.0
    Fy   = 0.0
    for i in range(N):
        dxp  = x[i] - xpc
        dyp  = y[i] - ypc
        dist = np.sqrt(dxp * dxp + dyp * dyp)
        if dist < R_PROJ + h:
            if dist < 1.0e-12:
                nx_ = 1.0;  ny_ = 0.0
            else:
                nx_ = dxp / dist;  ny_ = dyp / dist
            d_surf = dist - R_PROJ          # signed distance to surface
            # soft push-out: restore position if inside
            if d_surf < 0.0:
                x[i] = xpc + R_PROJ * nx_ * 1.001
                y[i] = ypc + R_PROJ * ny_ * 1.001
                d_surf = 0.0
            # smooth quadratic repulsion for d_surf < h
            if d_surf < h:
                s      = d_surf / h          # 0 at surface, 1 at cutoff
                a_mag  = D * (1.0 - s) * (1.0 - s)
                ax_p[i] += a_mag * nx_
                ay_p[i] += a_mag * ny_
                # Newton 3rd law: reaction on projectile
                Fx      -= m_part * a_mag * nx_
                Fy      -= m_part * a_mag * ny_
    return x, y, ax_p, ay_p, Fx, Fy


# ══════════════════════════════════════════════════════════════════════════════
#  BAFFLE FORCES  (objective e — stiffener configurations)
#
#  Both functions use the same quadratic penalty as proj_force: particles
#  within h of the baffle surface are repelled; those inside are pushed out.
#  Baffles are rigid (zero mass) — no Newton-3rd reaction term needed.
# ══════════════════════════════════════════════════════════════════════════════
@njit(parallel=True, cache=True)
def baffle_force_transverse(x, y, N, h, dx_v, c0_v):
    """Rigid transverse frame at x = Lv/2.
    Full height except hole of radius (R_PROJ + h) at y = Hv/2 for
    projectile and leading jet to pass through."""
    D      = 5.0 * c0_v * c0_v / h
    ax_b   = np.zeros(N, np.float64)
    ay_b   = np.zeros(N, np.float64)
    x_b    = Lv * 0.5
    r_hole = R_PROJ + h
    for i in prange(N):
        if abs(y[i] - Hv * 0.5) < r_hole:
            continue                           # inside hole — unobstructed
        d    = abs(x[i] - x_b)
        if d < h:
            sign = 1.0 if x[i] >= x_b else -1.0
            s    = d / h
            ax_b[i] += D * (1.0 - s) * (1.0 - s) * sign
            if d < dx_v:
                x[i] = x_b + sign * dx_v * 1.01
    return x, y, ax_b, ay_b


@njit(parallel=True, cache=True)
def baffle_force_longitudinal(x, y, N, h, dx_v, c0_v):
    """Rigid stringer pair at y = Hv/4 and y = 3·Hv/4, full vessel length."""
    D    = 5.0 * c0_v * c0_v / h
    ax_b = np.zeros(N, np.float64)
    ay_b = np.zeros(N, np.float64)
    yb1  = Hv * 0.25
    yb2  = Hv * 0.75
    for i in prange(N):
        d1 = abs(y[i] - yb1)
        if d1 < h:
            s1 = 1.0 if y[i] >= yb1 else -1.0
            ay_b[i] += D * (1.0 - d1 / h) * (1.0 - d1 / h) * s1
            if d1 < dx_v:
                y[i] = yb1 + s1 * dx_v * 1.01
        d2 = abs(y[i] - yb2)
        if d2 < h:
            s2 = 1.0 if y[i] >= yb2 else -1.0
            ay_b[i] += D * (1.0 - d2 / h) * (1.0 - d2 / h) * s2
            if d2 < dx_v:
                y[i] = yb2 + s2 * dx_v * 1.01
    return x, y, ax_b, ay_b


# ══════════════════════════════════════════════════════════════════════════════
#  PRESSURE PROBES  — SPH kernel-weighted average at 3 fixed points
#    P1: x=0.020m  (near entry wall)
#    P2: x=Lv/2    (vessel centre)
#    P3: x=Lv-0.020m (near exit wall)
#  All probes at y = Hv/2.  Averaged over particles within 2h.
# ══════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def probe_pressures(x, y, p, N, h):
    yp  = Hv * 0.5
    r2h = 2.0 * h
    s   = np.zeros(3, np.float64)
    c   = np.zeros(3, np.float64)
    px  = np.array([P1_X, P2_X, P3_X])
    for i in range(N):
        if abs(y[i] - yp) < r2h:
            for k in range(3):
                if abs(x[i] - px[k]) < r2h:
                    s[k] += p[i]
                    c[k] += 1.0
    result = np.zeros(3, np.float64)
    for k in range(3):
        result[k] = s[k] / c[k] if c[k] > 0 else 0.0
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  EXIT WALL FORCE  — average pressure × wall height (2D, per unit depth)
#  Particles within 2h of the exit wall (x = Lv).
# ══════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def wall_force_exit(x, p, N, h):
    s = 0.0;  c = 0
    for i in range(N):
        if x[i] > Lv - 2.0 * h:
            s += p[i];  c += 1
    return (s / c) * Hv if c > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  INITIALISATION  — uniform density, fluid at rest, full vessel
# ══════════════════════════════════════════════════════════════════════════════
def init_fluid(dx, rho0_v):
    nx  = int(Lv / dx)
    ny  = int(Hv / dx)
    N   = nx * ny
    idx = np.arange(N)
    x   = ((idx % nx) + 0.5) * dx
    y   = ((idx // nx) + 0.5) * dx
    vx  = np.zeros(N, np.float64)
    vy  = np.zeros(N, np.float64)
    rho = np.full(N, rho0_v, np.float64)
    return (x.astype(np.float64), y.astype(np.float64),
            vx, vy, rho, N)


# ══════════════════════════════════════════════════════════════════════════════
#  CPU SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
def run(dx=0.002, tend=5e-4, make_plot=False, v0=V0_DEF,
        fluid='water', alpha_av=ALPHA_AV, stiffener='none'):

    if fluid == 'air':
        rho0_v = 1.2;    c0_v = 340.0;   mu_v = 1.8e-5
    else:
        rho0_v = RHO0;   c0_v = C0;      mu_v = MU

    h    = 1.3 * dx
    dt   = 0.35 * h / (c0_v + v0)   # CFL: includes projectile speed
    m    = rho0_v * dx * dx
    nout = max(1, int(1.0e-5 / dt))  # output every ~10 µs

    nxc = int(Lv / h) + 3
    nyc = int(Hv / h) + 3
    dc  = h

    x, y, vx, vy, rho, N = init_fluid(dx, rho0_v)

    # Projectile starts just left of vessel, centred vertically
    xpc = -(R_PROJ + dx);  ypc = Hv / 2.0
    vxp = v0;              vyp = 0.0

    B = rho0_v * c0_v * c0_v / GAMMA
    print(f"\n{'─'*65}")
    print(f"  2D SPH-HRAM  (Varas 2009,  {fluid.upper()},  CPU,  stiffener={stiffener})")
    print(f"  N={N}  dx={dx*1e3:.1f}mm  h={h*1e3:.2f}mm  dt={dt:.2e}s")
    print(f"  v₀={v0} m/s  tend={tend*1e3:.2f} ms  steps≈{int(tend/dt):,}")
    print(f"  c₀={c0_v} m/s  B={B/1e6:.2f} MPa  α_AV={alpha_av}")
    print(f"{'─'*65}")

    print("  Compiling JIT kernels ...", end="", flush=True)
    head, nxt = build_cells(x, y, N, dc, nxc, nyc)
    rho  = density_sum(x, y, m, N, h, dc, head, nxt, nxc, nyc, rho0_v)
    p    = pressure_tait(rho, N, rho0_v, c0_v, P_CAV)
    x, y, ax_p, ay_p, _, _ = proj_force(x, y, m, N, xpc, ypc, h, c0_v)
    if stiffener == 'transverse':
        x, y, ax_b, ay_b = baffle_force_transverse(x, y, N, h, dx, c0_v)
    elif stiffener == 'longitudinal':
        x, y, ax_b, ay_b = baffle_force_longitudinal(x, y, N, h, dx, c0_v)
    else:
        ax_b = ay_b = np.zeros(N, np.float64)
    ax_ff, ay_ff = accel_hram(x, y, vx, vy, rho, p, m, N, h, dc, head, nxt,
                               nxc, nyc, mu_v, alpha_av, c0_v)
    ax = ax_ff + ax_p + ax_b
    ay = ay_ff + ay_p + ay_b
    print(" done.")

    probe_list = []   # (t, p1, p2, p3)
    force_list = []   # (t, F_exit, vxp, xpc)
    t, step = 0.0, 0
    t0 = clk.perf_counter()

    while t <= tend:
        # ── Leapfrog half-kick ──────────────────────────────────────────
        vxh = vx + 0.5 * ax * dt
        vyh = vy + 0.5 * ay * dt

        # ── Advance projectile to t+dt ─────────────────────────────────
        xpc += vxp * dt
        ypc += vyp * dt

        # ── XSPH drift ─────────────────────────────────────────────────
        x, y = xsph_drift(x, y, vxh, vyh, rho, m, N, h, dc,
                           head, nxt, nxc, nyc, dt)

        # ── Projectile penalty force at t+dt positions ──────────────────
        x, y, ax_p, ay_p, Fx, Fy = proj_force(x, y, m, N, xpc, ypc, h, c0_v)

        # ── Baffle force (stiffener BC) ─────────────────────────────────
        if stiffener == 'transverse':
            x, y, ax_b, ay_b = baffle_force_transverse(x, y, N, h, dx, c0_v)
        elif stiffener == 'longitudinal':
            x, y, ax_b, ay_b = baffle_force_longitudinal(x, y, N, h, dx, c0_v)
        else:
            ax_b = ay_b = np.zeros(N, np.float64)

        # ── Wall BC (4 sides, entry and exit holes) ─────────────────────
        x, y, vxh, vyh = wall_bc(x, y, vxh, vyh, N, R_PROJ, dx)

        # ── Rebuild cell list ───────────────────────────────────────────
        head, nxt = build_cells(x, y, N, dc, nxc, nyc)

        # ── Density, pressure, fluid–fluid acceleration at t+dt ─────────
        rho    = density_sum(x, y, m, N, h, dc, head, nxt, nxc, nyc, rho0_v)
        p      = pressure_tait(rho, N, rho0_v, c0_v, P_CAV)
        ax_ff, ay_ff = accel_hram(x, y, vxh, vyh, rho, p, m, N, h, dc, head, nxt,
                                   nxc, nyc, mu_v, alpha_av, c0_v)
        ax = ax_ff + ax_p + ax_b
        ay = ay_ff + ay_p + ay_b

        # ── Leapfrog full kick ──────────────────────────────────────────
        vx = vxh + 0.5 * ax * dt
        vy = vyh + 0.5 * ay * dt

        # ── Projectile velocity update (Newton 3rd law) ─────────────────
        vxp += Fx / M_PROJ * dt
        vyp += Fy / M_PROJ * dt - 9.81 * dt

        # ── Output ──────────────────────────────────────────────────────
        if step % nout == 0:
            probes = probe_pressures(x, y, p, N, h)
            F_exit = wall_force_exit(x, p, N, h)
            probe_list.append((t, probes[0], probes[1], probes[2]))
            force_list.append((t, F_exit, vxp, xpc))
            if step % (nout * 20) == 0 and t > 0.0:
                el  = clk.perf_counter() - t0
                eta = el / t * (tend - t) if t > 0 else 0.0
                print(f"  t={t*1e6:6.1f}µs  xp={xpc*1e3:6.1f}mm"
                      f"  vp={vxp:7.1f}m/s  p_ctr={probes[1]/1e6:6.3f}MPa"
                      f"  F_exit={F_exit:.2f}N"
                      f"  [{el:.1f}s  ETA {eta:.1f}s]")

        t    += dt
        step += 1

    # ── Save data ────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    tag     = f"{fluid}_v{int(v0)}_dx{dx*1e3:.1f}mm_a{alpha_av}_s{stiffener}"
    pr_arr  = np.array(probe_list)
    fm_arr  = np.array(force_list)
    np.savetxt(os.path.join(out_dir, f'hram_pressure_probes_{tag}.dat'), pr_arr,
               fmt='%14.6e',
               header='t[s]  p_entry[Pa]  p_center[Pa]  p_exit[Pa]')
    np.savetxt(os.path.join(out_dir, f'hram_wall_force_{tag}.dat'), fm_arr,
               fmt='%14.6e',
               header='t[s]  F_exit[N]  v_proj[m/s]  x_proj[m]')

    elapsed = clk.perf_counter() - t0
    print(f"\n  Done. {step:,} steps in {elapsed:.1f}s")
    print(f"  hram_pressure_probes_{tag}.dat")
    print(f"  hram_wall_force_{tag}.dat")

    if make_plot:
        plot_results(pr_arr, fm_arr, dx, fluid, v0, alpha_av, stiffener)

    return pr_arr, fm_arr


# ══════════════════════════════════════════════════════════════════════════════
#  GPU KERNELS  (Numba CUDA — cell-list versions)
# ══════════════════════════════════════════════════════════════════════════════
_CUDA_OK = False
try:
    from numba import cuda as _nc
    _nc.gpus
    from numba import cuda
    _CUDA_OK = True
except Exception:
    pass

if _CUDA_OK:

    @cuda.jit(device=True)
    def _kgpu(r, dxij, dyij, h):
        sig = 10.0 / (7.0 * math.pi * h * h)
        q   = r / h
        if q <= 1.0:
            W  = sig * (1.0 - 1.5*q*q + 0.75*q*q*q)
            gq = sig * (-3.0*q + 2.25*q*q) / h
        elif q <= 2.0:
            W  = sig * 0.25 * (2.0 - q)**3
            gq = -sig * 0.75 * (2.0 - q)**2 / h
        else:
            return 0.0, 0.0, 0.0
        if r > 1.0e-12:
            f = gq / r
            return W, f * dxij, f * dyij
        return W, 0.0, 0.0

    @cuda.jit
    def _gpu_density(x, y, m, N, h, dc, nxc, nyc, head, nxt, rho0_v, rho_out):
        i = cuda.grid(1)
        if i >= N:
            return
        acc = 0.0;  h2 = 2.0 * h
        xi  = x[i]; yi = y[i]
        ix  = min(max(int(xi / dc), 0), nxc - 1)
        iy  = min(max(int(yi / dc), 0), nyc - 1)
        for icx in range(max(0, ix - 2), min(nxc, ix + 3)):
            for icy in range(max(0, iy - 2), min(nyc, iy + 3)):
                j = head[icx, icy]
                while j >= 0:
                    dxij = xi - x[j];  dyij = yi - y[j]
                    r = math.sqrt(dxij*dxij + dyij*dyij)
                    if r < h2:
                        W, _, _ = _kgpu(r, dxij, dyij, h)
                        acc += m * W
                    j = nxt[j]
        rho_out[i] = acc if acc > 0.5 * rho0_v else 0.5 * rho0_v

    @cuda.jit
    def _gpu_pressure(rho, N, rho0_v, c0_v, p_cav_v, p_out):
        i = cuda.grid(1)
        if i >= N:
            return
        B    = rho0_v * c0_v * c0_v / GAMMA
        v    = B * ((rho[i] / rho0_v)**GAMMA - 1.0)
        p_out[i] = v if v > p_cav_v else p_cav_v

    @cuda.jit
    def _gpu_accel(x, y, vx, vy, rho, p, m, N, h, dc, nxc, nyc, head, nxt,
                   mu_v, alpha_v, c0_v, ax_out, ay_out):
        i = cuda.grid(1)
        if i >= N:
            return
        axi = 0.0;  ayi = -9.81
        xi  = x[i]; yi  = y[i]
        vxi = vx[i]; vyi = vy[i]
        ri  = rho[i]; pi_ = p[i]
        h2  = 2.0 * h
        ix  = min(max(int(xi / dc), 0), nxc - 1)
        iy  = min(max(int(yi / dc), 0), nyc - 1)
        for icx in range(max(0, ix - 2), min(nxc, ix + 3)):
            for icy in range(max(0, iy - 2), min(nyc, iy + 3)):
                j = head[icx, icy]
                while j >= 0:
                    if j != i:
                        dxij = xi - x[j];  dyij = yi - y[j]
                        r = math.sqrt(dxij*dxij + dyij*dyij)
                        if 0.0 < r < h2:
                            _, dWx, dWy = _kgpu(r, dxij, dyij, h)
                            pterm = m * (pi_ / ri**2 + p[j] / rho[j]**2)
                            axi  -= pterm * dWx
                            ayi  -= pterm * dWy
                            xdW   = dxij * dWx + dyij * dWy
                            vc    = (4.0 * m * (mu_v + mu_v) * xdW /
                                     ((ri + rho[j])**2 * (r*r + 0.01*h*h)))
                            axi  += vc * (vxi - vx[j])
                            ayi  += vc * (vyi - vy[j])
                            if alpha_v > 0.0:
                                vdotr = (vxi - vx[j])*dxij + (vyi - vy[j])*dyij
                                if vdotr < 0.0:
                                    mu_ij = h * vdotr / (r*r + 0.01*h*h)
                                    rho_b = 0.5 * (ri + rho[j])
                                    PI_ij = -alpha_v * c0_v * mu_ij / rho_b
                                    axi  -= m * PI_ij * dWx
                                    ayi  -= m * PI_ij * dWy
                    j = nxt[j]
        ax_out[i] = axi
        ay_out[i] = ayi

    @cuda.jit
    def _gpu_xsph(x, y, vx, vy, rho, m, N, h, dc, nxc, nyc, head, nxt,
                  dt_v, xn, yn):
        i = cuda.grid(1)
        if i >= N:
            return
        cxi = 0.0;  cyi = 0.0
        xi  = x[i]; yi  = y[i]
        vxi = vx[i]; vyi = vy[i]
        ri  = rho[i];  h2 = 2.0 * h
        ix  = min(max(int(xi / dc), 0), nxc - 1)
        iy  = min(max(int(yi / dc), 0), nyc - 1)
        for icx in range(max(0, ix - 2), min(nxc, ix + 3)):
            for icy in range(max(0, iy - 2), min(nyc, iy + 3)):
                j = head[icx, icy]
                while j >= 0:
                    if j != i:
                        dxij = xi - x[j];  dyij = yi - y[j]
                        r = math.sqrt(dxij*dxij + dyij*dyij)
                        if r < h2:
                            W, _, _ = _kgpu(r, dxij, dyij, h)
                            rav  = 0.5 * (ri + rho[j])
                            cxi += (m / rav) * (vx[j] - vxi) * W
                            cyi += (m / rav) * (vy[j] - vyi) * W
                    j = nxt[j]
        xn[i] = xi + (vxi + EPS * cxi) * dt_v
        yn[i] = yi + (vyi + EPS * cyi) * dt_v

    @cuda.jit
    def _gpu_halfkick(vx, vy, ax, ay, dt_v, N, vxh, vyh):
        i = cuda.grid(1)
        if i >= N:
            return
        vxh[i] = vx[i] + 0.5 * ax[i] * dt_v
        vyh[i] = vy[i] + 0.5 * ay[i] * dt_v

    @cuda.jit
    def _gpu_fullkick(vxh, vyh, ax, ay, dt_v, N, vx_out, vy_out):
        i = cuda.grid(1)
        if i >= N:
            return
        vx_out[i] = vxh[i] + 0.5 * ax[i] * dt_v
        vy_out[i] = vyh[i] + 0.5 * ay[i] * dt_v


# ══════════════════════════════════════════════════════════════════════════════
#  GPU SIMULATION
#
#  Strategy: heavy O(N·k) kernels (density, accel, xsph) run on GPU.
#  Projectile BC and wall BC run on CPU (O(N) simple loops, negligible cost).
#  Cell list rebuilt on CPU each step; head/nxt (~N×8 bytes) copied to device.
# ══════════════════════════════════════════════════════════════════════════════
def run_gpu(dx=0.002, tend=5e-4, make_plot=False, v0=V0_DEF,
            fluid='water', alpha_av=ALPHA_AV, stiffener='none'):
    if not _CUDA_OK:
        raise RuntimeError("CUDA GPU not available — run without --gpu.")
    from numba import cuda

    if fluid == 'air':
        rho0_v = 1.2;    c0_v = 340.0;   mu_v = 1.8e-5
    else:
        rho0_v = RHO0;   c0_v = C0;      mu_v = MU

    h    = 1.3 * dx
    dt   = 0.35 * h / (c0_v + v0)
    m    = rho0_v * dx * dx
    nout = max(1, int(1.0e-5 / dt))

    nxc = int(Lv / h) + 3
    nyc = int(Hv / h) + 3
    dc  = h

    x_h, y_h, vx_h, vy_h, rho_h, N = init_fluid(dx, rho0_v)

    THREADS = 256
    BLOCKS  = (N + THREADS - 1) // THREADS

    xpc = -(R_PROJ + dx);  ypc = Hv / 2.0
    vxp = v0;              vyp = 0.0

    B = rho0_v * c0_v * c0_v / GAMMA
    print(f"\n{'─'*65}")
    print(f"  2D SPH-HRAM  (Varas 2009,  {fluid.upper()},  GPU — Numba CUDA,  stiffener={stiffener})")
    print(f"  N={N}  dx={dx*1e3:.1f}mm  BLOCKS={BLOCKS}  THREADS={THREADS}")
    print(f"  v₀={v0} m/s  tend={tend*1e3:.2f} ms  steps≈{int(tend/dt):,}")
    print(f"  c₀={c0_v} m/s  B={B/1e6:.2f} MPa  α_AV={alpha_av}")
    print(f"{'─'*65}")

    d_x    = cuda.to_device(x_h)
    d_y    = cuda.to_device(y_h)
    d_vx   = cuda.to_device(vx_h)
    d_vy   = cuda.to_device(vy_h)
    d_rho  = cuda.to_device(rho_h)
    d_p    = cuda.device_array(N,           dtype=np.float64)
    d_ax   = cuda.device_array(N,           dtype=np.float64)
    d_ay   = cuda.device_array(N,           dtype=np.float64)
    d_vxh  = cuda.device_array(N,           dtype=np.float64)
    d_vyh  = cuda.device_array(N,           dtype=np.float64)
    d_xn   = cuda.device_array(N,           dtype=np.float64)
    d_yn   = cuda.device_array(N,           dtype=np.float64)
    d_head = cuda.device_array((nxc, nyc),  dtype=np.int32)
    d_nxt  = cuda.device_array(N,           dtype=np.int32)

    head_h, nxt_h = build_cells(x_h, y_h, N, dc, nxc, nyc)
    d_head.copy_to_device(head_h)
    d_nxt.copy_to_device(nxt_h)

    print("  Compiling GPU kernels ...", end="", flush=True)
    _gpu_density[BLOCKS, THREADS](d_x, d_y, m, N, h, dc, nxc, nyc,
                                   d_head, d_nxt, rho0_v, d_rho)
    _gpu_pressure[BLOCKS, THREADS](d_rho, N, rho0_v, c0_v, P_CAV, d_p)
    _gpu_accel[BLOCKS, THREADS](d_x, d_y, d_vx, d_vy, d_rho, d_p,
                                 m, N, h, dc, nxc, nyc, d_head, d_nxt,
                                 mu_v, alpha_av, c0_v, d_ax, d_ay)
    # initial projectile + baffle forces on host
    x_h, y_h, ax_p_h, ay_p_h, _, _ = proj_force(x_h, y_h, m, N, xpc, ypc, h, c0_v)
    if stiffener == 'transverse':
        x_h, y_h, ax_b_h, ay_b_h = baffle_force_transverse(x_h, y_h, N, h, dx, c0_v)
    elif stiffener == 'longitudinal':
        x_h, y_h, ax_b_h, ay_b_h = baffle_force_longitudinal(x_h, y_h, N, h, dx, c0_v)
    else:
        ax_b_h = ay_b_h = np.zeros(N, np.float64)
    ax_h = d_ax.copy_to_host() + ax_p_h + ax_b_h
    ay_h = d_ay.copy_to_host() + ay_p_h + ay_b_h
    d_ax.copy_to_device(ax_h)
    d_ay.copy_to_device(ay_h)
    cuda.synchronize()
    print(" done.")

    probe_list = []
    force_list = []
    t, step = 0.0, 0
    t0 = clk.perf_counter()

    while t <= tend:
        # half-kick on GPU
        _gpu_halfkick[BLOCKS, THREADS](d_vx, d_vy, d_ax, d_ay, dt, N, d_vxh, d_vyh)

        # advance projectile to t+dt
        xpc += vxp * dt
        ypc += vyp * dt

        # XSPH on GPU → d_xn, d_yn
        _gpu_xsph[BLOCKS, THREADS](d_x, d_y, d_vxh, d_vyh, d_rho,
                                    m, N, h, dc, nxc, nyc, d_head, d_nxt,
                                    dt, d_xn, d_yn)
        cuda.synchronize()

        # Copy new positions and half-velocities to host for BC + proj force
        x_h   = d_xn.copy_to_host()
        y_h   = d_yn.copy_to_host()
        vxh_h = d_vxh.copy_to_host()
        vyh_h = d_vyh.copy_to_host()

        # Projectile penalty force (also soft push-out) on CPU
        x_h, y_h, ax_p_h, ay_p_h, Fx, Fy = proj_force(
            x_h, y_h, m, N, xpc, ypc, h, c0_v)

        # Baffle force on CPU
        if stiffener == 'transverse':
            x_h, y_h, ax_b_h, ay_b_h = baffle_force_transverse(
                x_h, y_h, N, h, dx, c0_v)
        elif stiffener == 'longitudinal':
            x_h, y_h, ax_b_h, ay_b_h = baffle_force_longitudinal(
                x_h, y_h, N, h, dx, c0_v)
        else:
            ax_b_h = ay_b_h = np.zeros(N, np.float64)

        # Wall BC on CPU
        x_h, y_h, vxh_h, vyh_h = wall_bc(x_h, y_h, vxh_h, vyh_h, N, R_PROJ, dx)

        # Update projectile velocity (reaction force, Newton 3rd law)
        vxp += Fx / M_PROJ * dt
        vyp += Fy / M_PROJ * dt - 9.81 * dt

        # Copy modified positions and velocities back to device
        d_xn.copy_to_device(x_h)
        d_yn.copy_to_device(y_h)
        d_vxh.copy_to_device(vxh_h)
        d_vyh.copy_to_device(vyh_h)
        d_x, d_xn = d_xn, d_x   # swap buffers
        d_y, d_yn = d_yn, d_y

        # Rebuild cell list and upload
        head_h, nxt_h = build_cells(x_h, y_h, N, dc, nxc, nyc)
        d_head.copy_to_device(head_h)
        d_nxt.copy_to_device(nxt_h)

        # Fluid–fluid density, pressure, acceleration on GPU
        _gpu_density[BLOCKS, THREADS](d_x, d_y, m, N, h, dc, nxc, nyc,
                                       d_head, d_nxt, rho0_v, d_rho)
        _gpu_pressure[BLOCKS, THREADS](d_rho, N, rho0_v, c0_v, P_CAV, d_p)
        _gpu_accel[BLOCKS, THREADS](d_x, d_y, d_vxh, d_vyh, d_rho, d_p,
                                     m, N, h, dc, nxc, nyc, d_head, d_nxt,
                                     mu_v, alpha_av, c0_v, d_ax, d_ay)
        cuda.synchronize()
        # Add projectile + baffle forces (computed on CPU) to d_ax, d_ay
        ax_h = d_ax.copy_to_host() + ax_p_h + ax_b_h
        ay_h = d_ay.copy_to_host() + ay_p_h + ay_b_h
        d_ax.copy_to_device(ax_h)
        d_ay.copy_to_device(ay_h)
        # full kick on GPU
        _gpu_fullkick[BLOCKS, THREADS](d_vxh, d_vyh, d_ax, d_ay, dt, N, d_vx, d_vy)

        if step % nout == 0:
            cuda.synchronize()
            p_h    = d_p.copy_to_host()
            probes = probe_pressures(x_h, y_h, p_h, N, h)
            F_exit = wall_force_exit(x_h, p_h, N, h)
            probe_list.append((t, probes[0], probes[1], probes[2]))
            force_list.append((t, F_exit, vxp, xpc))
            if step % (nout * 20) == 0 and t > 0.0:
                el  = clk.perf_counter() - t0
                eta = el / t * (tend - t) if t > 0 else 0.0
                print(f"  t={t*1e6:6.1f}µs  xp={xpc*1e3:6.1f}mm"
                      f"  vp={vxp:7.1f}m/s  p_ctr={probes[1]/1e6:6.3f}MPa"
                      f"  F_exit={F_exit:.2f}N"
                      f"  [{el:.1f}s  ETA {eta:.1f}s]")

        t    += dt
        step += 1

    cuda.synchronize()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    tag     = f"{fluid}_v{int(v0)}_dx{dx*1e3:.1f}mm_a{alpha_av}_s{stiffener}"
    pr_arr  = np.array(probe_list)
    fm_arr  = np.array(force_list)
    np.savetxt(os.path.join(out_dir, f'hram_pressure_probes_{tag}.dat'), pr_arr,
               fmt='%14.6e',
               header='t[s]  p_entry[Pa]  p_center[Pa]  p_exit[Pa]')
    np.savetxt(os.path.join(out_dir, f'hram_wall_force_{tag}.dat'), fm_arr,
               fmt='%14.6e',
               header='t[s]  F_exit[N]  v_proj[m/s]  x_proj[m]')

    elapsed = clk.perf_counter() - t0
    print(f"\n  Done. {step:,} steps in {elapsed:.1f}s")
    print(f"  hram_pressure_probes_{tag}.dat")
    print(f"  hram_wall_force_{tag}.dat")

    if make_plot:
        plot_results(pr_arr, fm_arr, dx, fluid, v0, alpha_av, stiffener)

    return pr_arr, fm_arr


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
def plot_results(pr, fm, dx, fluid, v0, alpha_av, stiffener='none'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_us = pr[:, 0] * 1e6   # convert to µs
    t_fm = fm[:, 0] * 1e6

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f'2D SPH-HRAM  ({fluid.upper()},  v₀={v0} m/s,  dx={dx*1e3:.1f} mm,  α={alpha_av},  stiffener={stiffener})',
        fontsize=12)

    # Pressure probes
    ax0 = axes[0, 0]
    ax0.plot(t_us, pr[:, 1] / 1e6, label='P1 near entry (x=20 mm)',  lw=1.2)
    ax0.plot(t_us, pr[:, 2] / 1e6, label='P2 centre (x=100 mm)',     lw=1.2)
    ax0.plot(t_us, pr[:, 3] / 1e6, label='P3 near exit (x=180 mm)',  lw=1.2)
    ax0.set_xlabel('Time (µs)');  ax0.set_ylabel('Pressure (MPa)')
    ax0.set_title('Pressure history at probe locations')
    ax0.legend(fontsize=8);  ax0.grid(True, alpha=0.3)

    # Exit wall force
    ax1 = axes[0, 1]
    ax1.plot(t_fm, fm[:, 1], 'r-', lw=1.2)
    ax1.set_xlabel('Time (µs)');  ax1.set_ylabel('Force (N/m depth)')
    ax1.set_title('Exit wall force  F = ⟨p_exit⟩ × H_v')
    ax1.grid(True, alpha=0.3)

    # Projectile velocity
    ax2 = axes[1, 0]
    ax2.plot(t_fm, fm[:, 2], 'g-', lw=1.2)
    ax2.set_xlabel('Time (µs)');  ax2.set_ylabel('v_proj (m/s)')
    ax2.set_title('Projectile deceleration')
    ax2.grid(True, alpha=0.3)

    # Projectile position
    ax3 = axes[1, 1]
    ax3.plot(t_fm, fm[:, 3] * 1e3, 'b-', lw=1.2)
    ax3.axvline(0, color='k', lw=0.5, ls='--')
    ax3.axhline(Lv * 1e3, color='r', lw=0.8, ls='--', label='exit wall')
    ax3.set_xlabel('Time (µs)');  ax3.set_ylabel('x_proj (mm)')
    ax3.set_title('Projectile position')
    ax3.legend(fontsize=8);  ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    tag = f"{fluid}_v{int(v0)}_dx{dx*1e3:.1f}mm_a{alpha_av}_s{stiffener}"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f'hram_{tag}.png')
    plt.savefig(out, dpi=150)
    print(f"  Plot → {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='2D SPH Hydraulic Ram simulation (Varas et al. 2009)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--dx',    type=float, default=0.002,
                    help='particle spacing [m]  (0.002=2mm standard)')
    ap.add_argument('--tend',  type=float, default=5e-4,
                    help='simulation end time [s]')
    ap.add_argument('--v0',    type=float, default=V0_DEF,
                    help='projectile initial velocity [m/s]')
    ap.add_argument('--fluid', type=str,   default='water',
                    choices=['water', 'air'],
                    help='vessel fill medium  (objective d: run both)')
    ap.add_argument('--alpha', type=float, default=ALPHA_AV,
                    help='Monaghan AV coefficient α  (1.0 recommended for shocks)')
    ap.add_argument('--stiffener', type=str, default='none',
                    choices=['none', 'transverse', 'longitudinal'],
                    help='internal stiffener configuration  (objective e)')
    ap.add_argument('--plot',  action='store_true',
                    help='save 4-panel PNG plot of results')
    ap.add_argument('--gpu',   action='store_true',
                    help='run on NVIDIA GPU via Numba CUDA')
    args = ap.parse_args()

    kwargs = dict(dx=args.dx, tend=args.tend, make_plot=args.plot,
                  v0=args.v0, fluid=args.fluid, alpha_av=args.alpha,
                  stiffener=args.stiffener)

    if args.gpu:
        if not _CUDA_OK:
            print("ERROR: No CUDA GPU detected. Run without --gpu.")
        else:
            run_gpu(**kwargs)
    else:
        run(**kwargs)

#!/usr/bin/env python3
"""
hram_stiffener_compare.py — objective (e)

Runs sph_hram.py for the three stiffener configurations and produces a
4-panel comparison figure:
  (A) P3 (exit probe) pressure vs time
  (B) Exit wall force vs time
  (C) Projectile velocity vs time
  (D) Bar chart: peak P3 and peak F_exit per configuration
"""
import subprocess, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, 'sph_hram.py')
PYTHON = sys.executable

DX    = 0.002    # 2 mm  — same as validation runs
TEND  = 5e-4     # 0.5 ms
ALPHA = 2.0
V0    = 900

CONFIGS = [
    ('none',         'b',  '-',  'No stiffener  (baseline)'),
    ('transverse',   'r',  '--', 'Transverse rib  (x = L/2)'),
    ('longitudinal', 'g',  ':',  'Longitudinal stringers  (y = H/4, 3H/4)'),
]

# ── Run simulations if output files are missing ───────────────────────────────
for stiff, _, _, label in CONFIGS:
    tag  = f"water_v{int(V0)}_dx{DX*1e3:.1f}mm_a{ALPHA}_s{stiff}"
    pfile = os.path.join(HERE, f'hram_pressure_probes_{tag}.dat')
    if not os.path.isfile(pfile):
        print(f"\n  Running: stiffener={stiff}")
        cmd = [PYTHON, SCRIPT,
               '--dx',       str(DX),
               '--tend',     str(TEND),
               '--v0',       str(V0),
               '--fluid',    'water',
               '--alpha',    str(ALPHA),
               '--stiffener', stiff]
        subprocess.run(cmd, check=True)
    else:
        print(f"  Found existing data: {os.path.basename(pfile)}")

# ── Load results ──────────────────────────────────────────────────────────────
data = {}
for stiff, col, ls, label in CONFIGS:
    tag  = f"water_v{int(V0)}_dx{DX*1e3:.1f}mm_a{ALPHA}_s{stiff}"
    pr   = np.loadtxt(os.path.join(HERE, f'hram_pressure_probes_{tag}.dat'))
    fm   = np.loadtxt(os.path.join(HERE, f'hram_wall_force_{tag}.dat'))
    data[stiff] = dict(pr=pr, fm=fm, col=col, ls=ls, label=label)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    f'SPH-HRAM: Stiffener configurations comparison\n'
    f'(water,  v₀={V0} m/s,  dx={DX*1e3:.0f} mm,  α={ALPHA})',
    fontsize=12)

axA, axB = axes[0, 0], axes[0, 1]
axC, axD = axes[1, 0], axes[1, 1]

t_xlim = (0, 500)   # µs

for stiff, col, ls, label in CONFIGS:
    d  = data[stiff]
    pr = d['pr'];  fm = d['fm']
    t_us = pr[:, 0] * 1e6
    t_fm = fm[:, 0] * 1e6

    # clip blow-up values
    p3   = np.clip(pr[:, 3] / 1e6,  -5, 600)
    Fe   = fm[:, 1]
    mask = np.abs(Fe) < 1e10
    Fe_p = np.where(mask, np.abs(Fe), np.nan)

    axA.plot(t_us, p3,       color=col, ls=ls, lw=1.5, label=label)
    axB.semilogy(t_fm[mask], Fe_p[mask] + 0.001,
                 color=col, ls=ls, lw=1.5, label=label)
    axC.plot(t_fm, fm[:, 2], color=col, ls=ls, lw=1.5, label=label)

axA.set_xlabel('Time (µs)');  axA.set_ylabel('Pressure (MPa)')
axA.set_title('(A) P3 exit-probe pressure')
axA.set_xlim(*t_xlim);  axA.set_ylim(-5, 400)
axA.axhline(0, color='k', lw=0.4, ls=':')
axA.legend(fontsize=8);  axA.grid(True, alpha=0.25)

axB.set_xlabel('Time (µs)');  axB.set_ylabel('|F_exit| (N/m) — log scale')
axB.set_title('(B) Exit wall force')
axB.set_xlim(*t_xlim)
axB.legend(fontsize=8);  axB.grid(True, alpha=0.25)

axC.set_xlabel('Time (µs)');  axC.set_ylabel('v_proj (m/s)')
axC.set_title('(C) Projectile deceleration')
axC.set_xlim(*t_xlim)
axC.legend(fontsize=8);  axC.grid(True, alpha=0.25)

# Panel D: bar chart of peak metrics
labels_bar = [c[3] for c in CONFIGS]
cols_bar   = [c[1] for c in CONFIGS]
pk_P3, pk_Fe, vp_fin = [], [], []

for stiff, _, _, _ in CONFIGS:
    d  = data[stiff]
    pr = d['pr'];  fm = d['fm']
    mask = pr[:, 0] < 4e-4
    pk_P3.append(pr[mask, 3].max() / 1e6)
    Fe_ok = fm[:, 1][np.abs(fm[:, 1]) < 1e10]
    pk_Fe.append(Fe_ok.max() / 1e6 if len(Fe_ok) > 0 else 0.0)
    vp_fin.append(fm[-1, 2])

x_pos  = np.arange(len(CONFIGS))
width  = 0.28
short_labels = ['None', 'Transverse', 'Longit.']

b1 = axD.bar(x_pos - width, pk_P3, width, label='Peak P3 (MPa)',
             color=cols_bar, alpha=0.75)
b2 = axD.bar(x_pos,         pk_Fe, width, label='Peak F_exit (MN/m)',
             color=cols_bar, alpha=0.45, hatch='//')
b3 = axD.bar(x_pos + width, [900 - v for v in vp_fin], width,
             label='Δv_proj (m/s)',
             color=cols_bar, alpha=0.30, hatch='xx')

axD.set_xticks(x_pos)
axD.set_xticklabels(short_labels)
axD.set_ylabel('Magnitude (see legend)')
axD.set_title('(D) Peak values per configuration  (t < 400 µs)')
axD.legend(fontsize=8)
axD.grid(True, alpha=0.25, axis='y')

# annotate bar values
for rect, val in zip(b1, pk_P3):
    axD.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1,
             f'{val:.0f}', ha='center', va='bottom', fontsize=7)
for rect, val in zip(b2, pk_Fe):
    axD.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1,
             f'{val:.1f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
outfile = os.path.join(HERE, 'hram_stiffener_comparison.png')
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outfile}")
