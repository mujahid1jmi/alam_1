
#!/usr/bin/env python3
"""
HRAM validation for available water dataset only.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

outdir = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# constants
# --------------------------------------------------
c0   = 1480.0
B    = 313.0
rho0 = 1000.0

# --------------------------------------------------
# load files
# --------------------------------------------------
def load_water():

    tag = "water_v900_dx2.0mm_a2.0"

    pfile = os.path.join(
        outdir,
        f"hram_pressure_probes_{tag}.dat"
    )

    ffile = os.path.join(
        outdir,
        f"hram_wall_force_{tag}.dat"
    )

    if not os.path.exists(pfile):
        raise FileNotFoundError(pfile)

    if not os.path.exists(ffile):
        raise FileNotFoundError(ffile)

    pr = np.loadtxt(pfile)
    fm = np.loadtxt(ffile)

    return pr, fm


def tait_c(p_mpa):
    rr = (p_mpa/B + 1.0)**(1.0/7.0)
    return c0*rr**3


def first_crossing(t, p, thr=1.0):

    for i in range(1, len(p)):
        if p[i] > thr and p[i-1] <= thr:
            return t[i]

    return None


# --------------------------------------------------
# load data
# --------------------------------------------------
pr, fm = load_water()

t_us = pr[:,0]*1e6

p1 = pr[:,1]/1e6
p2 = pr[:,2]/1e6
p3 = pr[:,3]/1e6

vp = fm[:,2]

# --------------------------------------------------
# figure
# --------------------------------------------------
fig = plt.figure(figsize=(14,10))

gs = gridspec.GridSpec(
    2,
    2,
    hspace=0.35,
    wspace=0.30
)

# ==================================================
# Panel A
# ==================================================
axA = fig.add_subplot(gs[0,0])

axA.plot(t_us,p1,'b-',label='P1')
axA.plot(t_us,p2,'g-',label='P2')
axA.plot(t_us,p3,'r-',label='P3')

for xp,col in [
    (0.020,'b'),
    (0.100,'g'),
    (0.180,'r')
]:

    texp = xp/c0*1e6

    axA.axvline(
        texp,
        color=col,
        ls='--',
        lw=0.8
    )

axA.set_title("Pressure History")
axA.set_xlabel("Time (µs)")
axA.set_ylabel("Pressure (MPa)")
axA.grid(True)
axA.legend()

# ==================================================
# Panel B
# ==================================================
axB = fig.add_subplot(gs[0,1])

axB.plot(
    fm[:,0]*1e6,
    vp,
    'k-',
    lw=1.5
)

axB.set_title("Projectile Velocity")
axB.set_xlabel("Time (µs)")
axB.set_ylabel("Velocity (m/s)")
axB.grid(True)

# ==================================================
# Panel C
# ==================================================
axC = fig.add_subplot(gs[1,0])

t_obs = first_crossing(
    pr[:,0]*1e6,
    p2,
    1.0
)

if t_obs is not None:

    c_obs = 0.100/t_obs*1e6

    axC.axhline(
        t_obs,
        color='r',
        ls='--',
        label=f'Observed = {t_obs:.1f} µs'
    )

pvals = np.linspace(0,400,200)

cvals = np.array(
    [tait_c(p) for p in pvals]
)

tarr = 0.100/cvals*1e6

axC.plot(
    pvals,
    tarr,
    'k-',
    label='Tait EOS'
)

axC.set_xlabel("Pressure (MPa)")
axC.set_ylabel("Arrival Time (µs)")
axC.set_title("Wave-Speed Validation")
axC.grid(True)
axC.legend()

# ==================================================
# Panel D
# ==================================================
axD = fig.add_subplot(gs[1,1])

summary = []

summary.append("Available dataset:")
summary.append("water, α=2.0")
summary.append("")

summary.append(
    f"P1 peak = {np.max(p1):.1f} MPa"
)

summary.append(
    f"P2 peak = {np.max(p2):.1f} MPa"
)

summary.append(
    f"P3 peak = {np.max(p3):.1f} MPa"
)

summary.append("")

summary.append(
    f"Final projectile velocity = {vp[-1]:.1f} m/s"
)

if t_obs is not None:

    summary.append(
        f"Observed wave speed = {c_obs:.1f} m/s"
    )

axD.axis("off")

axD.text(
    0.02,
    0.98,
    "\n".join(summary),
    va="top",
    fontsize=10, 
    family="monospace"
)

plt.savefig(
    os.path.join(
        outdir,
        "hram_validation.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

print()
print("Saved : hram_validation.png")
print()
print("Peak P1 =", np.max(p1), "MPa")
print("Peak P2 =", np.max(p2), "MPa")
print("Peak P3 =", np.max(p3), "MPa")
print("Final projectile velocity =", vp[-1], "m/s")