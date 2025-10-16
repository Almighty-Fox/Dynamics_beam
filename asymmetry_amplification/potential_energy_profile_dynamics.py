#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anisotropic potential with particle dynamics.
V(r,θ) = 1/r^4 + (α cos 2θ - 1)/r^2

Integrates Hamilton ODEs in (r, θ, p_r, p_θ):
    ṙ   =  p_r / m
    θ̇   =  p_θ / (m r^2)
    ṗ_r =  p_θ^2/(m r^3) + 4/r^5 + 2(α cos 2θ - 1)/r^3
    ṗ_θ =  2α sin 2θ / r^2

• Center r<r_min is masked on the map; integration stops at r < r_stop.
• Trajectory overlaid on imshow+contours.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import bisect

# ---------- parameters ----------
@dataclass
class Params:
    alpha: float = 1.5
    m: float = 1.0
    r_min_mask: float = 0.15     # mask for visualization
    r_stop: float = 0.10         # hard stop for ODE if r gets too small
    r_max: float = 2.5           # window for plot
    N: int = 900                 # grid resolution
    pct_low: float = 5.0
    pct_high: float = 95.0
    use_log: bool = False
    cmap: str = "inferno"
    n_levels: int = 12
    t_span: tuple = (0.0, 8e-2)  # time interval
    t_points: int = 6000         # output samples along trajectory
    method: str = "DOP853"       # integrator


# ---------- potential and helpers ----------
def V_of_rtheta(r, th, alpha):
    return 1.0/r**4 + (alpha*np.cos(2.0*th) - 1.0)/r**2

def compute_potential(alpha, X, Y):
    R = np.hypot(X, Y)
    Theta = np.arctan2(Y, X)
    with np.errstate(divide='ignore', invalid='ignore'):
        V = (1.0 / R**4) - (1.0 / R**2) + (alpha * np.cos(2.0 * Theta)) / (R**2)
    return V, R

def stationary_points(alpha):
    """All stationary points with classification."""
    thetas = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    pts = []
    for th in thetas:
        c = np.cos(2*th)
        denom = 1.0 - alpha*c
        if denom <= 0:
            continue
        r = np.sqrt(2.0/denom)
        Vtt = -4.0*alpha*c / (r**2)
        Vrr = 20.0/(r**6) + 6.0*(alpha*c - 1.0)/(r**4)
        if Vrr > 0 and Vtt > 0:
            kind = 'min'
        elif Vrr < 0 and Vtt < 0:
            kind = 'max'
        else:
            kind = 'saddle'
        pts.append((r*np.cos(th), r*np.sin(th), kind))
    return pts

# ---------- Hamiltonian ODE ----------
def rhs(t, y, par: Params):
    r, th, pr, pth = y
    m, a = par.m, par.alpha
    # avoid zero division in the rare case solver steps onto r<=0
    if r <= 0:
        r = 1e-12
    r2, r3, r5 = r*r, r**3, r**5
    cr2 = np.cos(2.0*th)
    sr2 = np.sin(2.0*th)

    rdot   = pr / m
    thdot  = pth / (m * r2)
    prdot  = (pth*pth)/(m * r3) + 4.0/r5 + 2.0*(a*cr2 - 1.0)/r3
    pthdot = 2.0*a*sr2 / r2
    return (rdot, thdot, prdot, pthdot)

def energy(y, par: Params):
    r, th, pr, pth = y
    m, a = par.m, par.alpha
    T = 0.5*(pr*pr)/m + 0.5*(pth*pth)/(m*r*r)
    U = V_of_rtheta(r, th, a)
    return T + U

# stop events
def evt_hit_center(t, y, par: Params):
    return y[0] - par.r_stop
evt_hit_center.terminal = True
evt_hit_center.direction = -1

def evt_out_of_window(t, y, par: Params):
    return par.r_max - y[0]
evt_out_of_window.terminal = True
evt_out_of_window.direction = -1

# ---------- plotting ----------
def plot_map(par: Params):
    x = np.linspace(-par.r_max, par.r_max, par.N)
    y = np.linspace(-par.r_max, par.r_max, par.N)
    X, Y = np.meshgrid(x, y, indexing='xy')

    V, R = compute_potential(par.alpha, X, Y)
    mask = (R < par.r_min_mask) | ~np.isfinite(V)
    Vm = np.ma.masked_array(V, mask=mask)

    data = Vm.filled(np.nan)
    vmin = np.nanpercentile(data, par.pct_low)
    vmax = np.nanpercentile(data, par.pct_high)

    Vimg = np.clip(Vm, vmin, vmax)
    if par.use_log:
        with np.errstate(invalid='ignore'):
            Vimg = np.ma.masked_array(np.log1p(Vimg), mask=mask)
        levels = np.linspace(np.nanmin(Vimg), np.nanmax(Vimg), par.n_levels)
    else:
        levels = np.linspace(vmin, vmax, par.n_levels)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    im = ax.imshow(
        Vimg, extent=[-par.r_max, par.r_max, -par.r_max, par.r_max],
        origin='lower', cmap=par.cmap, interpolation='bilinear'
    )
    # contours over the TRUE potential (without clipping), for crisp shape
    cs = ax.contour(X, Y, V, levels=levels, colors='w', linewidths=0.8, alpha=0.6)
    ax.clabel(cs, fmt="%.2g", fontsize=8, inline=True)

    # stationary points
    for x0, y0, kind in stationary_points(par.alpha):
        if kind == 'min':
            ax.plot(x0, y0, 'o', ms=7, mfc='cyan', mec='k', label='min' if 'min' not in ax.get_legend_handles_labels()[1] else None)
        elif kind == 'max':
            ax.plot(x0, y0, '^', ms=7, mfc='tomato', mec='k', label='max' if 'max' not in ax.get_legend_handles_labels()[1] else None)
        else:
            ax.plot(x0, y0, 'x', ms=8, mew=2, color='yellow', label='saddle' if 'saddle' not in ax.get_legend_handles_labels()[1] else None)

    ax.legend(loc='upper right', fontsize=9, frameon=True)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    title = (r'$V(r,\theta)=\frac{1}{r^{4}}+\frac{\alpha\cos 2\theta-1}{r^{2}}$'
             f'\nα={par.alpha}, masked r<{par.r_min_mask}, window ±{par.r_max}')
    if par.use_log: title += " (log scale)"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='V' + (' (log1p)' if par.use_log else ''))
    return fig, ax

# ---------- run one scenario ----------
def simulate_and_plot(par: Params,
                      r0=None, th0=None, rdot0=None, thdot0=None,
                      pr0=None, pth0=None):
    """
    You may specify either velocities (rdot0, thdot0) or momenta (pr0, pth0).
    """
    # sensible demo IC near a true minimum for α>0: θ≈π/2, r≈sqrt(2/(1+α))
    if r0 is None or th0 is None:
        r0 = np.sqrt(2.0/(1.0 + par.alpha))
        th0 = np.pi/2 + 0.18  # small angular offset
    if (pr0 is None) != (rdot0 is None):
        pr0 = par.m * rdot0 if pr0 is None else pr0
    if (pth0 is None) != (thdot0 is None):
        pth0 = par.m * (r0**2) * thdot0 if pth0 is None else pth0
    if pr0 is None:     # if neither velocities nor momenta were given
        pr0 = 0.00*par.m
    if pth0 is None:
        pth0 = 0.45*par.m*(r0**2)  # give some rotation

    y0 = np.array([r0, th0, pr0, pth0], dtype=float)

    # integrate
    t_eval = np.linspace(par.t_span[0], par.t_span[1], par.t_points)
    sol = solve_ivp(lambda t, y: rhs(t, y, par),
                    par.t_span, y0, method=par.method, t_eval=t_eval,
                    rtol=1e-9, atol=1e-12,
                    events=[lambda t,y: evt_hit_center(t,y,par),
                            lambda t,y: evt_out_of_window(t,y,par)])
    r, th, pr, pth = sol.y
    x, y = r*np.cos(th), r*np.sin(th)

    # cutting trajectory if the position is becoming bigger than 2.0
    i_x = bisect.bisect_right(np.abs(x), 2)  # индекс первого > x
    i_y = bisect.bisect_right(np.abs(y), 2)  # индекс первого > x
    cut_x = x[:min(i_x, i_y)]
    cut_y = y[:min(i_x, i_y)]


    # energy drift (diagnostic)
    E = np.array([energy(state, par) for state in sol.y.T])
    drift = (E.max()-E.min())/max(1.0, abs(E.mean()))

    # plot background + trajectory
    fig, ax = plot_map(par)
    ax.plot(cut_x, cut_y, lw=2.0, label='trajectory', zorder=10)
    ax.plot(x[0], y[0], 'o', mfc='lime', mec='k', ms=7, label='start', zorder=11)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_title(ax.get_title() + f"\nEnergy drift ≈ {drift:.2e}; steps={len(sol.t)}")

    # optional: mark stop circle
    circ = plt.Circle((0,0), par.r_stop, ec='cyan', fc='none', ls='--', lw=1.2, alpha=0.7)
    ax.add_artist(circ)
    plt.tight_layout()
    plt.show()

    return sol

# ---------- run demo ----------
if __name__ == "__main__":
    P = Params(alpha=1.5, use_log=False)
    # Example 1: set by velocities (here: small radial, moderate angular)
    simulate_and_plot(P, rdot0=-1000.0, thdot0=2.0)
    # Example 2 (uncomment): set by momenta directly
    # simulate_and_plot(P, pr0=0.0, pth0=0.5*P.m*( (np.sqrt(2/(1+P.alpha)))**2 ))
