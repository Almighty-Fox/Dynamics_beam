#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anisotropic potential map with visible angular minima.
V(r,θ) = 1/r^4 + (α cos(2θ) - 1)/r^2

• Центр r<r_min маскируется (нет «выжигания»).
• Цветовая шкала обрезается по перцентилям (vmin/vmax), чтобы контраст ям был виден.
• Можно включить лог-скейл данных (log(1+V_clipped)).
• Теоретические положения минимумов размечаются маркерами.

Зависимости: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Params:
    alpha: float = 1.5          # сила анизотропии; 1.2–2.0 хорошо видно 4 ямки
    r_min: float = 0.15         # маска центра: r < r_min не рисуем
    r_max: float = 2.5          # окно визуализации
    N: int = 900                # разрешение сетки (N x N)
    pct_low: float = 5.0        # нижний перцентиль для vmin
    pct_high: float = 95.0      # верхний перцентиль для vmax
    use_log: bool = False       # True → рисовать log(1 + V_clipped)
    cmap: str = "inferno"       # колормэп с хорошим контрастом
    n_levels: int = 12          # число уровней изолиний
    save_path: str = "V_map.png"

def compute_potential(alpha, X, Y):
    R = np.hypot(X, Y)
    Theta = np.arctan2(Y, X)
    with np.errstate(divide='ignore', invalid='ignore'):
        V = (1.0 / R**4) - (1.0 / R**2) + (alpha * np.cos(2.0 * Theta)) / (R**2)
    return V, R

def minima_positions(alpha):
    """
    Теоретические положения минимумов.
    α>0 → диагонали θ = π/4 + kπ/2, r* = sqrt(2/(1+α))
    α<0 → оси      θ = kπ/2,        r* = sqrt(2/(1-α))
    Возвращает массив точек (x, y).
    """
    if alpha == 0:
        r_star = np.sqrt(2.0)
        thetas = np.linspace(0, 2*np.pi, 200, endpoint=False)  # целое кольцо
        return r_star * np.column_stack((np.cos(thetas), np.sin(thetas)))
    if alpha > 0:
        r_star = np.sqrt(2.0 / (1.0 + alpha))
        thetas = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
    else:  # alpha < 0
        r_star = np.sqrt(2.0 / (1.0 - alpha))
        thetas = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    pts = np.column_stack((r_star*np.cos(thetas), r_star*np.sin(thetas)))
    return pts


def stationary_points(alpha):
    """Возвращает список (x, y, kind) для kind in {'min','max','saddle'}."""
    thetas = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    pts = []
    for th in thetas:
        c = np.cos(2*th)
        denom = 1.0 - alpha*c
        if denom <= 0:  # нет конечного крит. радиуса
            continue
        r = np.sqrt(2.0/denom)
        # вторые производные в крит. точке
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


def plot_map(P: Params):
    # сетка
    x = np.linspace(-P.r_max, P.r_max, P.N)
    y = np.linspace(-P.r_max, P.r_max, P.N)
    X, Y = np.meshgrid(x, y, indexing='xy')

    # потенциал и маска центра
    V, R = compute_potential(P.alpha, X, Y)
    mask = (R < P.r_min) | ~np.isfinite(V)
    Vm = np.ma.masked_array(V, mask=mask)

    # перцентильная обрезка диапазона
    data = Vm.filled(np.nan)
    vmin = np.nanpercentile(data, P.pct_low)
    vmax = np.nanpercentile(data, P.pct_high)

    # опциональный log-скейл (после клипа)
    Vplot = np.clip(Vm, vmin, vmax)
    if P.use_log:
        with np.errstate(invalid='ignore'):
            Vplot = np.ma.masked_array(np.log1p(Vplot), mask=mask)
        # лог-скейл меняет диапазон; для контура можно взять равномерно по Vplot:
        vmin_plot = np.nanmin(Vplot.filled(np.nan))
        vmax_plot = np.nanmax(Vplot.filled(np.nan))
        levels = np.linspace(vmin_plot, vmax_plot, P.n_levels)
    else:
        levels = np.linspace(vmin, vmax, P.n_levels)

    # рисование
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        Vplot, extent=[-P.r_max, P.r_max, -P.r_max, P.r_max], origin='lower',
        cmap=P.cmap, interpolation='bilinear'
    )
    cs = ax.contour(X, Y, Vplot, levels=levels, linewidths=0.8, colors='w', alpha=0.6)
    ax.clabel(cs, fmt="%.2g", fontsize=8, inline=True)

    # # отметим теоретические минимумы
    # pts = minima_positions(P.alpha)
    # if pts.shape[0] > 0:
    #     ax.plot(pts[:, 0], pts[:, 1], 'o', ms=6, mfc='cyan', mec='k', label='minima (theory)')
    #     ax.legend(loc='upper right', fontsize=9, frameon=True)

    sp = stationary_points(P.alpha)
    for x0, y0, kind in sp:
        if kind == 'min':
            ax.plot(x0, y0, 'o', ms=7, mfc='cyan', mec='k',
                    label='min' if 'min' not in ax.get_legend_handles_labels()[1] else None)
        elif kind == 'max':
            ax.plot(x0, y0, '^', ms=7, mfc='tomato', mec='k',
                    label='max' if 'max' not in ax.get_legend_handles_labels()[1] else None)
        else:  # saddle
            ax.plot(x0, y0, 'x', ms=8, mew=2, color='yellow',
                    label='saddle' if 'saddle' not in ax.get_legend_handles_labels()[1] else None)
    ax.legend(loc='upper right', fontsize=9, frameon=True)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ttl = (r'$V(r,\theta)=\frac{1}{r^{4}}+\frac{\alpha\cos 2\theta-1}{r^{2}}$'
           f'\nα={P.alpha}, masked r<{P.r_min}, window ±{P.r_max}')
    if P.use_log:
        ttl += "  (log scale)"
    ax.set_title(ttl)
    cbar = fig.colorbar(im, ax=ax, label='V' + (' (log1p)' if P.use_log else ''))

    fig.tight_layout()
    # fig.savefig(P.save_path, dpi=240)
    # print(f"Saved: {P.save_path}")
    plt.show()

if __name__ == "__main__":
    # Подбито для хорошей видимости 4 минимумов:
    P = Params(
        alpha=1.5,     # ↑/↓ меняйте для контраста асимметрии (1.2–2.0 наглядно)
        r_min=0.15,    # маска центра — не слишком большая, чтобы не срезать ямки
        r_max=2.5,     # окно, где r*≈sqrt(2/(1±α)) попадает ~ в [0.8,1.2]
        N=900,
        pct_low=5.0, pct_high=95.0,
        use_log=False,           # True — если контраст всё ещё «забит» центром
        cmap="inferno",
        n_levels=12,
        save_path="V_map_alpha1p5.png"
    )
    plot_map(P)
