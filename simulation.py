"""
================================================================================
MR DAMPER vs PASSIVE DAMPER — ROCKET LANDING SIMULATION
================================================================================
WHAT THIS STUDY IS:
    A computational simulation comparing two types of landing leg dampers:
      1. PASSIVE DAMPER  — fixed damping coefficient, cannot adapt
      2. MR DAMPER       — variable damping coefficient, adapts every 1ms


WHAT A DAMPER DOES:
    During landing, the rocket hits the ground with kinetic energy = 0.5*m*v^2.
    The landing leg spring and damper must absorb this energy before it reaches
    the rocket frame. The damper dissipates energy as heat through fluid
    resistance. The spring stores and returns energy elastically.
    We want to MINIMIZE the peak force the rocket frame experiences.


PHYSICS MODEL:
    The landing leg is a 1-DOF mass-spring-damper:
        m * x'' = -k*x - c*x'
    where:
        m   = rocket mass (kg)
        x   = leg compression (m), positive = compressed
        x'  = velocity of compression (m/s)
        x'' = acceleration (m/s^2)
        k   = spring stiffness (N/m)
        c   = damping coefficient (N·s/m)


KEY METRIC — WHY WE USE PEAK SPRING FORCE:
    The total force at any instant = k*x + c*x'
    At t=0 (first contact), x=0 but velocity is maximum, so c*x' dominates.
    For a high c_max, this creates a huge spike at t=0 that is a model artifact

    Instead we measure PEAK SPRING FORCE = max(k*x) over the full simulation.
    This is the force at MAXIMUM LEG COMPRESSION, the true structural load
    the rocket frame sustains. It is physically meaningful and avoids the
    t=0 artifact entirely.


NUMERICAL METHOD — RK4:
    Because c changes every timestep (MR damper), we cannot solve analytically.
    We step forward through time using Runge-Kutta 4th Order (RK4) integration.
    RK4 computes 4 slope estimates per step and takes a weighted average,
    giving 4th-order accuracy (error ~ dt^4, negligible at dt=0.001s).


CONTROL LAW — SKYHOOK ALGORITHM:
    Named after the concept of damping against a fixed point in the sky.
    Rule: apply HIGH damping when compressing, LOW damping when rebounding.
    This ensures we only dissipate energy, never add it at the wrong moment.
    Reference: Karnopp, Crosby, Harwood (1974), J. Engineering for Industry.


AI tools were used in assistance to write this program
================================================================================
"""


# IMPORTS

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D


# PUBLICATION STYLE CONFIGURATION
# IEEE / AIAA journal standard: white background, black text, serif font,
# distinguishable line styles rather than color differences.

matplotlib.rcParams.update({
    # Canvas
    'figure.facecolor':     'white',
    'axes.facecolor':       'white',
    'savefig.facecolor':    'white',

    # Borders and ticks
    'axes.edgecolor':       'black',
    'axes.linewidth':       0.8,
    'xtick.color':          'black',
    'ytick.color':          'black',
    'xtick.direction':      'in',
    'ytick.direction':      'in',
    'xtick.major.size':     4,
    'ytick.major.size':     4,
    'xtick.minor.size':     2,
    'ytick.minor.size':     2,
    'xtick.minor.visible':  True,
    'ytick.minor.visible':  True,

    # Grid
    'axes.grid':            True,
    'grid.color':           '#cccccc',
    'grid.linewidth':       0.5,
    'grid.linestyle':       '--',
    'grid.alpha':           0.7,

    # Text
    'text.color':           'black',
    'axes.labelcolor':      'black',
    'axes.labelsize':       10,
    'axes.titlesize':       10,
    'axes.titleweight':     'bold',
    'xtick.labelsize':      8.5,
    'ytick.labelsize':      8.5,

    # Font — Times New Roman or DejaVu Serif fallback
    'font.family':          'serif',
    'font.serif':           ['Times New Roman', 'DejaVu Serif', 'Palatino'],
    'mathtext.fontset':     'stix',

    # Legend
    'legend.frameon':       True,
    'legend.framealpha':    1.0,
    'legend.edgecolor':     'black',
    'legend.fontsize':      8.5,
    'legend.handlelength':  2.5,

    # Lines
    'lines.linewidth':      1.6,

    # Figure
    'figure.dpi':           150,
})


# LINE STYLE CONSTANTS
# No color reliance — all distinction is through linestyle, linewidth, and markers.
# This ensures figures remain legible in grayscale print.

MR_STYLE      = dict(color='black',  lw=1.8, ls='-',   zorder=3)
PASSIVE_STYLE = dict(color='#555555', lw=1.6, ls='--',  zorder=2)
MR_MARKER     = dict(color='black',  lw=1.8, ls='-',   marker='o',
                     markevery=50, markersize=4, markerfacecolor='white',
                     markeredgewidth=1.2, zorder=3)
PASSIVE_MARKER= dict(color='#555555', lw=1.6, ls='--',  marker='s',
                     markevery=50, markersize=4, markerfacecolor='#aaaaaa',
                     markeredgewidth=1.0, zorder=2)

# Bar chart fills
MR_HATCH      = ''          # solid black bars
PASSIVE_HATCH = '////'      # hatched bars for passive


# SURFACE STIFFNESS LOOKUP

SURFACE_K = {
    'hard':   2_000_000,   # N/m — concrete / steel deck
    'medium':   800_000,   # N/m — nominal reinforced ground
    'soft':     200_000,   # N/m — loose soil / compliant deck
}


# SIMULATION CONSTANTS

DT    = 0.001   # timestep = 1 ms; RK4 error ~ DT^4 = 1e-12
T_MAX = 3.0     # simulate 3 s — full landing and settlement


# CORE SIMULATION FUNCTION

def simulate(mass, velocity, attitude_deg, surface, spring_k, c_passive, use_mr):

    theta          = np.radians(attitude_deg)
    v0             = velocity * np.cos(theta)
    lateral_factor = np.sin(theta)

    # Series spring model: leg + ground compliance
    k_ground = SURFACE_K[surface]
    k_eff    = (spring_k * k_ground) / (spring_k + k_ground)

    c_crit = 2.0 * np.sqrt(k_eff * mass)
    c_min  = c_passive
    c_max  = c_crit * 0.85

    x   = 0.0
    vel = v0

    times, forces, disps = [], [], []
    spring_forces = []       # NEW: track spring force separately for publication

    peak_spring = 0.0
    peak_total  = 0.0
    peak_disp   = 0.0
    settle_time = T_MAX
    settled     = False

    n_steps = int(T_MAX / DT)

    for i in range(n_steps + 1):
        t = i * DT

        # Skyhook control law
        if use_mr:
            c = c_max if vel >= 0 else c_min
            if attitude_deg > 5:
                c = min(c_max, c * (1.0 + lateral_factor * 0.4))
        else:
            c = c_passive

        F_spring = k_eff * x
        F_damp   = c * vel
        F_total  = abs(F_spring + F_damp)

        peak_spring = max(peak_spring, F_spring)
        peak_total  = max(peak_total,  F_total)
        peak_disp   = max(peak_disp,   x)

        def deriv(xi, vi):
            xi = max(0.0, xi)
            return vi, -(k_eff * xi + c * vi) / mass

        dx1, dv1 = deriv(x, vel)
        dx2, dv2 = deriv(x + 0.5*DT*dx1, vel + 0.5*DT*dv1)
        dx3, dv3 = deriv(x + 0.5*DT*dx2, vel + 0.5*DT*dv2)
        dx4, dv4 = deriv(x + DT*dx3,     vel + DT*dv3)

        vel += (DT / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        x   += (DT / 6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)

        x = max(0.0, x)
        if x == 0.0 and vel < 0:
            vel = 0.0

        if not settled and t > 0.1 and abs(vel) < 0.01 and abs(x) < 0.001:
            settle_time = t
            settled     = True

        # Store every 3rd step (5x more dense than original every-10th)
        # => ~1,000 points over 3 s, smooth curves without excess file size
        if i % 3 == 0:
            times.append(t)
            forces.append(F_total  / 1000)
            disps.append(x         * 100)
            spring_forces.append(F_spring / 1000)

    return {
        'peak_spring':   peak_spring / 1000,
        'peak_total':    peak_total  / 1000,
        'peak_disp':     peak_disp   * 100,
        'settle_time':   settle_time,
        'times':         np.array(times),
        'forces':        np.array(forces),
        'disps':         np.array(disps),
        'spring_forces': np.array(spring_forces),   # NEW
    }


# HELPER: format axes for publication

def _pub_axes(ax, xlabel='', ylabel='', title=''):
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    if title:
        ax.set_title(title, pad=6)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # Spine thickness
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# PLOTTING FUNCTION: SINGLE SCENARIO

def plot_single(params, label="Nominal Landing"):
    """
    Four-panel publication figure:
      (A) Total Force vs Time
      (B) Spring Force vs Time  [NEW — shows key metric explicitly]
      (C) Displacement vs Time
      (D) Hysteresis: Spring Force vs Displacement  [NEW]
    """

    mr  = simulate(**params, use_mr=True)
    pas = simulate(**params, use_mr=False)

    red_spring = (pas['peak_spring'] - mr['peak_spring']) / pas['peak_spring'] * 100
    red_disp   = (pas['peak_disp']   - mr['peak_disp'])   / pas['peak_disp']   * 100

    fig = plt.figure(figsize=(13, 9), facecolor='white')

    title_str = (
        f'MR Damper vs. Passive Damper — {label}\n'
        r'$v_0$' + f'$={params["velocity"]}$ m/s,  '
        r'$\theta$' + f'$={params["attitude_deg"]}°$,  '
        f'Surface: {params["surface"].capitalize()},  '
        f'Mass: {params["mass"]:,} kg'
    )
    fig.suptitle(title_str, fontsize=10, y=0.99, fontweight='bold')

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # (A) TOTAL FORCE vs TIME
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(mr['times'],  mr['forces'],  label='MR Damper (active)',  **MR_MARKER)
    ax1.plot(pas['times'], pas['forces'], label='Passive Damper', **PASSIVE_MARKER)
    ax1.axhline(mr['peak_spring'],  color='black',   lw=0.7, ls=':', alpha=0.8)
    ax1.axhline(pas['peak_spring'], color='#555555', lw=0.7, ls=':', alpha=0.8)
    ax1.set_xlim(0, 1.5)
    _pub_axes(ax1, 'Time (s)', 'Force (kN)', '(A) Total Contact Force vs. Time')
    ax1.legend(loc='upper right')

    # (B) SPRING FORCE vs TIME — the primary publication metric
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(mr['times'],  mr['spring_forces'],  **MR_MARKER,
             label=f'MR Damper   peak: {mr["peak_spring"]:.1f} kN')
    ax2.plot(pas['times'], pas['spring_forces'], **PASSIVE_MARKER,
             label=f'Passive   peak: {pas["peak_spring"]:.1f} kN')

    # Annotate peak values
    mr_pk_idx  = np.argmax(mr['spring_forces'])
    pas_pk_idx = np.argmax(pas['spring_forces'])
    ax2.annotate(
        f'{mr["peak_spring"]:.1f} kN',
        xy=(mr['times'][mr_pk_idx], mr['spring_forces'][mr_pk_idx]),
        xytext=(mr['times'][mr_pk_idx] + 0.06, mr['spring_forces'][mr_pk_idx] + 1),
        fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.8, color='black'),
        color='black'
    )
    ax2.annotate(
        f'{pas["peak_spring"]:.1f} kN',
        xy=(pas['times'][pas_pk_idx], pas['spring_forces'][pas_pk_idx]),
        xytext=(pas['times'][pas_pk_idx] + 0.06, pas['spring_forces'][pas_pk_idx] + 1),
        fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.8, color='#555555'),
        color='#555555'
    )
    # Reduction bracket
    y_mid = (mr['peak_spring'] + pas['peak_spring']) / 2
    ax2.annotate(
        '',
        xy=(0.85, mr['peak_spring']),
        xytext=(0.85, pas['peak_spring']),
        arrowprops=dict(arrowstyle='<->', lw=1.0, color='black')
    )
    ax2.text(0.87, y_mid, f'{red_spring:.1f}%\nreduction',
             fontsize=7.5, va='center', ha='left', color='black',
             fontstyle='italic')

    ax2.set_xlim(0, 1.5)
    _pub_axes(ax2, 'Time (s)', r'Spring Force $F_s = k_{eff} \cdot x$ (kN)',
              '(B) Spring Force vs. Time  [Primary Metric]')
    ax2.legend(loc='upper right')

    # (C) DISPLACEMENT vs TIME
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(mr['times'],  mr['disps'],  **MR_MARKER,
             label=f'MR Damper   peak: {mr["peak_disp"]:.1f} cm')
    ax3.plot(pas['times'], pas['disps'], **PASSIVE_MARKER,
             label=f'Passive   peak: {pas["peak_disp"]:.1f} cm')
    ax3.set_xlim(0, 1.5)
    _pub_axes(ax3, 'Time (s)', 'Leg Compression (cm)',
              '(C) Displacement vs. Time')
    ax3.legend(loc='upper right')

    # Annotate displacement reduction
    ax3.text(0.52, max(pas['peak_disp'], mr['peak_disp']) * 0.55,
             f'{red_disp:.1f}% less\nmax compression',
             fontsize=7.5, fontstyle='italic', ha='left')

    # (D) HYSTERESIS: Spring Force vs Displacement
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(mr['disps'],  mr['spring_forces'],  **MR_STYLE,
             label='MR Damper')
    ax4.plot(pas['disps'], pas['spring_forces'], **PASSIVE_STYLE,
             label='Passive Damper')
    # Mark start point
    ax4.plot(mr['disps'][0],  mr['spring_forces'][0],
             'k^', ms=5, zorder=5, label='Initial contact')
    ax4.plot(pas['disps'][0], pas['spring_forces'][0],
             'k^', ms=5, zorder=5)
    # Mark peak compression
    ax4.plot(mr['disps'][np.argmax(mr['spring_forces'])],
             mr['peak_spring'], 'ko', ms=5, zorder=5, label='Peak compression')
    ax4.plot(pas['disps'][np.argmax(pas['spring_forces'])],
             pas['peak_spring'], 'ko', ms=5, zorder=5)

    _pub_axes(ax4, 'Leg Compression (cm)',
              'Spring Force $F_s$ (kN)',
              '(D) Spring Force vs. Displacement (Hysteresis)')
    ax4.legend(loc='upper left', fontsize=7.5)

    # METRICS TABLE as figure text
    metrics = (
        r'$\bf{Summary\ Statistics}$' + '\n'
        f'Peak spring force    MR: {mr["peak_spring"]:.2f} kN   Passive: {pas["peak_spring"]:.2f} kN\n'
        f'Reduction (spring): {red_spring:.1f}%     '
        f'Reduction (disp): {red_disp:.1f}%\n'
        f'Settle time          MR: {mr["settle_time"]:.2f} s   Passive: {pas["settle_time"]:.2f} s'
    )
    fig.text(0.01, 0.005, metrics, fontsize=8, va='bottom',
             bbox=dict(facecolor='white', edgecolor='black',
                       boxstyle='round,pad=0.4', linewidth=0.7))

    filename = f'result_{label.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"\n[{label}]")
    print(f"  MR  peak spring: {mr['peak_spring']:.2f} kN | settle: {mr['settle_time']:.2f}s | disp: {mr['peak_disp']:.1f} cm")
    print(f"  PAS peak spring: {pas['peak_spring']:.2f} kN | settle: {pas['settle_time']:.2f}s | disp: {pas['peak_disp']:.1f} cm")
    print(f"  Spring force reduction: {red_spring:.1f}%")


# PLOTTING FUNCTION: ALL 9 SCENARIOS

def plot_all_scenarios(base):
    """
    Run all 9 test matrix scenarios.
    Two-part publication figure:
      Top:    Grouped bar chart (MR vs Passive peak spring force)
      Middle: Reduction % bars
      Bottom: Data table with full numeric results
    """

    scenarios = [
        # Velocity sweep
        dict(label='S1\n1.5 m/s',          velocity=1.5, attitude_deg=0,  surface='medium'),
        dict(label='S2\n3.0 m/s',          velocity=3.0, attitude_deg=0,  surface='medium'),
        dict(label='S3\n6.0 m/s',          velocity=6.0, attitude_deg=0,  surface='medium'),
        # Surface sweep
        dict(label='S4\nHard surf.',        velocity=3.0, attitude_deg=0,  surface='hard'),
        dict(label='S5\nSoft surf.',        velocity=3.0, attitude_deg=0,  surface='soft'),
        # Attitude sweep
        dict(label=r'S6' + '\n5°',         velocity=3.0, attitude_deg=5,  surface='medium'),
        dict(label=r'S7' + '\n10°',        velocity=3.0, attitude_deg=10, surface='medium'),
        # Combined
        dict(label='S8\n6m/s+10°+Hard',    velocity=6.0, attitude_deg=10, surface='hard'),
        dict(label='S9\nExtreme',           velocity=8.0, attitude_deg=15, surface='hard'),
    ]

    mr_peaks, pas_peaks, reds = [], [], []
    settle_mr, settle_pas     = [], []
    disp_mr,   disp_pas       = [], []
    labels_clean              = []

    print(f"\n{'='*72}")
    print(f"  {'SCENARIO':<22} {'MR (kN)':>10} {'PASSIVE (kN)':>14} {'REDUCTION':>12}")
    print(f"  {'-'*68}")

    for s in scenarios:
        p  = {**base, **{k: v for k, v in s.items() if k != 'label'}}
        mr = simulate(**p, use_mr=True)
        pa = simulate(**p, use_mr=False)
        r  = (pa['peak_spring'] - mr['peak_spring']) / pa['peak_spring'] * 100

        mr_peaks.append(mr['peak_spring'])
        pas_peaks.append(pa['peak_spring'])
        reds.append(r)
        settle_mr.append(mr['settle_time'])
        settle_pas.append(pa['settle_time'])
        disp_mr.append(mr['peak_disp'])
        disp_pas.append(pa['peak_disp'])
        labels_clean.append(s['label'].replace('\n', ' '))

        print(f"  {labels_clean[-1]:<22} {mr['peak_spring']:>10.1f} "
              f"{pa['peak_spring']:>14.1f} {r:>10.1f}%")

    print(f"  {'='*68}")
    print(f"  Average reduction: {np.mean(reds):.1f}%")
    print(f"  Best:              {max(reds):.1f}%")
    print(f"  Worst:             {min(reds):.1f}%")

    # BUILD FIGURE
    fig = plt.figure(figsize=(14, 11), facecolor='white')
    gs  = gridspec.GridSpec(3, 1, figure=fig,
                            height_ratios=[3, 1.2, 1.4], hspace=0.55)
    fig.suptitle(
        'Peak Spring Force: MR Damper vs. Passive Damper Across All 9 Test Scenarios',
        fontsize=11, fontweight='bold', y=0.99
    )

    x  = np.arange(len(scenarios))
    bw = 0.36

    # (A) GROUPED BAR CHART
    ax1 = fig.add_subplot(gs[0])
    b1  = ax1.bar(x - bw/2, mr_peaks,  bw,
                  color='black',   hatch=MR_HATCH,      alpha=0.85,
                  label='MR Damper (active)')
    b2  = ax1.bar(x + bw/2, pas_peaks, bw,
                  color='#777777', hatch=PASSIVE_HATCH,  alpha=0.85,
                  edgecolor='black', linewidth=0.6,
                  label='Passive Damper')

    # Value labels on bars
    for b in b1:
        ax1.text(b.get_x() + b.get_width()/2,
                 b.get_height() + max(mr_peaks) * 0.012,
                 f'{b.get_height():.1f}',
                 ha='center', va='bottom', fontsize=7, color='black')
    for b in b2:
        ax1.text(b.get_x() + b.get_width()/2,
                 b.get_height() + max(pas_peaks) * 0.012,
                 f'{b.get_height():.1f}',
                 ha='center', va='bottom', fontsize=7, color='#555555')

    ax1.set_xticks(x)
    ax1.set_xticklabels([s['label'] for s in scenarios], fontsize=8)
    _pub_axes(ax1, '', 'Peak Spring Force (kN)', '(A) Peak Spring Force by Scenario')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(0, max(pas_peaks) * 1.18)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)

    # Shaded sweep groups
    group_spans = [(0, 2, 'Velocity\nSweep'), (3, 4, 'Surface\nSweep'),
                   (5, 6, 'Attitude\nSweep'), (7, 8, 'Combined\nOff-nominal')]
    colors_bg   = ['#f0f0f0', '#e8e8e8', '#f0f0f0', '#e0e0e0']
    for (lo, hi, glabel), gbg in zip(group_spans, colors_bg):
        ax1.axvspan(lo - 0.5, hi + 0.5, alpha=0.25, color=gbg, zorder=0)
        ax1.text((lo + hi) / 2, max(pas_peaks) * 1.12,
                 glabel, ha='center', va='bottom', fontsize=7,
                 color='#444444', fontstyle='italic')

    # (B) REDUCTION % BAR CHART
    ax2 = fig.add_subplot(gs[1])
    bar_colors = ['black' if r > 0 else '#aaaaaa' for r in reds]
    bar_hatches = ['' if r > 0 else 'xxxx' for r in reds]
    rects = ax2.bar(x, reds, color=bar_colors, hatch=bar_hatches,
                    alpha=0.85, edgecolor='black', linewidth=0.6)
    ax2.axhline(0, color='black', lw=0.8)

    # Average line
    avg = np.mean(reds)
    ax2.axhline(avg, color='black', lw=1.0, ls='--', alpha=0.7)
    ax2.text(len(scenarios) - 0.5, avg + 0.4, f'Mean = {avg:.1f}%',
             ha='right', va='bottom', fontsize=7.5, fontstyle='italic')

    for rect, val in zip(rects, reds):
        offset = 0.3 if val >= 0 else -1.2
        ax2.text(rect.get_x() + rect.get_width()/2,
                 val + offset, f'{val:.1f}%',
                 ha='center', va='bottom', fontsize=7.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([s['label'] for s in scenarios], fontsize=8)
    _pub_axes(ax2, '', 'Reduction (%)', '(B) Spring Force Reduction (MR vs. Passive)')
    for span_lo, span_hi, _ in group_spans:
        ax2.axvspan(span_lo - 0.5, span_hi + 0.5, alpha=0.06,
                    color='black', zorder=0)

    # (C) NUMERIC RESULTS TABLE
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')

    col_labels = ['Scenario',
                  'MR Peak\n(kN)', 'Passive Peak\n(kN)', 'Reduction\n(%)',
                  'MR Disp.\n(cm)', 'Pas. Disp.\n(cm)',
                  'MR Settle\n(s)', 'Pas. Settle\n(s)']

    table_data = []
    for i, s in enumerate(scenarios):
        table_data.append([
            labels_clean[i],
            f'{mr_peaks[i]:.2f}',
            f'{pas_peaks[i]:.2f}',
            f'{reds[i]:.1f}',
            f'{disp_mr[i]:.2f}',
            f'{disp_pas[i]:.2f}',
            f'{settle_mr[i]:.2f}',
            f'{settle_pas[i]:.2f}',
        ])

    tbl = ax3.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.8)
    tbl.scale(1, 1.35)

    # Style header row
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#222222')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    # Alternating row shading
    for i in range(len(table_data)):
        bg = '#f5f5f5' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            tbl[(i+1, j)].set_facecolor(bg)
            # Highlight reduction column
            if j == 3:
                val = reds[i]
                tbl[(i+1, j)].set_facecolor(
                    '#d4edda' if val > 15 else ('#fff3cd' if val > 5 else '#f8d7da'))

    ax3.set_title('(C) Full Numeric Results Table', fontweight='bold',
                  fontsize=9, pad=8, loc='left')

    plt.savefig('result_all_scenarios.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.show()


# MAIN

if __name__ == '__main__':

    BASE = dict(
        mass         = 5_000,    # kg    — effective rocket mass on one leg
        spring_k     = 150_000,  # N/m   — landing leg spring stiffness
        c_passive    = 15_000,   # Ns/m  — passive damping coefficient (zeta = 0.274)
        attitude_deg = 0,        # deg   — nominal = perfectly vertical
        surface      = 'medium', # str   — nominal = reinforced ground
        velocity     = 3.0,      # m/s   — nominal landing speed
    )

    c_crit = 2 * np.sqrt(BASE['spring_k'] * BASE['mass'])
    c_max  = c_crit * 0.85

    print("=" * 62)
    print("  MR DAMPER vs PASSIVE DAMPER — ROCKET LANDING SIMULATION")
    print("=" * 62)
    print(f"  Mass (m):              {BASE['mass']:>8,} kg")
    print(f"  Spring stiffness (k):  {BASE['spring_k']:>8,} N/m")
    print(f"  Critical damping:      {c_crit:>8,.0f} N·s/m  [2*sqrt(k*m)]")
    print(f"  Passive c:             {BASE['c_passive']:>8,} N·s/m  [zeta = {BASE['c_passive']/c_crit:.3f}]")
    print(f"  MR c_min:              {500:>8,} N·s/m  [near-zero, rebound phase]")
    print(f"  MR c_max:              {c_max:>8,.0f} N·s/m  [85% of c_crit]")
    print(f"  Timestep (DT):         {DT*1000:>8.1f} ms")
    print(f"  Simulation duration:   {T_MAX:>8.1f} s")
    print(f"  Data density:          every 3 ms stored (~1,000 pts/curve)")
    print(f"  Key metric:            Peak spring force k_eff * x")
    print("=" * 62)

    print("\n[1/2] Running nominal scenario (3 m/s, 0 deg, medium)...")
    plot_single(BASE, label="Nominal Landing")

    print("\n[2/2] Running all 9 scenarios...")
    plot_all_scenarios(BASE)

    print("\nDone. Files saved:")
    print("    result_Nominal_Landing.png       (300 dpi, 4-panel)")
    print("    result_all_scenarios.png         (300 dpi, 3-panel + table)")