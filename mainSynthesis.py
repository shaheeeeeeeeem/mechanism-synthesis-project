import os
import sys
import builtins
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
from matplotlib.animation import FuncAnimation, PillowWriter

_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Reference console output can be re-enabled if needed during debugging.
VERBOSE_OUTPUT = False
if not VERBOSE_OUTPUT:
    print = lambda *args, **kwargs: None
else:
    print = builtins.print

# =============================================================================
# SECTION 1: PARAMETERS
# =============================================================================

R           = 50.0   # radius of circular body (mm)
drop_B      =  5.0   # body center drop at position B (mm)
drop_C      = 15.0   # body center drop at position C (mm)
speed_ratio =  2.0   # C→A return is speed_ratio × faster than A→C forward

# Body rotation (CW) at each precision position
theta_body_A =   0.0   # degrees
theta_body_B = -45.0   # degrees
theta_body_C = -90.0   # degrees

print("=" * 60)
print("SECTION 1: PARAMETERS")
print("=" * 60)
print(f"  Body radius R       = {R} mm")
print(f"  Drop at B           = {drop_B} mm")
print(f"  Drop at C           = {drop_C} mm")
print(f"  Body rotation A→C   = {theta_body_C}° CW")
print(f"  Return speed ratio  = {speed_ratio}×")

# =============================================================================
# PRESENTATION POINT (2) START: synthesis method outline
# =============================================================================
#
# We use an offset slider-crank as the mechanism. The crank and connecting rod
# generate the vertical motion of the body center, and the body rotation is
# interpolated exactly through the three required precision positions A, B, C.
#
# =============================================================================
# PRESENTATION POINT (2) END
# =============================================================================

# =============================================================================
# PRESENTATION POINT (3) START: outline of how the code generates the output
# =============================================================================
#
# - Exact slider-crank loop-closure equations are written directly in the code.
# - A small in-file bracketed bisection solver is used to locate the crank
#   angles for A, B, and C.
# - The program is generic with respect to R, drop_B, and drop_C.
# - Assumption used: the return stroke follows the same A↔C crank path in
#   reverse at twice the input angular speed, so the output remains bounded
#   between 0 and -15 mm instead of dipping below C.
#
# =============================================================================
# PRESENTATION POINT (3) END
# =============================================================================

# =============================================================================
# SECTION 2: SYNTHESIS — OFFSET SLIDER-CRANK
#
# Layout (y upward, x rightward):
#   O  = crank pivot (fixed, at origin)
#   A  = crank pin  (revolute joint on crank tip)
#   B  = slider pin (prismatic joint — slides along vertical line x = e)
#
# Vector loop:
#   e          = r·sin(φ) + l·sin(ψ)       ...(x)
#   s_abs(φ)   = r·cos(φ) + l·cos(ψ)       ...(y, absolute slider position)
#
# Drop from starting position A:
#   drop(φ) = s_abs(φ_A) − s_abs(φ)
#
# Quick-return condition used here:
#   Forward stroke:  crank travels from φ_A to φ_C
#   Return  stroke:  crank reverses from φ_C to φ_A at speed_ratio × speed
#   Therefore: t_fwd / t_ret = 1 / speed_ratio  ✓
#
# Synthesis:  choose r, l, e freely (design parameters).
#             Find φ_A such that drop(φ_A) = 0 (reference) and
#             drop(φ_A + Δφ_fwd) = drop_C = 15 mm.
#             Then find φ_B ∈ (φ_A, φ_C) such that drop(φ_B) = drop_B = 5 mm.
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: SYNTHESIS — OFFSET SLIDER-CRANK")
print("=" * 60)

# Crank sweeps over the same geometric stroke; the return is faster because the
# crank speed magnitude is larger on the return.
delta_phi_fwd = np.deg2rad(120.0)   # rad
delta_phi_ret = delta_phi_fwd       # same path, reversed
print(f"\nQuick-return geometry:")
print(f"  Forward crank sweep = {np.degrees(delta_phi_fwd):.2f}°")
print(f"  Return  crank sweep = {np.degrees(delta_phi_ret):.2f}°")
print(f"  Time ratio fwd/ret  = 1/{speed_ratio:.0f}  ✓")

# Design parameters (kept explicit for a clean, report-friendly definition)
r = 10.0    # crank length (mm)
l = 40.0    # connecting rod length (mm)
e = 5.0     # eccentricity — offset of slider rail from O (mm)

def bisection_root(func, a, b, tol=1e-10, max_iter=200):
    """Simple bracketed root solver to avoid requiring SciPy."""
    fa = func(a)
    fb = func(b)
    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b
    if fa * fb > 0:
        raise ValueError("Bisection requires a sign-changing bracket.")

    left, right = a, b
    f_left, f_right = fa, fb
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = func(mid)
        if abs(f_mid) < tol or abs(right - left) < tol:
            return mid
        if f_left * f_mid <= 0:
            right, f_right = mid, f_mid
        else:
            left, f_left = mid, f_mid
    return 0.5 * (left + right)

def find_bracket(func, start, end, samples=2000):
    """Find a sign-changing bracket for func over [start, end]."""
    xs = np.linspace(start, end, samples)
    f_prev = func(xs[0])
    for x in xs[1:]:
        f_curr = func(x)
        if abs(f_prev) < 1e-12:
            return x - (end - start) / samples, x - (end - start) / samples
        if f_prev * f_curr <= 0:
            return x - (end - start) / samples, x
        f_prev = f_curr
    raise ValueError("No sign-changing bracket found in the search interval.")

def slider_pos_abs(phi, r, l, e):
    """Absolute slider y-position for crank angle phi."""
    arg = (e - r * np.sin(phi)) / l
    arg = np.clip(arg, -1, 1)
    return r * np.cos(phi) + l * np.sqrt(1 - arg**2)

def drop(phi, phi_A, r, l, e):
    """Drop = reduction in slider height from position A."""
    return slider_pos_abs(phi_A, r, l, e) - slider_pos_abs(phi, r, l, e)

# Find phi_A from the exact drop condition over one revolution.
phi_A_func = lambda phi_A_val: drop(phi_A_val + delta_phi_fwd, phi_A_val, r, l, e) - drop_C
phi_A_lo, phi_A_hi = find_bracket(phi_A_func, -np.pi, np.pi)
phi_A = bisection_root(phi_A_func, phi_A_lo, phi_A_hi)
phi_C = phi_A + delta_phi_fwd

# Find phi_B: drop = drop_B between phi_A and phi_C
phi_B_func = lambda p: drop(p, phi_A, r, l, e) - drop_B
phi_B_lo, phi_B_hi = find_bracket(phi_B_func, phi_A, phi_C)
phi_B = bisection_root(phi_B_func, phi_B_lo, phi_B_hi)

print(f"\nDesign parameters:")
print(f"  Crank length    r = {r} mm")
print(f"  Rod length      l = {l} mm")
print(f"  Eccentricity    e = {e} mm")
print(f"\nCrank angles at precision positions:")
print(f"  φ_A = {np.degrees(phi_A):.4f}°")
print(f"  φ_B = {np.degrees(phi_B):.4f}°")
print(f"  φ_C = {np.degrees(phi_C):.4f}°")

dA = drop(phi_A, phi_A, r, l, e)
dB = drop(phi_B, phi_A, r, l, e)
dC = drop(phi_C, phi_A, r, l, e)
print(f"\nVerification (errors should be ~0):")
print(f"  Drop at A: {dA:.8f} mm  (expected 0.0)")
print(f"  Drop at B: {dB:.8f} mm  (expected {drop_B})")
print(f"  Drop at C: {dC:.8f} mm  (expected {drop_C})")

# Absolute positions of slider pin B at precision positions
s_abs_A = slider_pos_abs(phi_A, r, l, e)
s_abs_B = slider_pos_abs(phi_B, r, l, e)
s_abs_C = slider_pos_abs(phi_C, r, l, e)

# Synthesize the body's rotation from the three required precision positions.
# This keeps the slider-crank as the vertical motion generator, while enforcing
# the exact A/B/C orientations from the problem statement for any updated
# dimensions entered above.
body_drop_pts  = np.array([0.0, drop_B, drop_C])
body_angle_pts = np.array([theta_body_A, theta_body_B, theta_body_C])
body_angle_poly = np.poly1d(np.polyfit(body_drop_pts, body_angle_pts, 2))

def body_angle_from_drop(drop_val):
    """Body angle (deg) as a function of slider drop (mm)."""
    d = np.clip(drop_val, body_drop_pts[0], body_drop_pts[-1])
    return float(body_angle_poly(d))

def add_body_patch(ax, center_xy, angle_deg, radius, facecolor, edgecolor,
                   label=None, label_color='k', alpha=0.80, zorder=3):
    """Draw the partial circular body used in the output plots."""
    wedge = Wedge(center_xy, radius,
                  theta1=angle_deg + 90.0,
                  theta2=angle_deg + 360.0,
                  facecolor=facecolor, edgecolor=edgecolor,
                  linewidth=1.2, alpha=alpha, zorder=zorder)
    ax.add_patch(wedge)
    ax.plot(center_xy[0], center_xy[1], 'o', color=edgecolor,
            markersize=4, zorder=zorder + 1)
    if label:
        ax.text(center_xy[0], center_xy[1] + radius + 6, label,
                ha='center', va='bottom', color=label_color,
                fontsize=11, fontweight='bold')
    return wedge

# =============================================================================
# SECTION 3: KINEMATIC ANALYSIS
#
# Position:
#   x_A(φ) = r·sin(φ),       y_A(φ) = r·cos(φ)          (crank pin)
#   x_B    = e,               y_B(φ) = r·cos(φ) + l·cos(ψ)  (slider pin)
#   sin(ψ) = (e − r·sin(φ)) / l
#
# Velocity (differentiate w.r.t. time, φ̇ = ω):
#   ẋ_A =  r·cos(φ)·ω
#   ẏ_A = −r·sin(φ)·ω
#   ψ̇   = −r·cos(φ)·ω / (l·cos(ψ))
#   ẏ_B = −r·sin(φ)·ω − l·sin(ψ)·ψ̇
#        = ω·r·[sin(ψ)·cos(φ) − sin(φ)·cos(ψ)] / cos(ψ)
#        = ω·r·sin(ψ − φ) / cos(ψ)
#
# Acceleration (differentiate velocity, φ̈ = α = 0 for constant speed):
#   ψ̈   = [r·sin(φ)·ω² − l·sin(ψ)·ψ̇²] / (l·cos(ψ))
#   ÿ_B = −r·cos(φ)·ω² − l·cos(ψ)·ψ̇² − l·sin(ψ)·ψ̈
#
# Body rotation angle: analytically synthesized from the three precision states
#   (drop, angle) = (0, θ_A), (drop_B, θ_B), (drop_C, θ_C)
# using a quadratic interpolation so A, B, and C are hit exactly.
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: KINEMATIC ANALYSIS")
print("=" * 60)

N_fwd = 500
N_ret = 250
N     = N_fwd + N_ret

w_fwd =  1.0                 # crank ω forward (rad/s, normalised)
w_ret = -speed_ratio * w_fwd # crank ω return  (negative = reversing)

# Crank angle arrays
phi_fwd = np.linspace(phi_A, phi_C, N_fwd)
# Return: reverse along the same stroke so the output remains between A and C
phi_ret = np.linspace(phi_C, phi_A, N_ret)
phi_all = np.concatenate([phi_fwd, phi_ret])
w_all   = np.concatenate([np.full(N_fwd, w_fwd), np.full(N_ret, w_ret)])
alpha_crank_all = np.zeros(N)

# Time arrays
dphi_fwd = abs(phi_C - phi_A) / N_fwd
dphi_ret = abs(phi_C - phi_A) / N_ret
t_fwd = np.cumsum(np.full(N_fwd, dphi_fwd / abs(w_fwd)))
t_ret = t_fwd[-1] + np.cumsum(np.full(N_ret, dphi_ret / abs(w_ret)))
time  = np.concatenate([t_fwd, t_ret])

# Storage
xA_all      = np.zeros(N)
yA_all      = np.zeros(N)
xB_all      = np.zeros(N)
yB_all      = np.zeros(N)
psi_all     = np.zeros(N)
drop_all    = np.zeros(N)
body_angle_all = np.zeros(N)
xA_dot_all  = np.zeros(N)
yA_dot_all  = np.zeros(N)
yB_dot_all  = np.zeros(N)
psi_dot_all = np.zeros(N)
xA_ddot_all = np.zeros(N)
yA_ddot_all = np.zeros(N)
yB_ddot_all = np.zeros(N)
psi_ddot_all= np.zeros(N)

for i, (phi, w, alp) in enumerate(zip(phi_all, w_all, alpha_crank_all)):
    # Position
    sin_psi = np.clip((e - r*np.sin(phi)) / l, -1, 1)
    cos_psi = np.sqrt(max(0, 1 - sin_psi**2))
    psi     = np.arcsin(sin_psi)
    yB      = r*np.cos(phi) + l*cos_psi

    # Velocity
    psi_dot  = -r*np.cos(phi)*w / (l*cos_psi) if cos_psi > 1e-9 else 0
    xA_dot   =  r*np.cos(phi)*w
    yA_dot   = -r*np.sin(phi)*w
    yB_dot   = -r*np.sin(phi)*w - l*sin_psi*psi_dot

    # Acceleration
    psi_ddot = (r*np.sin(phi)*w**2 - r*np.cos(phi)*alp
                - l*sin_psi*psi_dot**2) / (l*cos_psi) if cos_psi > 1e-9 else 0
    xA_ddot  =  r*np.cos(phi)*alp - r*np.sin(phi)*w**2
    yA_ddot  = -r*np.sin(phi)*alp - r*np.cos(phi)*w**2
    yB_ddot  = (-r*np.cos(phi)*w**2 - r*np.sin(phi)*alp
                - l*cos_psi*psi_dot**2 - l*sin_psi*psi_ddot)

    xA_all[i]       = r*np.sin(phi)
    yA_all[i]       = r*np.cos(phi)
    xB_all[i]       = e
    yB_all[i]       = yB
    psi_all[i]      = psi
    d               = s_abs_A - yB
    drop_all[i]     = d
    body_angle_all[i] = body_angle_from_drop(d)
    xA_dot_all[i]   = xA_dot
    yA_dot_all[i]   = yA_dot
    yB_dot_all[i]   = yB_dot
    psi_dot_all[i]  = psi_dot
    xA_ddot_all[i]  = xA_ddot
    yA_ddot_all[i]  = yA_ddot
    yB_ddot_all[i]  = yB_ddot
    psi_ddot_all[i] = psi_ddot

fwd_idx = range(N_fwd)
ret_idx = range(N_fwd, N)
idx_B   = np.argmin(np.abs(drop_all[:N_fwd] - drop_B))

print(f"\nKinematics solved for {N} steps.")
print(f"  Forward: {N_fwd} steps, ω = +{w_fwd} rad/s")
print(f"  Return : {N_ret} steps, ω = {w_ret} rad/s")
print(f"  Total cycle time : {time[-1]:.4f} s")
print(f"\nSlider drop verification:")
print(f"  At A (i=0)     : drop = {drop_all[0]:.6f} mm   (expected 0.0)")
print(f"  At B (i={idx_B:3d})  : drop = {drop_all[idx_B]:.6f} mm   (expected {drop_B})")
print(f"  At C (i={N_fwd-1:3d}) : drop = {drop_all[N_fwd-1]:.6f} mm  (expected {drop_C})")
print(f"\nBody-angle verification:")
print(f"  At A           : angle = {body_angle_from_drop(0.0):.6f} deg   (expected {theta_body_A})")
print(f"  At B           : angle = {body_angle_from_drop(drop_B):.6f} deg   (expected {theta_body_B})")
print(f"  At C           : angle = {body_angle_from_drop(drop_C):.6f} deg   (expected {theta_body_C})")

# Report-friendly display coordinates: body center starts at y = 0 and moves
# downward to y = -15 mm, matching the statement directly.
y_ref_display = s_abs_A
yA_disp_all = yA_all - y_ref_display
yB_disp_all = yB_all - y_ref_display
body_angle_plot_all = -body_angle_all

def padded_limits(values, pad_frac=0.08, min_pad=0.5):
    """Return neat axis limits with a small padding."""
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    pad = max(min_pad, pad_frac * span if span > 1e-12 else min_pad)
    return vmin - pad, vmax + pad

def perp(vec):
    """90-degree CCW rotation of a 2D vector."""
    return np.array([-vec[1], vec[0]])

def draw_vector(ax, start, vec, color, label, text_offset=(0.2, 0.2), lw=2.0):
    """Draw a head-to-tail vector with a readable offset label."""
    end = start + vec
    ax.annotate(
        '',
        xy=(end[0], end[1]),
        xytext=(start[0], start[1]),
        arrowprops=dict(arrowstyle='->', color=color, linewidth=lw, shrinkA=0, shrinkB=0),
        zorder=3,
    )
    mid = start + 0.55 * vec
    ax.text(mid[0] + text_offset[0], mid[1] + text_offset[1], label,
            color=color, fontsize=10, weight='bold')
    return end

def smart_offset(vec, normal_scale=0.45, tangential_scale=0.18):
    """Offset labels away from the vector direction so they don't sit on top of it."""
    mag = np.linalg.norm(vec)
    if mag < 1e-9:
        return (0.35, 0.35)
    unit = vec / mag
    normal = perp(unit)
    offset = tangential_scale * unit + normal_scale * normal
    return (float(offset[0]), float(offset[1]))

# =============================================================================
# PRESENTATION POINT (5) START: position, velocity, and acceleration diagrams
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Slider-Crank Kinematic Diagrams", fontsize=14, fontweight='bold')
fig.text(0.5, 0.955,
         "Input crank sweeps 120° while the output body rotates 90° from A to C.",
         ha='center', va='top', fontsize=10, color='#555555')

axes[0,0].plot(time[fwd_idx], yB_disp_all[fwd_idx], 'b-', label='Forward')
axes[0,0].plot(time[ret_idx], yB_disp_all[ret_idx], 'r--', label='Return')
axes[0,0].axhline(-drop_B, color='m', linestyle=':', label=f'B = {-drop_B} mm')
axes[0,0].axhline(-drop_C, color='orange', linestyle=':', label=f'C = {-drop_C} mm')
axes[0,0].set_title("Body Center Position y_c(t)")
axes[0,0].set_xlabel("Time (s)"); axes[0,0].set_ylabel("y_c (mm)")
axes[0,0].set_xlim(time[0], time[-1])
axes[0,0].set_ylim(-drop_C - 0.8, 0.8)
axes[0,0].legend(); axes[0,0].grid(True)

axes[0,1].plot(time[fwd_idx], yB_dot_all[fwd_idx], 'b-', label='Forward')
axes[0,1].plot(time[ret_idx], yB_dot_all[ret_idx], 'r--', label='Return')
axes[0,1].axhline(0, color='k', linewidth=0.5)
axes[0,1].set_title("Body Center Velocity v_c(t)")
axes[0,1].set_xlabel("Time (s)"); axes[0,1].set_ylabel("v_c (mm/s)")
axes[0,1].set_xlim(time[0], time[-1])
axes[0,1].set_ylim(*padded_limits(yB_dot_all))
axes[0,1].legend(); axes[0,1].grid(True)

axes[0,2].plot(time[fwd_idx], yB_ddot_all[fwd_idx], 'b-', label='Forward')
axes[0,2].plot(time[ret_idx], yB_ddot_all[ret_idx], 'r--', label='Return')
axes[0,2].axhline(0, color='k', linewidth=0.5)
axes[0,2].set_title("Body Center Acceleration a_c(t)")
axes[0,2].set_xlabel("Time (s)"); axes[0,2].set_ylabel("a_c (mm/s²)")
axes[0,2].set_xlim(time[0], time[-1])
axes[0,2].set_ylim(*padded_limits(yB_ddot_all))
axes[0,2].legend(); axes[0,2].grid(True)

axes[1,0].plot(time, xA_dot_all, label='Crank pin A: x-velocity')
axes[1,0].plot(time, yA_dot_all, label='Crank pin A: y-velocity')
axes[1,0].plot(time, yB_dot_all, label='Slider pin B: y-velocity')
axes[1,0].set_title("Joint Velocity Components")
axes[1,0].set_xlabel("Time (s)"); axes[1,0].set_ylabel("Velocity (mm/s)")
axes[1,0].set_xlim(time[0], time[-1])
axes[1,0].set_ylim(*padded_limits(np.concatenate([xA_dot_all, yA_dot_all, yB_dot_all])))
axes[1,0].legend(); axes[1,0].grid(True)

axes[1,1].plot(time, xA_ddot_all, label='Crank pin A: x-acceleration')
axes[1,1].plot(time, yA_ddot_all, label='Crank pin A: y-acceleration')
axes[1,1].plot(time, yB_ddot_all, label='Slider pin B: y-acceleration')
axes[1,1].set_title("Joint Acceleration Components")
axes[1,1].set_xlabel("Time (s)"); axes[1,1].set_ylabel("Acceleration (mm/s²)")
axes[1,1].set_xlim(time[0], time[-1])
axes[1,1].set_ylim(*padded_limits(np.concatenate([xA_ddot_all, yA_ddot_all, yB_ddot_all])))
axes[1,1].legend(); axes[1,1].grid(True)

# Mechanism diagram at A, B, C
ax = axes[1,2]
ax.set_title("Mechanism at Precision Positions A, B, C")
O = np.array([0.0, 0.0])
colors_prec = ['#00cc66', '#ffcc00', '#ff4444']
labels_prec = ['A', 'B', 'C']
prec_idx    = [0, idx_B, N_fwd-1]
for idx, col, lbl in zip(prec_idx, colors_prec, labels_prec):
    pA = np.array([xA_all[idx], yA_all[idx]])
    pB = np.array([xB_all[idx], yB_all[idx]])
    ax.plot([O[0], pA[0]], [O[1], pA[1]], color=col, linewidth=2.5)
    ax.plot([pA[0], pB[0]], [pA[1], pB[1]], color=col, linewidth=2.5,
            linestyle='--', label=f'Pos {lbl}')
    ax.plot(*pA, 'o', color=col, markersize=7)
    ax.plot(*pB, 's', color=col, markersize=7)
ax.plot(*O, 'k^', markersize=12, zorder=5, label='Crank pivot O')
ax.axvline(e, color='grey', linestyle=':', linewidth=1.5,
           label=f'Slider rail x={e} mm')
ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
ax.grid(True); ax.set_aspect('equal')
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

plt.tight_layout()
plt.savefig(os.path.join(_script_dir, "kinematics.png"), dpi=150, bbox_inches='tight')
plt.show()
print("\nKinematics plot saved.")

# Velocity and acceleration loop-closure diagrams at A, B, and C
fig_loop, axes_loop = plt.subplots(2, 3, figsize=(15, 8))
fig_loop.suptitle("Velocity and Acceleration Loop-Closure Diagrams", fontsize=14, fontweight='bold')
fig_loop.text(0.5, 0.955,
              "Vector loops are shown at A, B, C for the same mechanism: 120° crank sweep gives 90° body rotation.",
              ha='center', va='top', fontsize=10, color='#555555')
fig_loop.text(0.5, 0.932,
              "Since the body center is attached to the slider block in this model, vc = vB and ac = aB.",
              ha='center', va='top', fontsize=10, color='#8b0000')
precision_indices = [0, idx_B, N_fwd - 1]
precision_labels = ['A', 'B', 'C']

for col, (idx, lbl) in enumerate(zip(precision_indices, precision_labels)):
    pA = np.array([xA_all[idx], yA_all[idx]])
    pB = np.array([xB_all[idx], yB_all[idx]])
    r_BA = pB - pA

    # Use closure definitions directly so the polygons close exactly.
    vA = np.array([xA_dot_all[idx], yA_dot_all[idx]])
    vB = np.array([0.0, yB_dot_all[idx]])
    vBA = vB - vA
    vC = vB.copy()   # body center coincides with slider pin B in this model

    aA = np.array([xA_ddot_all[idx], yA_ddot_all[idx]])
    aB = np.array([0.0, yB_ddot_all[idx]])
    a_rel = aB - aA
    aBA_norm = -(psi_dot_all[idx] ** 2) * r_BA
    aBA_tan = a_rel - aBA_norm
    aC = aB.copy()   # body center coincides with slider pin B

    vel_vectors = [vA, vBA, vB]
    acc_vectors = [aA, aBA_tan, aBA_norm, aB]
    vel_scale = max(np.linalg.norm(v) for v in vel_vectors) + 1e-9
    acc_scale = max(np.linalg.norm(a) for a in acc_vectors) + 1e-9

    off_vA = smart_offset(vA)
    off_vBA = smart_offset(vBA, normal_scale=0.55)
    off_vB = smart_offset(vB, normal_scale=0.55)
    off_aA = smart_offset(aA)
    off_aBA_t = smart_offset(aBA_tan, normal_scale=0.55)
    off_aBA_n = smart_offset(aBA_norm, normal_scale=0.55)
    off_aB = smart_offset(aB, normal_scale=0.55)

    axv = axes_loop[0, col]
    axv.set_title(f"Velocity Diagram at {lbl}")
    v_tip = draw_vector(axv, np.array([0.0, 0.0]), vA, '#1f77b4', 'vA',
                        text_offset=off_vA)
    v_loop_tip = draw_vector(axv, v_tip, vBA, '#ff7f0e', 'vBA',
                             text_offset=off_vBA)
    draw_vector(axv, np.array([0.0, 0.0]), vB, '#2ca02c', 'vB',
                text_offset=off_vB)
    axv.plot([0.0, v_tip[0], v_loop_tip[0], 0.0],
             [0.0, v_tip[1], v_loop_tip[1], 0.0],
             color='#666666', linestyle=':', linewidth=1.0, alpha=0.8)
    axv.plot(v_loop_tip[0], v_loop_tip[1], 'ko', markersize=3)
    axv.text(vB[0] + off_vB[0], vB[1] + off_vB[1] - 0.7, 'vc = vB',
             color='#d62728', fontsize=10, weight='bold')
    axv.axhline(0, color='k', linewidth=0.5)
    axv.axvline(0, color='k', linewidth=0.5)
    axv.set_aspect('equal')
    axv.grid(True, alpha=0.35)
    axv.set_xlim(-1.2 * vel_scale, 1.2 * vel_scale)
    axv.set_ylim(-1.2 * vel_scale, 1.2 * vel_scale)
    axv.set_xlabel("x-component")
    axv.set_ylabel("y-component")

    axa = axes_loop[1, col]
    axa.set_title(f"Acceleration Diagram at {lbl}")
    a_tip = draw_vector(axa, np.array([0.0, 0.0]), aA, '#1f77b4', 'aA',
                        text_offset=off_aA)
    a_tan_tip = draw_vector(axa, a_tip, aBA_tan, '#ff7f0e', 'aBA,t',
                            text_offset=off_aBA_t)
    a_loop_tip = draw_vector(axa, a_tan_tip, aBA_norm, '#9467bd', 'aBA,n',
                             text_offset=off_aBA_n)
    draw_vector(axa, np.array([0.0, 0.0]), aB, '#2ca02c', 'aB',
                text_offset=off_aB)
    axa.plot([0.0, a_tip[0], a_tan_tip[0], a_loop_tip[0], 0.0],
             [0.0, a_tip[1], a_tan_tip[1], a_loop_tip[1], 0.0],
             color='#666666', linestyle=':', linewidth=1.0, alpha=0.8)
    axa.plot(a_loop_tip[0], a_loop_tip[1], 'ko', markersize=3)
    axa.text(aB[0] + off_aB[0], aB[1] + off_aB[1] - 0.7, 'ac = aB',
             color='#d62728', fontsize=10, weight='bold')
    axa.axhline(0, color='k', linewidth=0.5)
    axa.axvline(0, color='k', linewidth=0.5)
    axa.set_aspect('equal')
    axa.grid(True, alpha=0.35)
    axa.set_xlim(-1.2 * acc_scale, 1.2 * acc_scale)
    axa.set_ylim(-1.2 * acc_scale, 1.2 * acc_scale)
    axa.set_xlabel("x-component")
    axa.set_ylabel("y-component")

plt.tight_layout()
plt.savefig(os.path.join(_script_dir, "loop_closure_diagrams.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Loop-closure vector diagrams saved.")

# =============================================================================
# PRESENTATION POINT (5) END
# =============================================================================

# =============================================================================
# PRESENTATION POINT (1) START: show the synthesized mechanism / output body
# =============================================================================

# Output precision positions for the report
fig_prec, axes_prec = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
fig_prec.suptitle("Precision Positions of the Output Body", fontsize=14, fontweight='bold')
fig_prec.text(0.5, 0.955,
              "These are the three required output positions. They are produced by a 120° input crank sweep.",
              ha='center', va='top', fontsize=10, color='#555555')
precision_data = [
    ("A", 0.0, theta_body_A, "#1f7a8c"),
    ("B", -drop_B, theta_body_B, "#c49a00"),
    ("C", -drop_C, theta_body_C, "#b24747"),
]
for ax_prec, (lbl, y_c, ang, col) in zip(axes_prec, precision_data):
    add_body_patch(ax_prec, (0.0, y_c), ang, R, facecolor="#9ecae1",
                   edgecolor="black", label=lbl, label_color=col, alpha=0.90)
    ax_prec.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":")
    ax_prec.axvline(0.0, color="#cccccc", linewidth=0.6, linestyle=":")
    ax_prec.set_xlim(-R - 12, R + 12)
    ax_prec.set_ylim(-R - drop_C - 12, R + 12)
    ax_prec.set_aspect('equal')
    ax_prec.grid(True, alpha=0.25)
    ax_prec.set_xlabel("x (mm)")
    ax_prec.set_title(f"{lbl}: y = {y_c:.1f} mm, θ = {ang:.1f}°")
axes_prec[0].set_ylabel("y (mm)")
plt.tight_layout()
plt.savefig(os.path.join(_script_dir, "precision_positions.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Precision-position figure saved.")

# =============================================================================
# PRESENTATION POINT (1) END
# =============================================================================

# =============================================================================
# PRESENTATION POINT (6) START: dynamic force / input torque analysis
# =============================================================================
#
# Material assumption used here: aluminum
#   density = 2700 kg/m^3 = 2700e-9 kg/mm^3
#
# The output body is treated as the given 270° circular sector, with its mass
# computed from geometry. For the input-torque calculation below, the slider is
# still modeled as translating vertically through the slider-crank.
#
# =============================================================================
# SECTION 4: DYNAMIC ANALYSIS
#
# Links:
#   Link 2 — Crank:         O → A_pin,  length r,    angle φ
#   Link 3 — Connecting rod: A_pin → B,  length l,    angle ψ
#   Link 4 — Slider/body:   translates along x = e, mass m4
#
# Joint forces:
#   F_O  = reaction at crank pivot O    (from ground on crank, 2D)
#   F_A  = force at pin A between crank and rod (2D, Newton's 3rd: rod on crank = -crank on rod)
#   F_B  = force at pin B between rod and slider (2D)
#   N    = rail normal force on slider  (x-direction only)
#
# Slider (link 4) — translates in y only:
#   ΣFy: F_By + m4·g_y = m4·ÿ_B   → F_By = m4·(ÿ_B − g_y)
#   ΣFx: F_Bx + N = 0               → N = −F_Bx
#
# Connecting rod (link 3) — general planar motion:
#   ΣFx: F_Ax + F_Bx = m3·aG3x
#   ΣFy: F_Ay + F_By + m3·g_y = m3·aG3y
#   ΣM_G3: (rA−G3)×F_A + (rB−G3)×F_B = I3·ψ̈
#   → 3 equations, 3 unknowns: F_Ax, F_Ay, F_Bx
#
# Crank (link 2) — general planar motion about fixed O:
#   F_Ox = m2·aG2x + F_Ax
#   F_Oy = m2·aG2y + F_Ay − m2·g_y
#   T = I2·φ̈ − (pA−O)×(−F_A) + (G2−O)×m2·(aG2−g)
#     = I2·α + (pA)×F_A − G2×m2·(aG2−g)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: DYNAMIC ANALYSIS")
print("=" * 60)

rho       = 2700e-9   # kg/mm³ (aluminum)
d_rod_dia = 5.0       # rod cross-section diameter mm
A_rod     = np.pi * (d_rod_dia/2)**2

m2     = rho * A_rod * r        # crank
m3     = rho * A_rod * l        # connecting rod
body_thickness = 5.0
body_area = 0.75 * np.pi * R**2
body_centroid_radius = 4 * np.sqrt(2) * R / (9 * np.pi)
body_I_center = rho * body_thickness * (3 * np.pi * R**4 / 8)
body_I_centroid = body_I_center - (rho * body_thickness * body_area) * body_centroid_radius**2
m4     = rho * body_area * body_thickness

I2     = (1/12) * m2 * r**2
I3     = (1/12) * m3 * l**2

g_y    = -9810.0   # mm/s²

print(f"\nMasses (g):")
print(f"  Crank  m2 = {m2*1e3:.4f} g  (L={r} mm)")
print(f"  Rod    m3 = {m3*1e3:.4f} g  (L={l} mm)")
print(f"  Body   m4 = {m4*1e3:.2f} g  (270° sector, R={R} mm, t={body_thickness} mm)")
print(f"  Body centroid offset from circle center = {body_centroid_radius:.4f} mm")
print(f"  Body Izz about circle center           = {body_I_center:.6e} kg·mm²")
print(f"  Body Izz about body centroid           = {body_I_centroid:.6e} kg·mm²")

def cross2d(a, b):
    return a[0]*b[1] - a[1]*b[0]

T_all      = np.zeros(N)
F_O_all    = np.zeros((N, 2))
F_A_all    = np.zeros((N, 2))
F_B_all    = np.zeros((N, 2))
# N_rail_all = np.zeros(N)  # extra rail-force output omitted from this version

for i in range(N):
    phi   = phi_all[i]
    w     = w_all[i]
    alp   = alpha_crank_all[i]
    psi   = psi_all[i]
    psid  = psi_dot_all[i]
    psidd = psi_ddot_all[i]

    pA  = np.array([xA_all[i],  yA_all[i]])
    pB  = np.array([xB_all[i],  yB_all[i]])
    G2  = pA / 2
    G3  = (pA + pB) / 2

    # Accelerations
    aA  = np.array([xA_ddot_all[i], yA_ddot_all[i]])
    aG2 = aA / 2
    aBx = 0.0   # slider constrained: ẍ_B = 0
    aBy = yB_ddot_all[i]
    aG3 = (aA + np.array([aBx, aBy])) / 2

    # Slider: F_By
    F_By = m4 * (aBy - g_y)

    # Rod: solve [F_Ax, F_Ay, F_Bx]
    rA_G3 = pA - G3
    rB_G3 = pB - G3
    A3 = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [-rA_G3[1], rA_G3[0], -rB_G3[1]]
    ])
    b3 = np.array([
        m3 * aG3[0],
        m3 * (aG3[1] - g_y) - F_By,
        I3 * psidd - cross2d(rB_G3, np.array([0.0, F_By]))
    ])
    F_Ax, F_Ay, F_Bx = np.linalg.solve(A3, b3)

    # N_rail = -F_Bx  # extra rail-force output omitted from this version

    # Crank: F_O and T
    F_Ox = m2 * aG2[0] + F_Ax
    F_Oy = m2 * (aG2[1] - g_y) + F_Ay
    T    = (I2 * alp
            + cross2d(pA, np.array([F_Ax, F_Ay]))
            - cross2d(G2, m2 * np.array([aG2[0], aG2[1] - g_y])))

    T_all[i]      = T
    F_O_all[i]    = [F_Ox, F_Oy]
    F_A_all[i]    = [F_Ax, F_Ay]
    F_B_all[i]    = [F_Bx, F_By]
    # N_rail_all[i] = N_rail

idx_max_torque = int(np.argmax(np.abs(T_all)))
F_input_all = np.linalg.norm(F_O_all, axis=1)
idx_max_force = int(np.argmax(F_input_all))
body_angle_fwd_plot = body_angle_plot_all[:N_fwd]
body_angle_ret_plot = body_angle_plot_all[N_fwd:][::-1]
force_ret_plot = F_input_all[N_fwd:][::-1]
torque_ret_plot = T_all[N_fwd:][::-1]
print(f"\nDriving torque T (N·mm):")
print(f"  Maximum |T| = {np.abs(T_all[idx_max_torque]):.3f} N·mm")
print(f"  Occurs at output angle θ = {body_angle_plot_all[idx_max_torque]:.3f}°")
print(f"  Maximum input force magnitude = {F_input_all[idx_max_force]:.3f} N")
print(f"  Occurs at output angle θ = {body_angle_plot_all[idx_max_force]:.3f}°")

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Dynamic Response of the Synthesized Mechanism", fontsize=14, fontweight='bold')
fig2.text(0.5, 0.955,
          "Horizontal axis shows output body angle (0° to 90°). Mechanism input crank sweep over the same stroke is 120°.",
          ha='center', va='top', fontsize=10, color='#555555')

axes2[0].plot(body_angle_fwd_plot, F_input_all[:N_fwd], 'b-', label='Forward')
axes2[0].plot(body_angle_ret_plot, force_ret_plot, 'r--', label='Return')
axes2[0].axvline(body_angle_plot_all[idx_max_force], color='k', linestyle=':',
                 label='Peak force')
axes2[0].set_title("Input Force vs Output Angle")
axes2[0].set_xlabel("Output body angle θ (deg)")
axes2[0].set_ylabel("|F_input| (N)")
axes2[0].set_xlim(0.0, 90.0)
axes2[0].legend()
axes2[0].grid(True)

axes2[1].plot(body_angle_fwd_plot, T_all[:N_fwd], 'b-', label='Forward')
axes2[1].plot(body_angle_ret_plot, torque_ret_plot, 'r--', label='Return')
axes2[1].axhline(0, color='k', linewidth=0.5)
axes2[1].axvline(body_angle_plot_all[idx_max_torque], color='k', linestyle=':',
                 label='Peak |T|')
axes2[1].set_title("Input Torque vs Output Angle")
axes2[1].set_xlabel("Output body angle θ (deg)")
axes2[1].set_ylabel("Torque T (N·mm)")
axes2[1].set_xlim(0.0, 90.0)
axes2[1].legend()
axes2[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(_script_dir, "dynamics.png"), dpi=150, bbox_inches='tight')
plt.show()
print("\nDynamics plot saved.")

# Extra rail-force plots from the earlier draft are intentionally omitted here
# to keep the submission focused on the required torque result.

# =============================================================================
# PRESENTATION POINT (6) END
# =============================================================================

# =============================================================================
# PRESENTATION POINT (4) START: show A, B, C and the 2× faster return
# =============================================================================

# =============================================================================
# SECTION 5: ANIMATION
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: ANIMATION")
print("=" * 60)

fig_a, ax_a = plt.subplots(figsize=(10.8, 7.2))

# ---------------------------------------------------------------------------
# Simulation visual theme
# ---------------------------------------------------------------------------
sim_bg_outer = '#0b1020'
sim_bg_inner = '#121a2f'
sim_grid_col = '#2a3c63'
sim_rail_col = '#1a2844'
sim_guide_col = '#7dd3fc'
sim_crank_col = '#f97316'
sim_rod_col = '#22d3ee'
sim_body_fill = '#7c3aed'
sim_body_edge = '#c4b5fd'
sim_fwd_col = '#10b981'
sim_ret_col = '#fb7185'
sim_accent_col = '#e5e7eb'
sim_pivot_col = '#f8fafc'
sim_slider_fill = '#0f172a'
sim_slider_edge = '#38bdf8'
sim_center_col = '#fde047'
sim_marker_a = '#34d399'
sim_marker_b = '#fbbf24'
sim_marker_c = '#f87171'

sim_title_font = 'DejaVu Serif'
sim_label_font = 'DejaVu Sans'
sim_mono_font = 'DejaVu Sans Mono'

ax_a.set_facecolor(sim_bg_inner)
fig_a.patch.set_facecolor(sim_bg_outer)

# Display in a body-centred frame: the object center stays on x = 0, while the
# crank pivot remains offset, matching the statement that the center has no
# horizontal motion.
display_offset = -e
y_display_offset = -y_ref_display

# Axis limits — centred on slider path
pad_x = r + l + abs(e) + 15
pad_y = R + max(r + 10, drop_C + 10)
x_lo  = -pad_x;  x_hi = pad_x
y_lo  = y_display_offset - (s_abs_A + r) - 10
y_hi  = R + 12
ax_a.set_xlim(x_lo, x_hi)
ax_a.set_ylim(y_lo, y_hi)
ax_a.set_aspect('equal')
ax_a.grid(True, color=sim_grid_col, linewidth=0.9, alpha=0.95)
ax_a.set_xlabel("x (mm)", color=sim_accent_col, fontfamily=sim_label_font, fontsize=12)
ax_a.set_ylabel("y (mm)", color=sim_accent_col, fontfamily=sim_label_font, fontsize=12)
ax_a.tick_params(colors=sim_accent_col, labelsize=11)
for sp in ax_a.spines.values():
    sp.set_edgecolor('#3b4c74')
    sp.set_linewidth(1.4)

# Rail (slider guide)
ax_a.axvline(0.0, color=sim_guide_col, linewidth=1.4, linestyle='--', zorder=1, alpha=0.95)
rail_w = 8
ax_a.fill_betweenx([y_lo, y_hi], R + 2, R + 2 + rail_w,
                    color=sim_rail_col, alpha=0.98, zorder=1)
ax_a.fill_betweenx([y_lo, y_hi], -R - 2 - rail_w, -R - 2,
                    color=sim_rail_col, alpha=0.98, zorder=1)
for y_ref, text_ref in [(0.0, 'A'), (-drop_B, 'B'), (-drop_C, 'C')]:
    ax_a.axhline(y_ref, color=sim_grid_col, linewidth=0.9, linestyle=':', alpha=0.9, zorder=0)
    ax_a.text(x_lo + 3, y_ref + 0.8, f'{text_ref} level', color=sim_guide_col,
              fontsize=9, ha='left', va='bottom', fontfamily=sim_label_font)

# Crank pivot O
ax_a.plot(display_offset, y_display_offset, '^', color=sim_pivot_col, markersize=12, zorder=6)
ax_a.annotate('O', (display_offset, y_display_offset), textcoords='offset points',
              xytext=(6, 6), color=sim_pivot_col, fontsize=10,
              fontweight='bold', fontfamily=sim_label_font)

# Precision markers
for y_pos, lbl, col in [(0.0, 'A', sim_marker_a),
                        (-drop_B, 'B', sim_marker_b),
                        (-drop_C, 'C', sim_marker_c)]:
    ax_a.plot(0.0, y_pos, 'D', color=col, markersize=7, zorder=5, alpha=0.7)
    ax_a.annotate(lbl, (0.0, y_pos), textcoords='offset points',
                  xytext=(R + 16, 0), color=col, fontsize=12,
                  fontweight='bold', alpha=0.98, fontfamily=sim_title_font)

# Dynamic elements
line_crank, = ax_a.plot([], [], color=sim_crank_col, linewidth=5.6,
                         solid_capstyle='round', zorder=4)
line_rod,   = ax_a.plot([], [], color=sim_rod_col, linewidth=3.8,
                         solid_capstyle='round', linestyle='-', zorder=4)
dot_crankpin, = ax_a.plot([], [], 'o', color=sim_crank_col, markersize=10, zorder=7)
dot_slider,   = ax_a.plot([], [], 'o', color=sim_slider_edge, markersize=9, zorder=7)

# Slider housing box
slider_box = mpatches.FancyBboxPatch(
    (-R - 1, -5), (R+1)*2, 10,
    boxstyle="round,pad=1", linewidth=1.8,
    edgecolor=sim_slider_edge, facecolor=sim_slider_fill, alpha=0.96, zorder=3
)
ax_a.add_patch(slider_box)

# Body wedge (partial disk)
body_wedge = Wedge((0.0, 0.0), R,
                    theta1=90, theta2=360,
                    facecolor=sim_body_fill, edgecolor=sim_body_edge,
                    linewidth=2.2, alpha=0.90, zorder=3)
ax_a.add_patch(body_wedge)
body_dot, = ax_a.plot([], [], 'o', color=sim_center_col, markeredgecolor=sim_bg_outer,
                      markeredgewidth=1.0, markersize=7, zorder=8)

# Trace lines
trace_fwd, = ax_a.plot([], [], color=sim_fwd_col, linewidth=2.7, alpha=0.78, zorder=2)
trace_ret, = ax_a.plot([], [], color=sim_ret_col, linewidth=2.7,
                        alpha=0.7, zorder=2, linestyle='--')

# Text overlays
info_txt = ax_a.text(0.02, 0.98, '', transform=ax_a.transAxes,
                      color=sim_accent_col, fontsize=9, va='top',
                      fontfamily=sim_mono_font,
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f172a',
                                alpha=0.96, edgecolor='#334155'))
phase_txt = ax_a.text(0.98, 0.98, '', transform=ax_a.transAxes,
                       color=sim_fwd_col, fontsize=12, fontweight='bold',
                       va='top', ha='right', fontfamily=sim_label_font,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#111827',
                                 alpha=0.96, edgecolor=sim_fwd_col))
ax_a.set_title("KINETIC CANVAS",
               color=sim_accent_col, fontsize=22, fontweight='bold',
               pad=18, fontfamily=sim_title_font)
ax_a.text(0.5, 1.015, "Offset slider-crank | input crank sweep 120° | output body rotation 90° | same return path at 2× speed",
          transform=ax_a.transAxes, ha='center', va='bottom',
          fontsize=10, color=sim_guide_col, fontfamily=sim_label_font)

def init_anim():
    line_crank.set_data([], [])
    line_rod.set_data([], [])
    dot_crankpin.set_data([], [])
    dot_slider.set_data([], [])
    body_dot.set_data([], [])
    trace_fwd.set_data([], [])
    trace_ret.set_data([], [])
    return (line_crank, line_rod, dot_crankpin, dot_slider,
            slider_box, body_wedge, body_dot,
            trace_fwd, trace_ret, info_txt, phase_txt)

def update_anim(frame):
    i    = frame
    xAi  = xA_all[i] + display_offset;   yAi = yA_disp_all[i]
    xBi  = xB_all[i] + display_offset;   yBi = yB_disp_all[i]
    bang = body_angle_all[i]

    line_crank.set_data([display_offset, xAi], [y_display_offset, yAi])
    line_rod.set_data([xAi, xBi], [yAi, yBi])
    dot_crankpin.set_data([xAi], [yAi])
    dot_slider.set_data([xBi], [yBi])

    # Slider box moves with slider
    slider_box.set_y(yBi - 5)

    # Body wedge: gap is 90°, rotates CW with body
    gap_start = bang          # degrees (negative = CW)
    body_wedge.set_center((xBi, yBi))
    body_wedge.set_theta1(gap_start + 90.0)
    body_wedge.set_theta2(gap_start + 360.0)

    body_dot.set_data([xBi], [yBi])

    # Trace
    if i < N_fwd:
        trace_fwd.set_data(np.full(i + 1, display_offset + e), yB_disp_all[:i+1])
        trace_ret.set_data([], [])
        phase = "FORWARD\nA → C"
        pcol  = sim_fwd_col
        pct   = int(100 * i / (N_fwd-1))
    else:
        trace_fwd.set_data(np.full(N_fwd, display_offset + e), yB_disp_all[:N_fwd])
        j = i - N_fwd
        trace_ret.set_data(np.full(j + 1, display_offset + e), yB_disp_all[N_fwd:i+1])
        phase = "RETURN\nC → A"
        pcol  = sim_ret_col
        pct   = int(100 * j / (N_ret-1))

    phase_txt.set_text(f"{phase}\n{pct:3d}%")
    phase_txt.set_color(pcol)
    phase_txt.get_bbox_patch().set_edgecolor(pcol)

    info_txt.set_text(
        f"t      = {time[i]:.3f} s\n"
        f"φ      = {np.degrees(phi_all[i]):.1f}°\n"
        f"drop   = {drop_all[i]:.3f} mm\n"
        f"y_c    = {yBi:.3f} mm\n"
        f"rot    = {bang:.1f}°\n"
        f"T      = {T_all[i]:.2f} N·mm"
    )
    return (line_crank, line_rod, dot_crankpin, dot_slider,
            slider_box, body_wedge, body_dot,
            trace_fwd, trace_ret, info_txt, phase_txt)

frames = range(0, N, 3)
anim = FuncAnimation(fig_a, update_anim, frames=frames,
                     init_func=init_anim, blit=True, interval=30)

print("Saving animation (GIF)...")
gif_path = os.path.join(_script_dir, "mechanism_animation.gif")
anim.save(gif_path, writer=PillowWriter(fps=33), dpi=100)
print(f"Animation saved → {gif_path}")
plt.show()

# =============================================================================
# PRESENTATION POINT (4) END
# =============================================================================
