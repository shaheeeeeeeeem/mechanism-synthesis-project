import os
import sys
import builtins
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.animation import FuncAnimation, PillowWriter

_script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Runtime switches for local use.
# - VERBOSE_OUTPUT prints synthesis/analysis values to the terminal.
# - SHOW_PLOTS opens the saved PNG figures in windows after generation.
# - SHOW_ANIMATION opens the live Matplotlib animation window.
# - SAVE_GIF exports mechanism_animation.gif during the same run.
VERBOSE_OUTPUT = False
if not VERBOSE_OUTPUT:
    print = lambda *args, **kwargs: None
else:
    print = builtins.print

SHOW_PLOTS = False
SHOW_ANIMATION = True
SAVE_GIF = True


def show_or_close(fig):
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

def rot2_deg(theta_deg):
    t = np.deg2rad(theta_deg)
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s], [s, c]])


def padded_limits(values, pad_frac=0.08, min_pad=0.5):
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    pad = max(min_pad, pad_frac * span if span > 1e-12 else min_pad)
    return vmin - pad, vmax + pad


def add_body_patch(ax, center_xy, angle_deg, radius, facecolor, edgecolor,
                   label=None, label_color="k", alpha=0.80, zorder=3):
    wedge = Wedge(
        center_xy,
        radius,
        theta1=angle_deg + 90.0,
        theta2=angle_deg + 360.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(wedge)
    ax.plot(center_xy[0], center_xy[1], "o", color=edgecolor, markersize=4, zorder=zorder + 1)
    if label:
        ax.text(
            center_xy[0],
            center_xy[1] + radius + 6,
            label,
            ha="center",
            va="bottom",
            color=label_color,
            fontsize=11,
            fontweight="bold",
        )
    return wedge


def add_corner_key(ax, items, loc="upper left"):
    if loc == "upper left":
        x0, y0, ha = 0.03, 0.97, "left"
    else:
        x0, y0, ha = 0.97, 0.97, "right"

    line_gap = 0.065
    for i, (label, color) in enumerate(items):
        ax.text(
            x0,
            y0 - i * line_gap,
            label,
            transform=ax.transAxes,
            ha=ha,
            va="top",
            fontsize=9,
            weight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.72),
            zorder=6,
        )


def draw_vector(ax, start, vec, color, lw=2.0):
    end = start + vec
    ax.annotate(
        "",
        xy=(end[0], end[1]),
        xytext=(start[0], start[1]),
        arrowprops=dict(arrowstyle="->", color=color, linewidth=lw, shrinkA=0, shrinkB=0),
        zorder=3,
    )
    return end


def bisection_root(func, a, b, tol=1e-12, max_iter=200):
    fa = func(a)
    fb = func(b)
    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b
    if fa * fb > 0:
        raise ValueError("Bisection requires a sign-changing bracket.")

    left, right = a, b
    f_left = fa
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = func(mid)
        if abs(f_mid) < tol or abs(right - left) < tol:
            return mid
        if f_left * f_mid <= 0:
            right = mid
        else:
            left = mid
            f_left = f_mid
    return 0.5 * (left + right)


def find_all_brackets(func, start, end, samples=4000):
    xs = np.linspace(start, end, samples)
    brackets = []
    x_prev = xs[0]
    f_prev = func(x_prev)
    for x in xs[1:]:
        f_curr = func(x)
        if np.isnan(f_prev) or np.isnan(f_curr):
            x_prev = x
            f_prev = f_curr
            continue
        if abs(f_prev) < 1e-12:
            brackets.append((x_prev, x_prev))
        elif f_prev * f_curr < 0:
            brackets.append((x_prev, x))
        x_prev = x
        f_prev = f_curr
    return brackets


def numerical_derivatives(values, x_vals):
    first = np.gradient(values, x_vals, edge_order=2)
    second = np.gradient(first, x_vals, edge_order=2)
    return first, second


def circumcenter_from_points(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    A = np.array([
        [2.0 * (x2 - x1), 2.0 * (y2 - y1)],
        [2.0 * (x3 - x1), 2.0 * (y3 - y1)],
    ])
    b = np.array([
        x2**2 + y2**2 - x1**2 - y1**2,
        x3**2 + y3**2 - x1**2 - y1**2,
    ])
    if abs(np.linalg.det(A)) < 1e-10:
        raise ValueError("Guidance-link synthesis failed because the three body-fixed positions became nearly collinear.")
    return np.linalg.solve(A, b)


def guidance_constraint(theta_deg, y_center, p_body, G_rot, L_rot):
    P_global = np.array([0.0, y_center]) + rot2_deg(theta_deg) @ p_body
    return np.dot(P_global - G_rot, P_global - G_rot) - L_rot**2


def solve_theta_branch(y_center, theta_guess_deg, p_body, G_rot, L_rot,
                       span_deg=180.0, samples=721, tol=1e-12):
    thetas = np.linspace(theta_guess_deg - span_deg, theta_guess_deg + span_deg, samples)
    f_prev = guidance_constraint(thetas[0], y_center, p_body, G_rot, L_rot)
    th_prev = thetas[0]
    candidates = []

    for th in thetas[1:]:
        f_curr = guidance_constraint(th, y_center, p_body, G_rot, L_rot)
        if abs(f_curr) < tol:
            candidates.append(th)
        elif abs(f_prev) < tol:
            candidates.append(th_prev)
        elif f_prev * f_curr < 0:
            root = bisection_root(
                lambda ang: guidance_constraint(ang, y_center, p_body, G_rot, L_rot),
                th_prev,
                th,
                tol=tol,
                max_iter=200,
            )
            candidates.append(root)
        th_prev = th
        f_prev = f_curr

    if not candidates:
        raise ValueError("Could not continue body rotation branch.")

    # Follow the branch closest to the previous accepted angle.
    return min(candidates, key=lambda th: abs(th - theta_guess_deg))


def synthesize_guidance_link(R, drop_B, drop_C, theta_body_A, theta_body_B, theta_body_C):
    # Candidate attachment-point fractions over the body. The baseline case
    # prefers the historical (-0.06R, -0.30R) point, but the search is broad
    # enough to remain feasible when the problem dimensions change.
    frac_x_values = np.linspace(-0.32, -0.02, 16)
    frac_y_values = np.linspace(-0.40, -0.08, 17)

    best = None
    best_score = None
    y_samples = np.linspace(0.0, -drop_C, 80)
    baseline_frac = np.array([-0.06, -0.30])

    for fx in frac_x_values:
        for fy in frac_y_values:
            p_body = np.array([fx * R, fy * R])
            try:
                P_A = np.array([0.0, 0.0]) + rot2_deg(theta_body_A) @ p_body
                P_B = np.array([0.0, -drop_B]) + rot2_deg(theta_body_B) @ p_body
                P_C = np.array([0.0, -drop_C]) + rot2_deg(theta_body_C) @ p_body
                G_rot = circumcenter_from_points(P_A, P_B, P_C)
                L_rot = np.linalg.norm(P_A - G_rot)

                theta_prev = theta_body_A
                theta_path = []
                for y_center in y_samples:
                    theta_prev = solve_theta_branch(y_center, theta_prev, p_body, G_rot, L_rot)
                    theta_path.append(theta_prev)

                # Enforce that the branch reaches the desired end orientation and
                # stays generally monotone toward -90 deg.
                end_err = abs(theta_path[-1] - theta_body_C)
                monotone_penalty = sum(
                    max(0.0, theta_path[i + 1] - theta_path[i])
                    for i in range(len(theta_path) - 1)
                )
                baseline_penalty = np.linalg.norm(np.array([fx, fy]) - baseline_frac)
                score = 200.0 * end_err + 50.0 * monotone_penalty + baseline_penalty

                if best_score is None or score < best_score:
                    best_score = score
                    best = (p_body, G_rot, L_rot)
            except Exception:
                continue

    if best is None:
        raise ValueError(
            "Could not synthesize a feasible guidance link for the current values of R, drop_B, and drop_C."
        )
    return best


# =============================================================================
# SECTION 1: PARAMETERS
# =============================================================================

R = 35
drop_B = 14
drop_C = 15.0
theta_body_A = 0.0
theta_body_B = -45.0
theta_body_C = -90.0
quick_return_ratio_target = 2.0

print("=" * 60)
print("SECTION 1: PARAMETERS")
print("=" * 60)
print(f"  Body radius R          = {R} mm")
print(f"  Drop at B              = {drop_B} mm")
print(f"  Drop at C              = {drop_C} mm")
print(f"  Body rotation A->C     = {theta_body_C} deg CW")
print(f"  Quick-return ratio     = {quick_return_ratio_target:.2f}:1")

# =============================================================================
# PRESENTATION POINT (2) START: synthesis method outline
# =============================================================================
#
# Mechanism used here:
#   1. A Whitworth-style quick-return stage with constant-RPM crank input
#      generates the vertical motion of the body center.
#   2. A second revolute link from ground to a body-fixed point forces the body
#      to rotate as the center translates, so the rotation is mechanism-driven
#      rather than imposed separately.
#
# =============================================================================
# PRESENTATION POINT (2) END
# =============================================================================

# =============================================================================
# PRESENTATION POINT (3) START: outline of how the code generates the output
# =============================================================================
#
# - Exact geometric relations are written directly for both mechanism stages.
# - The quick return comes from the Whitworth dead-center geometry under a
#   constant-speed full crank rotation.
# - The body-guidance link is synthesized so the rigid body passes exactly
#   through A, B, and C.
# - The program remains generic with respect to R, drop_B, and drop_C.
#
# =============================================================================
# PRESENTATION POINT (3) END
# =============================================================================

# =============================================================================
# SECTION 2: SYNTHESIS - WHITWORTH QUICK RETURN + BODY GUIDANCE LINK
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: SYNTHESIS - WHITWORTH QUICK RETURN + BODY GUIDANCE")
print("=" * 60)

# Primary mechanism:
#   O2-A is the constant-speed input crank.
#   A slides in the slot of lever O4-Q.
#   Q is connected by rod Q-C to the vertical ram slider at C.
#
# With O2O4 = 2*r_input, the Whitworth dead-center geometry gives a 2:1
# quick-return ratio for the ram motion.

r_input_norm = 1.0
O2_norm = np.array([0.0, 0.0])
O4_norm = np.array([2.0, 0.0])
Q_radius_norm = 4.0
rod_QC_norm = 3.4117647058823533
slider_x_norm = -0.10526315789473673
stroke_norm = 4.0
scale = drop_C / stroke_norm

r_input = r_input_norm * scale
O2 = O2_norm * scale
O4 = O4_norm * scale
Q_radius = Q_radius_norm * scale
rod_QC = rod_QC_norm * scale
slider_x = slider_x_norm * scale

# Recenter the entire Whitworth stage so the solved ram slider itself lies on
# x = 0. This preserves all primary-link lengths and keeps the displayed body
# center consistent with the actual mechanism geometry.
x_stage_shift = -slider_x
O2 = O2 + np.array([x_stage_shift, 0.0])
O4 = O4 + np.array([x_stage_shift, 0.0])
slider_x = slider_x + x_stage_shift


def whitworth_raw_state(phi):
    A = O2 + r_input * np.array([np.cos(phi), np.sin(phi)])
    psi = np.arctan2(A[1] - O4[1], A[0] - O4[0])
    Q = O4 + Q_radius * np.array([np.cos(psi), np.sin(psi)])
    dx = slider_x - Q[0]
    radicand = rod_QC**2 - dx**2
    if radicand < -1e-10:
        raise ValueError("Whitworth stage geometry became infeasible.")
    C = np.array([slider_x, Q[1] - np.sqrt(max(0.0, radicand))])
    return A, psi, Q, C


def y_center_raw(phi):
    return whitworth_raw_state(phi)[3][1]


def y_center_raw_dot(phi, h=1e-6):
    return (y_center_raw(phi + h) - y_center_raw(phi - h)) / (2.0 * h)


dead_center_brackets = find_all_brackets(y_center_raw_dot, 0.0, 2.0 * np.pi, samples=4000)
dead_centers = []
for a, b in dead_center_brackets:
    root = a if abs(a - b) < 1e-12 else bisection_root(y_center_raw_dot, a, b, tol=1e-12, max_iter=300)
    root_mod = root % (2.0 * np.pi)
    if not any(abs(((root_mod - r0 + np.pi) % (2.0 * np.pi)) - np.pi) < 1e-6 for r0 in dead_centers):
        dead_centers.append(root_mod)
dead_centers = sorted(dead_centers)

y_dead = [y_center_raw(phi) for phi in dead_centers]
if y_dead[0] >= y_dead[1]:
    phi_A = dead_centers[0]
    phi_C_base = dead_centers[1]
else:
    phi_A = dead_centers[1]
    phi_C_base = dead_centers[0]

delta_A_to_C = (phi_C_base - phi_A) % (2.0 * np.pi)
if delta_A_to_C < np.pi:
    phi_C = phi_C_base + 2.0 * np.pi
    delta_phi_forward = 2.0 * np.pi - delta_A_to_C
    delta_phi_return = delta_A_to_C
else:
    phi_C = phi_C_base
    delta_phi_forward = delta_A_to_C
    delta_phi_return = 2.0 * np.pi - delta_A_to_C

y_shift = y_center_raw(phi_A)

# Recenter the entire Whitworth stage vertically so the start pose A is at
# y = 0 for the ram slider and all connected stage points.
O2 = O2 + np.array([0.0, -y_shift])
O4 = O4 + np.array([0.0, -y_shift])


# Secondary mechanism:
#   A revolute guidance link from ground pivot G to body-fixed point P
#   forces the body to rotate as the center moves on x = 0.
#
# To keep the script generic, the guidance link is synthesized directly from
# the current R, drop_B, and drop_C values by searching for a feasible
# body-fixed attachment point and then constructing the corresponding ground
# pivot and link length.
p_body, G_rot, L_rot = synthesize_guidance_link(
    R, drop_B, drop_C, theta_body_A, theta_body_B, theta_body_C
)


def theta_from_center_y(y_center, theta_guess_deg):
    return solve_theta_branch(y_center, theta_guess_deg, p_body, G_rot, L_rot)


theta_A_check = theta_from_center_y(0.0, 0.0)
theta_B_check = theta_from_center_y(-drop_B, theta_body_B)
theta_C_check = theta_from_center_y(-drop_C, theta_body_C)

print("\nWhitworth quick-return stage:")
print(f"  Input crank length r_in     = {r_input:.6f} mm")
print(f"  Slotted lever pivot O4      = ({O4[0]:.6f}, {O4[1]:.6f}) mm")
print(f"  Lever point radius O4Q      = {Q_radius:.6f} mm")
print(f"  Connecting rod length QC    = {rod_QC:.6f} mm")
print(f"  Vertical slider line x_C    = {slider_x:.6f} mm")
print(f"  Forward crank sweep         = {np.degrees(delta_phi_forward):.6f} deg")
print(f"  Return crank sweep          = {np.degrees(delta_phi_return):.6f} deg")
print(f"  Quick-return ratio          = {delta_phi_forward / delta_phi_return:.6f}")

print("\nBody-guidance link:")
print(f"  Body-fixed attachment P     = ({p_body[0]:.6f}, {p_body[1]:.6f}) mm")
print(f"  Ground pivot G              = ({G_rot[0]:.6f}, {G_rot[1]:.6f}) mm")
print(f"  Guidance link length GP     = {L_rot:.6f} mm")
print(f"  Check at A                  = {theta_A_check:.6f} deg")
print(f"  Check at B                  = {theta_B_check:.6f} deg")
print(f"  Check at C                  = {theta_C_check:.6f} deg")

# =============================================================================
# SECTION 3: KINEMATIC ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: KINEMATIC ANALYSIS")
print("=" * 60)

N = 1200
omega_in = 1.0
phi_all = np.linspace(phi_A, phi_A + 2.0 * np.pi, N, endpoint=False)
phi_rel_all = phi_all - phi_A
time = phi_rel_all / omega_in

xA_all = np.zeros(N)
yA_all = np.zeros(N)
xQ_all = np.zeros(N)
yQ_all = np.zeros(N)
xC_all = np.zeros(N)
yC_all = np.zeros(N)
psi_all = np.zeros(N)
theta_all = np.zeros(N)
xP_all = np.zeros(N)
yP_all = np.zeros(N)

theta_guess = 0.0
for i, phi in enumerate(phi_all):
    A, psi, Q, C_raw = whitworth_raw_state(phi)
    y_center = C_raw[1]
    theta = theta_from_center_y(y_center, theta_guess)
    theta_guess = theta
    P_global = np.array([0.0, y_center]) + rot2_deg(theta) @ p_body

    xA_all[i], yA_all[i] = A[0], A[1]
    xQ_all[i], yQ_all[i] = Q[0], Q[1]
    xC_all[i], yC_all[i] = C_raw[0], y_center
    psi_all[i] = psi
    theta_all[i] = theta
    xP_all[i], yP_all[i] = P_global[0], P_global[1]

theta_unwrapped_rad = np.unwrap(np.deg2rad(theta_all))

xA_dot_all, xA_ddot_all = numerical_derivatives(xA_all, time)
yA_dot_all, yA_ddot_all = numerical_derivatives(yA_all, time)
xQ_dot_all, xQ_ddot_all = numerical_derivatives(xQ_all, time)
yQ_dot_all, yQ_ddot_all = numerical_derivatives(yQ_all, time)
yC_dot_all, yC_ddot_all = numerical_derivatives(yC_all, time)
theta_dot_all, theta_ddot_all = numerical_derivatives(theta_unwrapped_rad, time)
psi_unwrapped_all = np.unwrap(psi_all)
psi_dot_all, psi_ddot_all = numerical_derivatives(psi_unwrapped_all, time)

idx_C = int(np.argmin(np.abs((phi_all % (2.0 * np.pi)) - (phi_C % (2.0 * np.pi)))))
idx_B = int(np.argmin(np.abs(yC_all[:idx_C + 1] + drop_B)))
prec_idx = [0, idx_B, idx_C]
prec_labels = ["A", "B", "C"]
N_fwd = idx_C + 1
N_ret = N - N_fwd
fwd_idx = np.arange(0, idx_C + 1)
ret_idx = np.arange(idx_C + 1, N)

print(f"  Input crank speed           = {omega_in:.6f} rad/s")
print(f"  Total cycle time            = {time[-1]:.6f} s")
print(f"  Forward steps               = {N_fwd}")
print(f"  Return steps                = {N_ret}")
print(f"  Center at A                 = {yC_all[0]:.8f} mm")
print(f"  Center at B                 = {yC_all[idx_B]:.8f} mm")
print(f"  Center at C                 = {yC_all[idx_C]:.8f} mm")
print(f"  Body angle at A             = {theta_all[0]:.8f} deg")
print(f"  Body angle at B             = {theta_all[idx_B]:.8f} deg")
print(f"  Body angle at C             = {theta_all[idx_C]:.8f} deg")

# =============================================================================
# PRESENTATION POINT (5) START: position, velocity, and acceleration diagrams
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Whitworth + Guidance-Link Kinematic Diagrams", fontsize=14, fontweight="bold")
fig.text(
    0.5,
    0.935,
    "Constant-RPM input gives a 240 deg forward stroke and a 120 deg return stroke while the guidance link rotates the output body.",
    ha="center",
    va="top",
    fontsize=10,
    color="#555555",
)
fig.text(
    0.5,
    0.912,
    "Velocity and acceleration component plots include all moving joints a, q, c, and p. Ground pivots o2, o4, and g are fixed, so their values are identically zero.",
    ha="center",
    va="top",
    fontsize=9.5,
    color="#666666",
)

axes[0, 0].plot(time[fwd_idx], yC_all[fwd_idx], "b-", label="Forward")
if len(ret_idx) > 0:
    axes[0, 0].plot(time[ret_idx], yC_all[ret_idx], "r--", label="Return")
axes[0, 0].axhline(-drop_B, color="m", linestyle=":", label=f"B = {-drop_B} mm")
axes[0, 0].axhline(-drop_C, color="orange", linestyle=":", label=f"C = {-drop_C} mm")
axes[0, 0].set_title("Body Center Position y_c(t)")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("y_c (mm)")
axes[0, 0].set_xlim(time[0], time[-1])
axes[0, 0].set_ylim(-drop_C - 0.8, 0.8)
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(time[fwd_idx], yC_dot_all[fwd_idx], "b-", label="Forward")
if len(ret_idx) > 0:
    axes[0, 1].plot(time[ret_idx], yC_dot_all[ret_idx], "r--", label="Return")
axes[0, 1].axhline(0, color="k", linewidth=0.5)
axes[0, 1].set_title("Body Center Velocity v_c(t)")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("v_c (mm/s)")
axes[0, 1].set_xlim(time[0], time[-1])
axes[0, 1].set_ylim(*padded_limits(yC_dot_all))
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[0, 2].plot(time[fwd_idx], yC_ddot_all[fwd_idx], "b-", label="Forward")
if len(ret_idx) > 0:
    axes[0, 2].plot(time[ret_idx], yC_ddot_all[ret_idx], "r--", label="Return")
axes[0, 2].axhline(0, color="k", linewidth=0.5)
axes[0, 2].set_title("Body Center Acceleration a_c(t)")
axes[0, 2].set_xlabel("Time (s)")
axes[0, 2].set_ylabel("a_c (mm/s^2)")
axes[0, 2].set_xlim(time[0], time[-1])
axes[0, 2].set_ylim(*padded_limits(yC_ddot_all))
axes[0, 2].legend()
axes[0, 2].grid(True)

axes[1, 0].plot(time, xA_dot_all, label="point a: x-velocity")
axes[1, 0].plot(time, yA_dot_all, label="point a: y-velocity")
axes[1, 0].plot(time, xQ_dot_all, label="point q: x-velocity")
axes[1, 0].plot(time, yQ_dot_all, label="point q: y-velocity")
axes[1, 0].plot(time, yC_dot_all, label="point c: y-velocity")
axes[1, 0].plot(time, np.zeros_like(yC_dot_all), label="point c: x-velocity")
axes[1, 0].plot(time, np.gradient(xP_all, time, edge_order=2), label="point p: x-velocity")
axes[1, 0].plot(time, np.gradient(yP_all, time, edge_order=2), label="point p: y-velocity")
axes[1, 0].set_title("Joint Velocity Components")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Velocity (mm/s)")
axes[1, 0].set_xlim(time[0], time[-1])
axes[1, 0].set_ylim(*padded_limits(np.concatenate([
    xA_dot_all, yA_dot_all, xQ_dot_all, yQ_dot_all,
    np.zeros_like(yC_dot_all), yC_dot_all,
    np.gradient(xP_all, time, edge_order=2), np.gradient(yP_all, time, edge_order=2)
])))
axes[1, 0].legend(fontsize=8, ncol=2)
axes[1, 0].grid(True)

xp_dot_all, xp_ddot_all = numerical_derivatives(xP_all, time)
yp_dot_all, yp_ddot_all = numerical_derivatives(yP_all, time)

axes[1, 1].plot(time, xA_ddot_all, label="point a: x-acceleration")
axes[1, 1].plot(time, yA_ddot_all, label="point a: y-acceleration")
axes[1, 1].plot(time, xQ_ddot_all, label="point q: x-acceleration")
axes[1, 1].plot(time, yQ_ddot_all, label="point q: y-acceleration")
axes[1, 1].plot(time, np.zeros_like(yC_ddot_all), label="point c: x-acceleration")
axes[1, 1].plot(time, yC_ddot_all, label="point c: y-acceleration")
axes[1, 1].plot(time, xp_ddot_all, label="point p: x-acceleration")
axes[1, 1].plot(time, yp_ddot_all, label="point p: y-acceleration")
axes[1, 1].set_title("Joint Acceleration Components")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Acceleration (mm/s^2)")
axes[1, 1].set_xlim(time[0], time[-1])
axes[1, 1].set_ylim(*padded_limits(np.concatenate([
    xA_ddot_all, yA_ddot_all, xQ_ddot_all, yQ_ddot_all,
    np.zeros_like(yC_ddot_all), yC_ddot_all,
    xp_ddot_all, yp_ddot_all
])))
axes[1, 1].legend(fontsize=8, ncol=2)
axes[1, 1].grid(True)

axes[1, 2].axis("off")

plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
plt.savefig(os.path.join(_script_dir, "kinematics.png"), dpi=150, bbox_inches="tight")
show_or_close(fig)

# Velocity and acceleration loop-closure diagrams
fig_vel_loop, axes_vel_loop = plt.subplots(1, 3, figsize=(16, 5.6))
fig_vel_loop.suptitle("Velocity Loop-Closure Diagrams", fontsize=15, fontweight="bold")
fig_vel_loop.text(
    0.5,
    0.935,
    "All moving joints a, q, c, and p are shown at poses A, B, C. Fixed ground pivots o2, o4, and g have zero velocity.",
    ha="center",
    va="top",
    fontsize=10,
    color="#555555",
)
fig_vel_loop.text(
    0.5,
    0.910,
    "At poses A and C the mechanism is at reversal, so several velocity vectors become very small and bunch near the origin. Dashed arrows are relative vectors q/a, c/q, and p/c.",
    ha="center",
    va="top",
    fontsize=10,
    color="#8b0000",
)

fig_acc_loop, axes_acc_loop = plt.subplots(1, 3, figsize=(16, 5.6))
fig_acc_loop.suptitle("Acceleration Loop-Closure Diagrams", fontsize=15, fontweight="bold")
fig_acc_loop.text(
    0.5,
    0.935,
    "All moving joints a, q, c, and p are shown at poses A, B, C. Fixed ground pivots o2, o4, and g have zero acceleration.",
    ha="center",
    va="top",
    fontsize=10,
    color="#555555",
)
fig_acc_loop.text(
    0.5,
    0.910,
    "Dashed arrows are relative vectors q/a, c/q, and p/c. Solid arrows from the origin are the absolute joint vectors.",
    ha="center",
    va="top",
    fontsize=10,
    color="#8b0000",
)
fig_acc_loop.text(
    0.5,
    0.885,
    "At some poses, especially near A, the absolute acceleration vectors a_a and a_q are nearly collinear and overlap visually, so one arrow can partially hide the other.",
    ha="center",
    va="top",
    fontsize=9.5,
    color="#666666",
)

for col, (idx, lbl) in enumerate(zip(prec_idx, prec_labels)):
    vA = np.array([xA_dot_all[idx], yA_dot_all[idx]])
    vQ = np.array([xQ_dot_all[idx], yQ_dot_all[idx]])
    vC = np.array([0.0, yC_dot_all[idx]])
    vP = np.array([xp_dot_all[idx], yp_dot_all[idx]])
    vQA = vQ - vA
    vCQ = vC - vQ
    vPC = vP - vC

    aA = np.array([xA_ddot_all[idx], yA_ddot_all[idx]])
    aQ = np.array([xQ_ddot_all[idx], yQ_ddot_all[idx]])
    aC = np.array([0.0, yC_ddot_all[idx]])
    aP = np.array([xp_ddot_all[idx], yp_ddot_all[idx]])
    aQA = aQ - aA
    aCQ = aC - aQ
    aPC = aP - aC

    vel_scale = max(np.linalg.norm(vA), np.linalg.norm(vQ), np.linalg.norm(vC), np.linalg.norm(vP),
                    np.linalg.norm(vQA), np.linalg.norm(vCQ), np.linalg.norm(vPC), 1e-6)
    acc_scale = max(np.linalg.norm(aA), np.linalg.norm(aQ), np.linalg.norm(aC), np.linalg.norm(aP),
                    np.linalg.norm(aQA), np.linalg.norm(aCQ), np.linalg.norm(aPC), 1e-6)

    axv = axes_vel_loop[col]
    axv.set_title(f"Velocity Diagram at Pose {lbl}")
    draw_vector(axv, np.array([0.0, 0.0]), vA, "#1f77b4", lw=2.2)
    draw_vector(axv, np.array([0.0, 0.0]), vQ, "#2ca02c", lw=2.2)
    draw_vector(axv, np.array([0.0, 0.0]), vC, "#d62728", lw=2.2)
    draw_vector(axv, np.array([0.0, 0.0]), vP, "#9467bd", lw=2.2)
    v_tip_a = draw_vector(axv, vA, vQA, "#ff7f0e", lw=1.8)
    v_tip_q = draw_vector(axv, vQ, vCQ, "#17becf", lw=1.8)
    v_tip_c = draw_vector(axv, vC, vPC, "#e377c2", lw=1.8)
    axv.plot([0.0, vA[0], vQ[0], vC[0], vP[0], 0.0],
             [0.0, vA[1], vQ[1], vC[1], vP[1], 0.0],
             color="#888888", linestyle=":", linewidth=0.9, alpha=0.65)
    add_corner_key(axv, [("v_a", "#1f77b4"), ("v_q", "#2ca02c"), ("v_c", "#d62728"), ("v_p", "#9467bd")], loc="upper left")
    add_corner_key(axv, [("v_q/a", "#ff7f0e"), ("v_c/q", "#17becf"), ("v_p/c", "#e377c2")], loc="upper right")
    axv.axhline(0, color="k", linewidth=0.5)
    axv.axvline(0, color="k", linewidth=0.5)
    axv.set_aspect("equal")
    axv.grid(True, alpha=0.35)
    axv.set_xlim(-1.2 * vel_scale, 1.2 * vel_scale)
    axv.set_ylim(-1.2 * vel_scale, 1.2 * vel_scale)
    axv.set_xlabel("x-component")
    axv.set_ylabel("y-component")

    axa = axes_acc_loop[col]
    axa.set_title(f"Acceleration Diagram at Pose {lbl}")
    draw_vector(axa, np.array([0.0, 0.0]), aA, "#1f77b4", lw=2.2)
    draw_vector(axa, np.array([0.0, 0.0]), aQ, "#2ca02c", lw=2.2)
    draw_vector(axa, np.array([0.0, 0.0]), aC, "#d62728", lw=2.2)
    draw_vector(axa, np.array([0.0, 0.0]), aP, "#9467bd", lw=2.2)
    a_tip_a = draw_vector(axa, aA, aQA, "#ff7f0e", lw=1.8)
    a_tip_q = draw_vector(axa, aQ, aCQ, "#17becf", lw=1.8)
    a_tip_c = draw_vector(axa, aC, aPC, "#e377c2", lw=1.8)
    axa.plot([0.0, aA[0], aQ[0], aC[0], aP[0], 0.0],
             [0.0, aA[1], aQ[1], aC[1], aP[1], 0.0],
             color="#888888", linestyle=":", linewidth=0.9, alpha=0.65)
    add_corner_key(axa, [("a_a", "#1f77b4"), ("a_q", "#2ca02c"), ("a_c", "#d62728"), ("a_p", "#9467bd")], loc="upper left")
    add_corner_key(axa, [("a_q/a", "#ff7f0e"), ("a_c/q", "#17becf"), ("a_p/c", "#e377c2")], loc="upper right")
    axa.axhline(0, color="k", linewidth=0.5)
    axa.axvline(0, color="k", linewidth=0.5)
    axa.set_aspect("equal")
    axa.grid(True, alpha=0.35)
    axa.set_xlim(-1.2 * acc_scale, 1.2 * acc_scale)
    axa.set_ylim(-1.2 * acc_scale, 1.2 * acc_scale)
    axa.set_xlabel("x-component")
    axa.set_ylabel("y-component")

plt.figure(fig_vel_loop.number)
plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.84])
plt.savefig(os.path.join(_script_dir, "loop_closure_velocity.png"), dpi=150, bbox_inches="tight")
show_or_close(fig_vel_loop)

plt.figure(fig_acc_loop.number)
plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.82])
plt.savefig(os.path.join(_script_dir, "loop_closure_acceleration.png"), dpi=150, bbox_inches="tight")
show_or_close(fig_acc_loop)

# =============================================================================
# PRESENTATION POINT (5) END
# =============================================================================

# =============================================================================
# PRESENTATION POINT (1) START: show the synthesized mechanism / output body
# =============================================================================

fig_prec, axes_prec = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
fig_prec.suptitle("Precision Positions of the Output Body", fontsize=14, fontweight="bold")
fig_prec.text(
    0.5,
    0.935,
    "These are the required A, B, C output poses produced by the Whitworth quick-return stage together with the guidance link g-p.",
    ha="center",
    va="top",
    fontsize=10,
    color="#555555",
)
precision_data = [
    ("A", 0.0, theta_body_A, "#1f7a8c"),
    ("B", -drop_B, theta_body_B, "#c49a00"),
    ("C", -drop_C, theta_body_C, "#b24747"),
]
for ax_prec, (lbl, y_c, ang, col) in zip(axes_prec, precision_data):
    add_body_patch(ax_prec, (0.0, y_c), ang, R, facecolor="#9ecae1",
                   edgecolor="black", label=lbl, label_color=col, alpha=0.90)
    Pp = np.array([0.0, y_c]) + rot2_deg(ang) @ p_body
    ax_prec.plot([G_rot[0], Pp[0]], [G_rot[1], Pp[1]], color=col, linewidth=2.0)
    ax_prec.plot(*G_rot, "kD", markersize=6)
    ax_prec.plot(*Pp, "o", color=col, markersize=5)
    ax_prec.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":")
    ax_prec.axvline(0.0, color="#cccccc", linewidth=0.6, linestyle=":")
    ax_prec.set_xlim(-R - 20, R + 12)
    ax_prec.set_ylim(-R - drop_C - 12, R + 12)
    ax_prec.set_aspect("equal")
    ax_prec.grid(True, alpha=0.25)
    ax_prec.set_xlabel("x (mm)")
    ax_prec.set_title(f"{lbl}: y = {y_c:.1f} mm, theta = {ang:.1f} deg")
axes_prec[0].set_ylabel("y (mm)")
plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
plt.savefig(os.path.join(_script_dir, "precision_positions.png"), dpi=150, bbox_inches="tight")
show_or_close(fig_prec)

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
# Gravity is neglected here, per instructor guidance.
# The input torque below is estimated from the kinetic-energy variation over the
# cycle under constant input speed:
#   T_in = d(K_total) / d(phi)
#
# =============================================================================

rho = 2700e-9
d_rod_dia = 5.0
A_rod = np.pi * (d_rod_dia / 2.0) ** 2
body_thickness = 5.0
body_area = 0.75 * np.pi * R**2
body_centroid_radius = 4.0 * np.sqrt(2.0) * R / (9.0 * np.pi)
body_centroid_vec0 = np.array([-body_centroid_radius / np.sqrt(2.0), -body_centroid_radius / np.sqrt(2.0)])
body_I_center = rho * body_thickness * (3.0 * np.pi * R**4 / 8.0)
m_body = rho * body_area * body_thickness
body_I_centroid = body_I_center - m_body * body_centroid_radius**2

m_crank = rho * A_rod * r_input
m_lever = rho * A_rod * Q_radius
m_qc = rho * A_rod * rod_QC
m_rot = rho * A_rod * L_rot

crank_angle_all = phi_all
lever_angle_all = np.unwrap(psi_all)
qc_angle_all = np.unwrap(np.arctan2(yC_all - yQ_all, xC_all - xQ_all))
rotlink_angle_all = np.unwrap(np.arctan2(yP_all - G_rot[1], xP_all - G_rot[0]))

lever_angle_dot_all, _ = numerical_derivatives(lever_angle_all, time)
qc_angle_dot_all, _ = numerical_derivatives(qc_angle_all, time)
rotlink_angle_dot_all, _ = numerical_derivatives(rotlink_angle_all, time)

G_crank_x = 0.5 * (O2[0] + xA_all)
G_crank_y = 0.5 * (O2[1] + yA_all)
G_lever_x = 0.5 * (O4[0] + xQ_all)
G_lever_y = 0.5 * (O4[1] + yQ_all)
G_qc_x = 0.5 * (xQ_all + xC_all)
G_qc_y = 0.5 * (yQ_all + yC_all)
G_rot_x = 0.5 * (G_rot[0] + xP_all)
G_rot_y = 0.5 * (G_rot[1] + yP_all)

G_crank_vx, _ = numerical_derivatives(G_crank_x, time)
G_crank_vy, _ = numerical_derivatives(G_crank_y, time)
G_lever_vx, _ = numerical_derivatives(G_lever_x, time)
G_lever_vy, _ = numerical_derivatives(G_lever_y, time)
G_qc_vx, _ = numerical_derivatives(G_qc_x, time)
G_qc_vy, _ = numerical_derivatives(G_qc_y, time)
G_rot_vx, _ = numerical_derivatives(G_rot_x, time)
G_rot_vy, _ = numerical_derivatives(G_rot_y, time)

body_centroid_x = xC_all + (rot2_deg(0.0) @ body_centroid_vec0)[0]
body_centroid_y = np.zeros(N)
for i in range(N):
    body_centroid = np.array([xC_all[i], yC_all[i]]) + rot2_deg(theta_all[i]) @ body_centroid_vec0
    body_centroid_x[i] = body_centroid[0]
    body_centroid_y[i] = body_centroid[1]
body_centroid_vx, _ = numerical_derivatives(body_centroid_x, time)
body_centroid_vy, _ = numerical_derivatives(body_centroid_y, time)

I_crank = (1.0 / 12.0) * m_crank * r_input**2
I_lever = (1.0 / 12.0) * m_lever * Q_radius**2
I_qc = (1.0 / 12.0) * m_qc * rod_QC**2
I_rot = (1.0 / 12.0) * m_rot * L_rot**2

K_total = (
    0.5 * m_crank * (G_crank_vx**2 + G_crank_vy**2) + 0.5 * I_crank * omega_in**2
    + 0.5 * m_lever * (G_lever_vx**2 + G_lever_vy**2) + 0.5 * I_lever * lever_angle_dot_all**2
    + 0.5 * m_qc * (G_qc_vx**2 + G_qc_vy**2) + 0.5 * I_qc * qc_angle_dot_all**2
    + 0.5 * m_rot * (G_rot_vx**2 + G_rot_vy**2) + 0.5 * I_rot * rotlink_angle_dot_all**2
    + 0.5 * m_body * (body_centroid_vx**2 + body_centroid_vy**2) + 0.5 * body_I_centroid * theta_dot_all**2
)

T_all = np.gradient(K_total, phi_all, edge_order=2)
F_input_all = np.abs(T_all) / max(r_input, 1e-9)
idx_max_torque = int(np.argmax(np.abs(T_all)))
idx_max_force = int(np.argmax(F_input_all))

phi_fwd_plot = np.degrees(phi_rel_all[:N_fwd])
phi_ret_plot = np.degrees(phi_rel_all[idx_C + 1:])

print("\n" + "=" * 60)
print("SECTION 4: DYNAMIC ANALYSIS")
print("=" * 60)
print(f"  Crank mass                = {m_crank * 1e3:.6f} g")
print(f"  Slotted lever mass        = {m_lever * 1e3:.6f} g")
print(f"  Rod QC mass               = {m_qc * 1e3:.6f} g")
print(f"  Rotation link mass        = {m_rot * 1e3:.6f} g")
print(f"  Body mass                 = {m_body * 1e3:.6f} g")
print(f"  Maximum |T|               = {np.abs(T_all[idx_max_torque]):.6f} N·mm")
print(f"  Occurs at crank angle     = {np.degrees(phi_rel_all[idx_max_torque]):.6f} deg")
print(f"  Maximum |F_input|         = {F_input_all[idx_max_force]:.6f} N")
print(f"  Occurs at crank angle     = {np.degrees(phi_rel_all[idx_max_force]):.6f} deg")

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Dynamic Response of the Synthesized Mechanism", fontsize=14, fontweight="bold")
fig2.text(
    0.5,
    0.935,
    "Input crank o2-a rotates at constant RPM through 360 deg. The Whitworth stage gives a 240 deg forward stroke and a 120 deg return stroke.",
    ha="center",
    va="top",
    fontsize=10,
    color="#555555",
)

axes2[0].plot(phi_fwd_plot, F_input_all[:N_fwd], "b-", label="Forward")
if len(phi_ret_plot) > 0:
    axes2[0].plot(phi_ret_plot, F_input_all[idx_C + 1:], "r--", label="Return")
axes2[0].axvline(np.degrees(phi_rel_all[idx_max_force]), color="k", linestyle=":", label="Peak force")
axes2[0].set_title("Input Force vs Input Crank Angle")
axes2[0].set_xlabel("Input crank angle phi (deg)")
axes2[0].set_ylabel("|F_input| (N)")
axes2[0].set_xlim(0.0, 360.0)
axes2[0].legend()
axes2[0].grid(True)

axes2[1].plot(phi_fwd_plot, T_all[:N_fwd], "b-", label="Forward")
if len(phi_ret_plot) > 0:
    axes2[1].plot(phi_ret_plot, T_all[idx_C + 1:], "r--", label="Return")
axes2[1].axhline(0, color="k", linewidth=0.5)
axes2[1].axvline(np.degrees(phi_rel_all[idx_max_torque]), color="k", linestyle=":", label="Peak |T|")
axes2[1].set_title("Input Torque vs Input Crank Angle")
axes2[1].set_xlabel("Input crank angle phi (deg)")
axes2[1].set_ylabel("Torque T (N·mm)")
axes2[1].set_xlim(0.0, 360.0)
axes2[1].legend()
axes2[1].grid(True)

plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
plt.savefig(os.path.join(_script_dir, "dynamics.png"), dpi=150, bbox_inches="tight")
show_or_close(fig2)

# =============================================================================
# PRESENTATION POINT (6) END
# =============================================================================

# =============================================================================
# PRESENTATION POINT (4) START: show A, B, C and the quick return
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: ANIMATION")
print("=" * 60)

fig_a, ax_a = plt.subplots(figsize=(11.2, 7.4))

sim_bg_outer = "#0b1020"
sim_bg_inner = "#121a2f"
sim_grid_col = "#2a3c63"
sim_guide_col = "#7dd3fc"
sim_crank_col = "#f97316"
sim_slot_col = "#f8fafc"
sim_rod_col = "#22d3ee"
sim_rot_col = "#fbbf24"
sim_body_fill = "#7c3aed"
sim_body_edge = "#c4b5fd"
sim_fwd_col = "#10b981"
sim_ret_col = "#fb7185"
sim_accent_col = "#e5e7eb"
sim_center_col = "#fde047"
sim_pivot_col = "#f8fafc"

sim_title_font = "DejaVu Serif"
sim_label_font = "DejaVu Sans"
sim_mono_font = "DejaVu Sans Mono"
sim_point_font = "DejaVu Sans Mono"

ax_a.set_facecolor(sim_bg_inner)
fig_a.patch.set_facecolor(sim_bg_outer)
ax_a.grid(True, color=sim_grid_col, linewidth=0.9, alpha=0.95)
for sp in ax_a.spines.values():
    sp.set_edgecolor("#3b4c74")
    sp.set_linewidth(1.4)
ax_a.tick_params(colors=sim_accent_col, labelsize=11)
ax_a.set_xlabel("x (mm)", color=sim_accent_col, fontfamily=sim_label_font, fontsize=12)
ax_a.set_ylabel("y (mm)", color=sim_accent_col, fontfamily=sim_label_font, fontsize=12)

pad_x = max(abs(O4[0]), abs(G_rot[0]), Q_radius, R) + 55
pad_y = max(R + 15, abs(np.min(yC_all)) + 30, abs(np.max(yQ_all)) + 20)
ax_a.set_xlim(-pad_x, pad_x)
ax_a.set_ylim(-pad_y, pad_y)
ax_a.set_aspect("equal")

ax_a.axvline(0.0, color=sim_guide_col, linewidth=1.4, linestyle="--", zorder=1, alpha=0.95)
for y_ref, text_ref in [(0.0, "A"), (-drop_B, "B"), (-drop_C, "C")]:
    ax_a.axhline(y_ref, color=sim_grid_col, linewidth=0.9, linestyle=":", alpha=0.9, zorder=0)
    ax_a.text(-pad_x + 3, y_ref + 0.8, f"{text_ref} level", color=sim_guide_col,
              fontsize=9, ha="left", va="bottom", fontfamily=sim_label_font)

pivot_annotations = [
    (O2, "o2", (8, 8)),
    (O4, "o4", (10, 8)),
    (G_rot, "g", (-10, 8)),
]
for pt, name, offset in pivot_annotations:
    ax_a.plot(pt[0], pt[1], "^", color=sim_pivot_col, markersize=10, zorder=6)
    ax_a.annotate(
        name,
        (pt[0], pt[1]),
        textcoords="offset points",
        xytext=offset,
        ha="left" if offset[0] >= 0 else "right",
        color=sim_pivot_col,
        fontsize=10,
        fontweight="bold",
        fontfamily=sim_point_font,
    )

for y_pos, lbl, col in [(0.0, "A", "#34d399"), (-drop_B, "B", "#fbbf24"), (-drop_C, "C", "#f87171")]:
    ax_a.plot(0.0, y_pos, "D", color=col, markersize=7, zorder=5, alpha=0.7)
    ax_a.annotate(lbl, (0.0, y_pos), textcoords="offset points",
                  xytext=(R + 16, 0), color=col, fontsize=12,
                  fontweight="bold", alpha=0.98, fontfamily=sim_title_font)

line_crank, = ax_a.plot([], [], color=sim_crank_col, linewidth=4.8, solid_capstyle="round", zorder=7)
line_slot, = ax_a.plot([], [], color=sim_slot_col, linewidth=3.4, solid_capstyle="round", zorder=6)
line_qc, = ax_a.plot([], [], color=sim_rod_col, linewidth=4.2, solid_capstyle="round", zorder=8)
line_rot, = ax_a.plot([], [], color=sim_rot_col, linewidth=3.0, solid_capstyle="round", alpha=0.88, zorder=7)
line_body_attach, = ax_a.plot([], [], color="#fef08a", linewidth=2.1, linestyle="--", alpha=0.95, zorder=8)
dot_A, = ax_a.plot([], [], "o", color=sim_crank_col, markersize=9, zorder=9)
dot_Q, = ax_a.plot([], [], "o", color=sim_rod_col, markersize=8, zorder=9)
dot_C, = ax_a.plot([], [], "o", color=sim_center_col, markeredgecolor=sim_bg_outer,
                   markeredgewidth=1.0, markersize=8, zorder=10)
dot_P, = ax_a.plot([], [], "o", color=sim_rot_col, markersize=7, zorder=10)

txt_a = ax_a.text(0.0, 0.0, "a", color=sim_crank_col, fontsize=10, fontweight="bold",
                  fontfamily=sim_point_font, zorder=11)
txt_q = ax_a.text(0.0, 0.0, "q", color=sim_rod_col, fontsize=10, fontweight="bold",
                  fontfamily=sim_point_font, zorder=11)
txt_c = ax_a.text(0.0, 0.0, "c", color=sim_center_col, fontsize=10, fontweight="bold",
                  fontfamily=sim_point_font, zorder=11)
txt_p = ax_a.text(0.0, 0.0, "p", color=sim_rot_col, fontsize=10, fontweight="bold",
                  fontfamily=sim_point_font, zorder=11)

body_wedge = Wedge((0.0, 0.0), R, theta1=90, theta2=360,
                   facecolor=sim_body_fill, edgecolor=sim_body_edge,
                   linewidth=2.2, alpha=0.72, zorder=3)
ax_a.add_patch(body_wedge)

trace_fwd, = ax_a.plot([], [], color=sim_fwd_col, linewidth=2.7, alpha=0.78, zorder=2)
trace_ret, = ax_a.plot([], [], color=sim_ret_col, linewidth=2.7, alpha=0.70, zorder=2, linestyle="--")

info_txt = ax_a.text(
    0.02, 0.98, "", transform=ax_a.transAxes, color=sim_accent_col, fontsize=9,
    va="top", fontfamily=sim_mono_font,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", alpha=0.96, edgecolor="#334155")
)
phase_txt = ax_a.text(
    0.98, 0.98, "", transform=ax_a.transAxes, color=sim_fwd_col, fontsize=12,
    fontweight="bold", va="top", ha="right", fontfamily=sim_label_font,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#111827", alpha=0.96, edgecolor=sim_fwd_col)
)
ax_a.set_title("KINETIC CANVAS",
               color=sim_accent_col, fontsize=22, fontweight="bold",
               pad=18, fontfamily=sim_title_font)
ax_a.text(
    0.5,
    1.015,
    "Whitworth quick-return | constant RPM input | mechanism-driven rotation | 240 deg forward stroke | 120 deg return stroke",
    transform=ax_a.transAxes,
    ha="center",
    va="bottom",
    fontsize=10,
    color=sim_guide_col,
    fontfamily=sim_label_font,
)
legend_items = [
    ("o2-a : input crank", sim_crank_col),
    ("o4-q : slotted lever", sim_slot_col),
    ("q-c : ram link", sim_rod_col),
    ("g-p : guidance link", sim_rot_col),
    ("c-p : body-fixed offset", "#fef08a"),
]
for j, (txt, col) in enumerate(legend_items):
    y = 0.17 - 0.043 * j
    ax_a.text(
        0.02,
        y,
        txt,
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=col,
        fontfamily=sim_point_font,
        bbox=dict(boxstyle="round,pad=0.16", facecolor="#0f172a", edgecolor="#334155", alpha=0.92),
        zorder=12,
    )


def init_anim():
    line_crank.set_data([], [])
    line_slot.set_data([], [])
    line_qc.set_data([], [])
    line_rot.set_data([], [])
    line_body_attach.set_data([], [])
    dot_A.set_data([], [])
    dot_Q.set_data([], [])
    dot_C.set_data([], [])
    dot_P.set_data([], [])
    txt_a.set_position((0.0, 0.0))
    txt_q.set_position((0.0, 0.0))
    txt_c.set_position((0.0, 0.0))
    txt_p.set_position((0.0, 0.0))
    trace_fwd.set_data([], [])
    trace_ret.set_data([], [])
    return (
        line_crank, line_slot, line_qc, line_rot, line_body_attach,
        dot_A, dot_Q, dot_C, dot_P,
        txt_a, txt_q, txt_c, txt_p,
        body_wedge, trace_fwd, trace_ret, info_txt, phase_txt
    )


def update_anim(frame):
    i = frame
    A = np.array([xA_all[i], yA_all[i]])
    Q = np.array([xQ_all[i], yQ_all[i]])
    C = np.array([xC_all[i], yC_all[i]])
    P = np.array([xP_all[i], yP_all[i]])
    theta = theta_all[i]

    line_crank.set_data([O2[0], A[0]], [O2[1], A[1]])
    line_slot.set_data([O4[0], Q[0]], [O4[1], Q[1]])
    line_qc.set_data([Q[0], C[0]], [Q[1], C[1]])
    line_rot.set_data([G_rot[0], P[0]], [G_rot[1], P[1]])
    line_body_attach.set_data([C[0], P[0]], [C[1], P[1]])
    dot_A.set_data([A[0]], [A[1]])
    dot_Q.set_data([Q[0]], [Q[1]])
    dot_C.set_data([C[0]], [C[1]])
    dot_P.set_data([P[0]], [P[1]])
    txt_a.set_position((A[0] + 3.0, A[1] + 2.0))
    txt_q.set_position((Q[0] - 4.0, Q[1] + 2.0))
    txt_c.set_position((C[0] + 2.0, C[1] - 3.0))
    txt_p.set_position((P[0] - 4.0, P[1] + 2.0))

    body_wedge.set_center((C[0], C[1]))
    body_wedge.set_theta1(theta + 90.0)
    body_wedge.set_theta2(theta + 360.0)

    if i <= idx_C:
        trace_fwd.set_data(xC_all[:i + 1], yC_all[:i + 1])
        trace_ret.set_data([], [])
        phase = "FORWARD\nA -> C"
        pcol = sim_fwd_col
        pct = int(100 * i / max(idx_C, 1))
    else:
        trace_fwd.set_data(xC_all[:idx_C + 1], yC_all[:idx_C + 1])
        trace_ret.set_data(xC_all[idx_C:i + 1], yC_all[idx_C:i + 1])
        phase = "RETURN\nC -> A"
        pcol = sim_ret_col
        pct = int(100 * (i - idx_C) / max(N - idx_C - 1, 1))

    phase_txt.set_text(f"{phase}\n{pct:3d}%")
    phase_txt.set_color(pcol)
    phase_txt.get_bbox_patch().set_edgecolor(pcol)

    info_txt.set_text(
        f"t      = {time[i]:.3f} s\n"
        f"phi    = {np.degrees(phi_rel_all[i]):.1f} deg\n"
        f"y_c    = {C[1]:.3f} mm\n"
        f"rot    = {theta:.1f} deg\n"
        f"T      = {T_all[i]:.2f} N·mm\n"
        f"v_c    = {yC_dot_all[i]:.2f} mm/s"
    )

    return (
        line_crank, line_slot, line_qc, line_rot, line_body_attach,
        dot_A, dot_Q, dot_C, dot_P,
        txt_a, txt_q, txt_c, txt_p,
        body_wedge, trace_fwd, trace_ret, info_txt, phase_txt
    )


frames = range(0, N, 5)
anim = FuncAnimation(fig_a, update_anim, frames=frames, init_func=init_anim, blit=True, interval=24)
if SAVE_GIF:
    gif_path = os.path.join(_script_dir, "mechanism_animation.gif")
    anim.save(gif_path, writer=PillowWriter(fps=36), dpi=100)
if SHOW_ANIMATION:
    plt.show()
else:
    plt.close(fig_a)

# =============================================================================
# PRESENTATION POINT (4) END
# =============================================================================
