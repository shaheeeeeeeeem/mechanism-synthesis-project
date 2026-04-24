import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Purpose
# ---------------------------------------------------------------------------
#
# This file is a "rigorous embodiment" companion to mainSynthesis.py.
#
# It checks the idea discussed earlier:
#   "Can the same slider-crank drive a REAL body-fixed attachment point P
#    on the circular object, instead of directly driving the body center C?"
#
# For a single vertical slider-crank output, the attachment point must lie on
# one vertical slider line in every precision position. This script shows that,
# for the required A/B/C poses, the only body-fixed point that satisfies that
# condition is the body center itself.
#
# That means:
#   - The current synthesis is kinematically valid as a center-driven model.
#   - A non-central, in-plane direct attachment point is NOT feasible with the
#     same one-slider topology.
#   - To get a truly non-central attachment point, you would need either:
#       1. an out-of-plane bracket/spacer embodiment, or
#       2. an additional linkage stage / different topology.
#
# So this file does not invent a fake "option 2" mechanism. Instead, it makes
# the feasibility limit explicit and visual.
#


# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------
R = 50.0
drop_B = 5.0
drop_C = 15.0

theta_A = 0.0
theta_B = -45.0
theta_C = -90.0

C_A = np.array([0.0, 0.0])
C_B = np.array([0.0, -drop_B])
C_C = np.array([0.0, -drop_C])


def rot2(theta_deg):
    """2D rotation matrix for degrees."""
    t = np.deg2rad(theta_deg)
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s], [s, c]])


def global_point(p_body, center_xy, theta_deg):
    """Map a body-fixed point into the global frame."""
    return center_xy + rot2(theta_deg) @ p_body


def attachment_x_residuals(p_body):
    """
    Residuals for a direct slider attachment:
    the attachment point must have the same global x-coordinate in A, B, C.
    """
    P_A = global_point(p_body, C_A, theta_A)
    P_B = global_point(p_body, C_B, theta_B)
    P_C = global_point(p_body, C_C, theta_C)
    return np.array([P_B[0] - P_A[0], P_C[0] - P_A[0]])


def draw_body(ax, center_xy, theta_deg, label, color, alpha=0.88):
    """Draw the 270-degree body in one precision position."""
    patch = Wedge(
        center_xy,
        R,
        theta1=theta_deg + 90.0,
        theta2=theta_deg + 360.0,
        facecolor=color,
        edgecolor='black',
        linewidth=1.2,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.plot(center_xy[0], center_xy[1], 'ko', markersize=4, zorder=4)
    ax.text(center_xy[0], center_xy[1] + R + 7, label, ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')


def main():
    # -----------------------------------------------------------------------
    # Analytic feasibility result
    # -----------------------------------------------------------------------
    #
    # Require x_A = x_B and x_A = x_C for a body-fixed point p_body = [u, v].
    #
    # With the required rotations:
    #   A: 0 deg
    #   B: -45 deg
    #   C: -90 deg
    #
    # The x-equality equations become:
    #   u = (u + v)/sqrt(2)
    #   u = v
    #
    # which together imply u = v = 0.
    #
    # So the only feasible direct in-plane slider attachment point is the
    # body center itself.
    #
    A_mat = np.array([
        [1.0 - 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
        [1.0, -1.0],
    ])
    b_vec = np.zeros(2)
    p_feasible = np.linalg.solve(A_mat, b_vec)

    # A nearby "would-be realistic" off-center candidate, used only to show
    # why a non-central attachment fails the vertical-slider test.
    p_candidate = np.array([10.0, -8.0])

    P_center_A = global_point(p_feasible, C_A, theta_A)
    P_center_B = global_point(p_feasible, C_B, theta_B)
    P_center_C = global_point(p_feasible, C_C, theta_C)

    P_cand_A = global_point(p_candidate, C_A, theta_A)
    P_cand_B = global_point(p_candidate, C_B, theta_B)
    P_cand_C = global_point(p_candidate, C_C, theta_C)

    cand_res = attachment_x_residuals(p_candidate)

    # -----------------------------------------------------------------------
    # Figure 1: Feasibility of a direct attachment point
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.suptitle("Rigorous Attachment-Point Study", fontsize=16, fontweight='bold')
    fig.text(
        0.5,
        0.94,
        "A one-slider output requires the driven body-fixed point to stay on one vertical line in A, B, and C.",
        ha='center',
        va='top',
        fontsize=10,
        color='#555555',
    )
    fig.text(
        0.5,
        0.915,
        "For the required 0 deg, -45 deg, -90 deg poses, the only such point is the body center.",
        ha='center',
        va='top',
        fontsize=10,
        color='#8b0000',
    )

    # Left panel: feasible point
    ax0 = axes[0]
    ax0.set_title("Only Feasible Direct Attachment")
    draw_body(ax0, C_A, theta_A, "A", "#9ecae1")
    draw_body(ax0, C_B, theta_B, "B", "#fdd49e")
    draw_body(ax0, C_C, theta_C, "C", "#fcbba1")
    ax0.axvline(0.0, color='#2c7fb8', linestyle='--', linewidth=1.5, label='Single slider line')
    ax0.plot(
        [P_center_A[0], P_center_B[0], P_center_C[0]],
        [P_center_A[1], P_center_B[1], P_center_C[1]],
        'o',
        color='#2c7fb8',
        markersize=6,
        label='Driven point P = body center',
        zorder=5,
    )
    ax0.text(
        0.02,
        0.02,
        "Result: P = (0, 0) in body coordinates",
        transform=ax0.transAxes,
        ha='left',
        va='bottom',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='#cccccc'),
    )
    ax0.set_aspect('equal')
    ax0.set_xlim(-R - 18, R + 18)
    ax0.set_ylim(-R - drop_C - 18, R + 18)
    ax0.grid(True, alpha=0.25)
    ax0.set_xlabel("x (mm)")
    ax0.set_ylabel("y (mm)")
    ax0.legend(loc='upper left', fontsize=9)

    # Right panel: infeasible off-center point
    ax1 = axes[1]
    ax1.set_title("Off-Center Point Fails Vertical-Slider Constraint")
    draw_body(ax1, C_A, theta_A, "A", "#9ecae1")
    draw_body(ax1, C_B, theta_B, "B", "#fdd49e")
    draw_body(ax1, C_C, theta_C, "C", "#fcbba1")
    ax1.axvline(P_cand_A[0], color='#2c7fb8', linestyle='--', linewidth=1.5, label='Slider line through A')
    ax1.plot(P_cand_A[0], P_cand_A[1], 'o', color='#1f77b4', markersize=7, label='P_A', zorder=5)
    ax1.plot(P_cand_B[0], P_cand_B[1], 'o', color='#ff7f0e', markersize=7, label='P_B', zorder=5)
    ax1.plot(P_cand_C[0], P_cand_C[1], 'o', color='#d62728', markersize=7, label='P_C', zorder=5)
    ax1.text(
        0.02,
        0.02,
        f"Example P = ({p_candidate[0]:.1f}, {p_candidate[1]:.1f})\n"
        f"x_B - x_A = {cand_res[0]:.2f} mm\n"
        f"x_C - x_A = {cand_res[1]:.2f} mm",
        transform=ax1.transAxes,
        ha='left',
        va='bottom',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='#cccccc'),
    )
    ax1.set_aspect('equal')
    ax1.set_xlim(-R - 18, R + 18)
    ax1.set_ylim(-R - drop_C - 18, R + 18)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax1.legend(loc='upper left', fontsize=9)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    plt.savefig(os.path.join(_script_dir, "realism_attachment_study.png"), dpi=150, bbox_inches='tight')
    plt.show()

    # -----------------------------------------------------------------------
    # Figure 2: Practical interpretation for the current project
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8.8, 5.8))
    fig2.suptitle("What This Means for the Project", fontsize=16, fontweight='bold')
    ax2.axis('off')

    summary = (
        "Conclusion\n\n"
        "1. The current slider-crank model is kinematically valid.\n"
        "   It drives the required A/B/C motion by attaching at the body center.\n\n"
        "2. A non-central, in-plane body-fixed point cannot be driven by the same\n"
        "   single vertical slider line for these three poses.\n\n"
        "3. Therefore, a more realistic embodiment would require either:\n"
        "   - a rear-plane bracket / spacer embodiment, or\n"
        "   - an additional linkage stage / different mechanism topology.\n\n"
        "So this file makes the geometry limit explicit: option 2 under the SAME\n"
        "single-slider topology collapses back to the center-driven case."
    )

    ax2.text(
        0.05,
        0.92,
        summary,
        ha='left',
        va='top',
        fontsize=12,
        linespacing=1.45,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', edgecolor='#cccccc'),
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    plt.savefig(os.path.join(_script_dir, "realism_summary.png"), dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
