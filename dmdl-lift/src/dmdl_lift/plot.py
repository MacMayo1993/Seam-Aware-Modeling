"""Visualization for validation results."""
import numpy as np
import matplotlib.pyplot as plt


def plot_results(results):
    """Generate comprehensive validation report figure.

    Creates a 3x2 grid showing:
    - K-mer complexity curves
    - Model complexity (phases, ESS)
    - Signal examples
    - ΔMDL trajectories

    Parameters
    ----------
    results : dict
        Results from validate_method()

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Panel 1: K-mer complexity
    ax1 = fig.add_subplot(gs[0, 0])
    comp_p = results['complexity']['periodic']
    comp_f = results['complexity']['fibonacci']
    Ls = sorted(comp_p.keys())

    ax1.plot(Ls, [comp_p[L] for L in Ls], 'bo-', linewidth=2.5,
             markersize=9, label='Periodic')
    ax1.plot(Ls, [comp_f[L] for L in Ls], 'ro-', linewidth=2.5,
             markersize=9, label='Fibonacci')
    ax1.plot(Ls, [L+1 for L in Ls], 'k--', alpha=0.5, linewidth=1.5,
             label='L+1 (Sturmian)')
    ax1.set_xlabel('Substring Length L', fontweight='bold', fontsize=11)
    ax1.set_ylabel('# Distinct Substrings', fontweight='bold', fontsize=11)
    ax1.set_title('K-mer Complexity', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Panel 2: Model complexity and sample size
    ax2 = fig.add_subplot(gs[0, 1])
    stats = results['stats']

    x_pos = np.arange(2)
    width = 0.35

    ax2_twin = ax2.twinx()

    bars1 = ax2.bar(x_pos - width/2, [stats['per_phases'], stats['fib_phases']],
                    width, label='Learned Phases', color=['blue', 'red'], alpha=0.7)
    bars2 = ax2_twin.bar(x_pos + width/2, [stats['per_ess'], stats['fib_ess']],
                         width, label='ESS/Phase', color=['cyan', 'orange'], alpha=0.7)

    ax2.set_ylabel('Learned Phases', fontweight='bold', fontsize=11)
    ax2_twin.set_ylabel('ESS/Phase', fontweight='bold', fontsize=11)
    ax2.set_title('Model Complexity & Sample Size', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Periodic', 'Fibonacci'])
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 3: Periodic signal
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(results['periodic']['signal'][:400], 'b-', alpha=0.7, linewidth=0.8)
    ax3.set_title('Periodic (lifted=False)', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Observable x(t)')
    ax3.grid(alpha=0.3)

    # Panel 4: Fibonacci signal
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(results['fibonacci']['signal'][:400], 'r-', alpha=0.7, linewidth=0.8)
    ax4.set_title('Fibonacci (lifted=True)', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Observable x(t)')
    ax4.grid(alpha=0.3)

    # Panel 5: ΔMDL trajectories (spans both columns)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axhline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.6)

    ax5.axhline(stats['per_mean'], color='blue', linestyle=':',
               linewidth=2, alpha=0.7, label=f"Per mean: {stats['per_mean']:.1f}")
    ax5.axhline(stats['fib_mean'], color='red', linestyle=':',
               linewidth=2, alpha=0.7, label=f"Fib mean: {stats['fib_mean']:.1f}")

    ax5.plot(results['periodic']['ts'], results['periodic']['delta'],
             'b-', linewidth=2, label='Periodic (rank-1 truth)', alpha=0.7)
    ax5.plot(results['fibonacci']['ts'], results['fibonacci']['delta'],
             'r-', linewidth=2, label='Fibonacci (rank-2 truth)', alpha=0.7)

    ax5.fill_between(results['fibonacci']['ts'], 0, results['fibonacci']['delta'],
                     where=(results['fibonacci']['delta'] > 0),
                     alpha=0.15, color='red')

    ax5.set_xlabel('Time', fontweight='bold', fontsize=12)
    ax5.set_ylabel('ΔMDL = MDL₁ - MDL₂\n(positive → rank-2 preferred)',
                   fontweight='bold', fontsize=12)
    ax5.set_title('Model Selection: Lift Detection with Enforced Ground Truth',
                  fontweight='bold', fontsize=13)
    ax5.legend(fontsize=10, loc='best')
    ax5.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if results['success']:
        fig.suptitle('✓ HARDENED VALIDATION PASSED',
                     fontsize=16, fontweight='bold', color='green', y=0.995)
    else:
        fig.suptitle('✗ Validation Failed',
                     fontsize=16, fontweight='bold', color='red', y=0.995)

    return fig
