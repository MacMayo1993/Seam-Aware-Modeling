"""Interactive SeamAware Visualizer using Streamlit.

This app lets users explore regime-switching signals and see real-time
comparisons between standard methods and SeamAware's non-orientable approach.

Run with: streamlit run apps/streamlit_visualizer.py
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from seamaware.core import SignFlipAtom, detect_seams_roughness, compute_roughness
from seamaware.models.baselines import FourierBaseline, PolynomialBaseline
from seamaware.utils.synthetic_data import (
    generate_sign_flip_signal,
    generate_hvac_like_signal,
    generate_multi_seam_signal,
    generate_variance_shift_signal
)


# Page config
st.set_page_config(
    page_title="SeamAware Interactive Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color scheme
COLORS = {
    'raw': '#2C3E50',
    'baseline': '#E74C3C',
    'seamaware': '#27AE60',
    'seam_true': '#E67E22',
    'seam_detected': '#3498DB',
}


def generate_ecg_like_signal(length=200, seam_location=0.5, noise_std=0.05, seed=None):
    """Generate ECG-like signal with polarity inversion (common in medical devices)."""
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, 10, length)
    seam_idx = int(length * seam_location)

    # Simplified ECG waveform (P-QRS-T complex)
    signal = np.zeros(length)
    for i in range(length):
        phase = (t[i] % 1.0) * 2 * np.pi
        # P wave
        signal[i] += 0.2 * np.exp(-((phase - 0.3)**2) / 0.05)
        # QRS complex
        signal[i] += 1.0 * np.exp(-((phase - np.pi)**2) / 0.02)
        # T wave
        signal[i] += 0.3 * np.exp(-((phase - 4.5)**2) / 0.1)

    # Add polarity inversion at seam
    signal[seam_idx:] *= -1

    # Add noise
    signal += noise_std * np.random.randn(length)

    return signal, seam_idx


def generate_audio_phase_flip(length=200, seam_location=0.5, noise_std=0.03, seed=None):
    """Generate audio-like signal with phase inversion (common in stereo mismatch)."""
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, 8 * np.pi, length)
    seam_idx = int(length * seam_location)

    # Multi-harmonic audio-like signal
    signal = (0.5 * np.sin(2 * t) +
             0.3 * np.sin(5 * t + 0.5) +
             0.2 * np.sin(11 * t + 1.2))

    # Add phase flip
    signal[seam_idx:] *= -1

    # Add noise
    signal += noise_std * np.random.randn(length)

    return signal, seam_idx


@st.cache_data
def load_example_dataset(example_name, seed=42):
    """Load preloaded example datasets."""

    if example_name == "Sine Wave (Single Seam)":
        signal, seam_idx = generate_sign_flip_signal(
            length=200,
            seam_location=0.5,
            base_signal='sin',
            frequency=2.0,
            noise_std=0.08,
            seed=seed
        )
        true_seams = [seam_idx]
        description = """
        **Classic sign flip example**: Clean sine wave with orientation discontinuity at midpoint.
        Standard Fourier analysis treats this as high-frequency noise, but SeamAware recognizes
        it as a single bit of orientation information.
        """

    elif example_name == "HVAC Regime Switching":
        signal, true_seams = generate_hvac_like_signal(
            length=300,
            on_duration=50,
            off_duration=50,
            noise_std=0.05,
            seed=seed
        )
        description = """
        **Real-world HVAC dynamics**: Heating/cooling cycles create regime switches where
        temperature exponentially approaches setpoint (ON) or drifts away (OFF). Each switch
        can be modeled as a seam where the dynamical model changes. Standard AR models struggle
        at regime boundaries, wasting bits on transition noise.
        """

    elif example_name == "ECG Polarity Inversion":
        signal, seam_idx = generate_ecg_like_signal(
            length=200,
            seam_location=0.5,
            noise_std=0.05,
            seed=seed
        )
        true_seams = [seam_idx]
        description = """
        **Medical device scenario**: ECG signals can experience polarity inversion due to
        lead misplacement or device switching. The cardiac rhythm is identical, but the
        entire waveform is negated. Standard feature extractors see this as completely
        different, but SeamAware recognizes the Z₂ symmetry.
        """

    elif example_name == "Audio Phase Flip":
        signal, seam_idx = generate_audio_phase_flip(
            length=200,
            seam_location=0.5,
            noise_std=0.03,
            seed=seed
        )
        true_seams = [seam_idx]
        description = """
        **Stereo channel mismatch**: When left/right audio channels are out of phase,
        one channel is effectively sign-flipped. This is perceptually similar but wastes
        bits if encoded separately. SeamAware detects and corrects the phase inversion.
        """

    elif example_name == "Multi-Seam (3 Flips)":
        signal, true_seams = generate_multi_seam_signal(
            length=300,
            num_seams=3,
            seam_type='sign_flip',
            base_signal='sin',
            frequency=2.0,
            noise_std=0.06,
            seed=seed
        )
        description = """
        **Multiple regime switches**: Signal with 3 equally-spaced orientation flips.
        Each seam costs ~10 bits (location + orientation), but saves ~140 bits in
        improved fit. Net gain scales with number of seams at high SNR.
        """

    elif example_name == "Variance Shift":
        signal, seam_idx = generate_variance_shift_signal(
            length=200,
            seam_location=0.5,
            base_signal='ar1',
            variance_ratio=4.0,
            noise_std=0.05,
            seed=seed
        )
        true_seams = [seam_idx]
        description = """
        **Variance regime change**: Signal variance quadruples at midpoint (e.g., market
        volatility shift). While not a pure orientation flip, this can be preprocessed
        by normalizing each regime separately, revealing hidden seam structure.
        """

    return signal, true_seams, description


def compute_mdl(signal, reconstruction, num_params, num_seams=0):
    """Compute MDL with proper encoding costs."""
    N = len(signal)
    residuals = signal - reconstruction
    rss = np.sum(residuals**2)

    # BIC-style MDL
    param_cost = num_params * np.log2(N) / 2
    seam_cost = num_seams * (np.log2(N) + 1)  # Location + orientation
    data_cost = (N / 2) * np.log2(rss / N + 1e-10)

    total_mdl = param_cost + seam_cost + data_cost

    return {
        'total': total_mdl,
        'param_cost': param_cost,
        'seam_cost': seam_cost,
        'data_cost': data_cost,
        'residual_variance': np.var(residuals)
    }


def main():
    """Main Streamlit app."""

    # Header
    st.title("🚀 SeamAware Interactive Visualizer")
    st.markdown("""
    **Explore how non-orientable modeling achieves 16-63% better compression on regime-switching signals.**

    Choose a preloaded example or adjust parameters to see real-time comparisons between
    standard methods (Fourier/Polynomial) and SeamAware's orientation-aware approach.
    """)

    st.divider()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")

        # Example selector
        st.subheader("1. Choose Example")
        example = st.selectbox(
            "Dataset",
            [
                "Sine Wave (Single Seam)",
                "HVAC Regime Switching",
                "ECG Polarity Inversion",
                "Audio Phase Flip",
                "Multi-Seam (3 Flips)",
                "Variance Shift"
            ],
            help="Preloaded regime-switching signals with known ground truth"
        )

        seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)

        st.divider()

        # Model parameters
        st.subheader("2. Baseline Model")
        baseline_type = st.radio(
            "Model Type",
            ["Fourier", "Polynomial"],
            help="Standard (orientable) baseline model"
        )

        if baseline_type == "Fourier":
            K = st.slider("K (harmonics)", min_value=3, max_value=20, value=10, step=1)
        else:
            K = st.slider("Degree", min_value=2, max_value=10, value=5, step=1)

        st.divider()

        # Detection parameters
        st.subheader("3. Seam Detection")
        threshold_sigma = st.slider(
            "Threshold σ",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Roughness threshold in standard deviations"
        )

        min_distance = st.slider(
            "Min Distance",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Minimum samples between detected seams"
        )

        show_roughness = st.checkbox("Show Roughness Curve", value=False)

        st.divider()
        st.markdown("---")
        st.markdown("**🔗 Links**")
        st.markdown("[GitHub Repo](https://github.com/MacMayo1993/Seam-Aware-Modeling)")
        st.markdown("[Documentation](https://github.com/MacMayo1993/Seam-Aware-Modeling#readme)")

    # Load data
    signal, true_seams, description = load_example_dataset(example, seed=seed)
    N = len(signal)
    t = np.arange(N)

    # Show description
    with st.expander("ℹ️ About This Example", expanded=False):
        st.markdown(description)

    # Compute baseline
    if baseline_type == "Fourier":
        baseline_model = FourierBaseline(K=K)
    else:
        baseline_model = PolynomialBaseline(degree=K)

    recon_baseline = baseline_model.fit_predict(signal)
    mdl_baseline = compute_mdl(signal, recon_baseline, baseline_model.num_params(), num_seams=0)

    # Detect seams
    detected_seams = detect_seams_roughness(
        signal,
        threshold_sigma=threshold_sigma,
        min_distance=min_distance,
        window=20,
        poly_degree=2
    )

    # Apply seam correction (using first detected seam for now)
    if len(detected_seams) > 0:
        flip_atom = SignFlipAtom()
        # Apply flips at all detected seams
        corrected_signal = signal.copy()
        for seam_idx in detected_seams:
            corrected_signal = flip_atom.apply(corrected_signal, int(seam_idx))

        # Fit model on corrected signal
        if baseline_type == "Fourier":
            seamaware_model = FourierBaseline(K=K)
        else:
            seamaware_model = PolynomialBaseline(degree=K)

        recon_seamaware_corrected = seamaware_model.fit_predict(corrected_signal)

        # Undo flips for visualization
        recon_seamaware = recon_seamaware_corrected.copy()
        for seam_idx in detected_seams:
            recon_seamaware = flip_atom.apply(recon_seamaware, int(seam_idx))

        mdl_seamaware = compute_mdl(signal, recon_seamaware, seamaware_model.num_params(),
                                    num_seams=len(detected_seams))
    else:
        recon_seamaware = recon_baseline
        mdl_seamaware = mdl_baseline.copy()
        st.warning("⚠️ No seams detected! Try lowering the threshold or adjusting parameters.")

    # Compute metrics
    mdl_reduction = mdl_baseline['total'] - mdl_seamaware['total']
    mdl_reduction_pct = 100 * mdl_reduction / mdl_baseline['total']

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Baseline MDL",
            f"{mdl_baseline['total']:.1f} bits",
            help="Standard method (assumes orientable space)"
        )

    with col2:
        st.metric(
            "SeamAware MDL",
            f"{mdl_seamaware['total']:.1f} bits",
            delta=f"-{mdl_reduction:.1f} bits" if mdl_reduction > 0 else f"+{abs(mdl_reduction):.1f} bits",
            delta_color="normal" if mdl_reduction > 0 else "inverse",
            help="Non-orientable method (models seams explicitly)"
        )

    with col3:
        st.metric(
            "Compression Gain",
            f"{mdl_reduction_pct:.1f}%",
            help="Percentage MDL reduction"
        )

    with col4:
        st.metric(
            "Seams Detected",
            f"{len(detected_seams)} / {len(true_seams)}",
            help=f"Detected at: {list(map(int, detected_seams))}"
        )

    st.divider()

    # Main visualization
    st.subheader("📊 Side-by-Side Comparison")

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, height_ratios=[2, 1.5, 1], hspace=0.35, wspace=0.25)

    # ============ TOP LEFT: Baseline ============
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, signal, '-', color=COLORS['raw'], linewidth=1.5, alpha=0.6, label='Observed', zorder=1)
    ax1.plot(t, recon_baseline, '-', color=COLORS['baseline'], linewidth=2.5,
            label=f'{baseline_type} baseline', zorder=2)

    # Mark true seams
    for seam in true_seams:
        ax1.axvline(seam, color=COLORS['seam_true'], linestyle='--',
                   linewidth=2, alpha=0.5)

    # Highlight poor fit regions
    for seam in true_seams:
        seam_window = 15
        if seam < N - seam_window:
            ax1.axvspan(seam, min(N, seam + seam_window),
                       color=COLORS['baseline'], alpha=0.15)

    ax1.set_title(f'❌ Standard Method: {baseline_type}(K={K})\nMDL = {mdl_baseline["total"]:.1f} bits',
                 fontsize=13, fontweight='bold', color=COLORS['baseline'])
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.25, linestyle=':')

    # ============ TOP RIGHT: SeamAware ============
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.plot(t, signal, '-', color=COLORS['raw'], linewidth=1.5, alpha=0.6, label='Observed', zorder=1)
    ax2.plot(t, recon_seamaware, '-', color=COLORS['seamaware'], linewidth=2.5,
            label=f'{baseline_type} + SeamAware', zorder=2)

    # Mark detected seams
    for seam in detected_seams:
        ax2.axvline(seam, color=COLORS['seam_detected'], linestyle='--',
                   linewidth=2.5, alpha=0.7, label='Detected seam')

    # Mark true seams (faint)
    for seam in true_seams:
        ax2.axvline(seam, color=COLORS['seam_true'], linestyle=':',
                   linewidth=1.5, alpha=0.3)

    ax2.set_title(f'✅ SeamAware Method: {baseline_type}(K={K}) + Flip\nMDL = {mdl_seamaware["total"]:.1f} bits (↓{mdl_reduction_pct:.1f}%)',
                 fontsize=13, fontweight='bold', color=COLORS['seamaware'])
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.25, linestyle=':')

    # ============ MIDDLE LEFT: Baseline Residuals ============
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    residuals_baseline = signal - recon_baseline
    ax3.plot(t, residuals_baseline, '-', color=COLORS['baseline'], linewidth=1.5, alpha=0.8)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

    for seam in true_seams:
        ax3.axvline(seam, color=COLORS['seam_true'], linestyle='--',
                   linewidth=2, alpha=0.4)

    ax3.set_title(f'Baseline Residuals (σ² = {mdl_baseline["residual_variance"]:.4f})',
                 fontsize=12, fontweight='bold', color=COLORS['baseline'])
    ax3.set_ylabel('Error', fontsize=10)
    ax3.grid(True, alpha=0.25, linestyle=':')

    # ============ MIDDLE RIGHT: SeamAware Residuals ============
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax3)
    residuals_seamaware = signal - recon_seamaware
    ax4.plot(t, residuals_seamaware, '-', color=COLORS['seamaware'], linewidth=1.5, alpha=0.8)
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

    for seam in detected_seams:
        ax4.axvline(seam, color=COLORS['seam_detected'], linestyle='--',
                   linewidth=2, alpha=0.5)

    ax4.set_title(f'SeamAware Residuals (σ² = {mdl_seamaware["residual_variance"]:.4f})',
                 fontsize=12, fontweight='bold', color=COLORS['seamaware'])
    ax4.set_ylabel('Error', fontsize=10)
    ax4.grid(True, alpha=0.25, linestyle=':')

    # ============ BOTTOM: MDL Breakdown or Roughness ============
    if show_roughness:
        ax5 = fig.add_subplot(gs[2, :])
        roughness = compute_roughness(signal, window=20, poly_degree=2, mode='fast')
        ax5.plot(t, roughness, '-', color='purple', linewidth=2, label='Local roughness')

        # Mark threshold
        threshold = np.mean(roughness) + threshold_sigma * np.std(roughness)
        ax5.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (μ + {threshold_sigma}σ)')

        # Mark detected seams
        for seam in detected_seams:
            ax5.axvline(seam, color=COLORS['seam_detected'], linestyle='--',
                       linewidth=2.5, alpha=0.7)

        # Mark true seams
        for seam in true_seams:
            ax5.axvline(seam, color=COLORS['seam_true'], linestyle=':',
                       linewidth=2, alpha=0.5)

        ax5.set_title('Roughness Analysis (Polynomial Residual Variance)',
                     fontsize=12, fontweight='bold')
        ax5.set_xlabel('Sample Index', fontsize=11)
        ax5.set_ylabel('Roughness', fontsize=10)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.25, linestyle=':')
    else:
        # MDL breakdown comparison
        ax5 = fig.add_subplot(gs[2, :])

        categories = ['Parameter\nCost', 'Seam\nCost', 'Data\nCost']
        baseline_costs = [
            mdl_baseline['param_cost'],
            mdl_baseline['seam_cost'],
            mdl_baseline['data_cost']
        ]
        seamaware_costs = [
            mdl_seamaware['param_cost'],
            mdl_seamaware['seam_cost'],
            mdl_seamaware['data_cost']
        ]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax5.bar(x - width/2, baseline_costs, width, label='Baseline',
                       color=COLORS['baseline'], alpha=0.7, edgecolor='black')
        bars2 = ax5.bar(x + width/2, seamaware_costs, width, label='SeamAware',
                       color=COLORS['seamaware'], alpha=0.7, edgecolor='black')

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9)

        ax5.set_ylabel('Bits', fontsize=11, fontweight='bold')
        ax5.set_title('MDL Cost Breakdown', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(categories, fontsize=10)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.25, axis='y', linestyle=':')

    ax3.set_xlabel('Sample Index', fontsize=11)
    ax4.set_xlabel('Sample Index', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig)

    # Summary table
    st.divider()
    st.subheader("📈 Detailed Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Baseline (Standard)**")
        baseline_df = pd.DataFrame({
            'Component': ['Parameter Cost', 'Seam Cost', 'Data Cost', 'Total MDL', 'Residual Variance'],
            'Value': [
                f"{mdl_baseline['param_cost']:.2f} bits",
                f"{mdl_baseline['seam_cost']:.2f} bits",
                f"{mdl_baseline['data_cost']:.2f} bits",
                f"{mdl_baseline['total']:.2f} bits",
                f"{mdl_baseline['residual_variance']:.6f}"
            ]
        })
        st.dataframe(baseline_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**SeamAware (Non-Orientable)**")
        seamaware_df = pd.DataFrame({
            'Component': ['Parameter Cost', 'Seam Cost', 'Data Cost', 'Total MDL', 'Residual Variance'],
            'Value': [
                f"{mdl_seamaware['param_cost']:.2f} bits",
                f"{mdl_seamaware['seam_cost']:.2f} bits (×{len(detected_seams)} seams)",
                f"{mdl_seamaware['data_cost']:.2f} bits",
                f"{mdl_seamaware['total']:.2f} bits",
                f"{mdl_seamaware['residual_variance']:.6f}"
            ]
        })
        st.dataframe(seamaware_df, use_container_width=True, hide_index=True)

    # Interpretation
    st.divider()
    st.subheader("💡 Interpretation")

    if mdl_reduction > 0:
        st.success(f"""
        ✅ **SeamAware wins!** Saves **{mdl_reduction:.1f} bits ({mdl_reduction_pct:.1f}% reduction)**.

        The compression gain comes primarily from **lower data encoding cost** due to better fit.
        Even though SeamAware pays {mdl_seamaware['seam_cost']:.1f} bits for encoding {len(detected_seams)} seam(s),
        it saves {mdl_baseline['data_cost'] - mdl_seamaware['data_cost']:.1f} bits by achieving
        {mdl_baseline['residual_variance'] / mdl_seamaware['residual_variance']:.2f}× lower residual variance.

        **Why?** By recognizing that the signal lives in non-orientable projective space ℝP^(n-1),
        SeamAware treats orientation flips as topologically trivial (1 bit) instead of modeling
        them as high-frequency noise (hundreds of bits).
        """)
    else:
        st.warning(f"""
        ⚠️ **Baseline wins** by {abs(mdl_reduction):.1f} bits.

        This typically happens when:
        - SNR is below k* ≈ 0.721 (seam encoding cost exceeds benefit)
        - No true seams exist (false positive detection)
        - Noise is too high to reliably detect seams

        Try increasing the random seed to get higher SNR, or choose a different example.
        """)

    # Footer
    st.divider()
    st.markdown("""
    ---
    **Built with ❤️ by the SeamAware team**

    📚 [Read the paper](https://github.com/MacMayo1993/Seam-Aware-Modeling) |
    ⭐ [Star on GitHub](https://github.com/MacMayo1993/Seam-Aware-Modeling) |
    🐛 [Report issues](https://github.com/MacMayo1993/Seam-Aware-Modeling/issues)
    """)


if __name__ == "__main__":
    main()
