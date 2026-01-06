"""
Baseline models for comparison with seam-aware methods.

These models assume data lives in orientable spaces (ℝⁿ or ℂⁿ)
and provide MDL benchmarks for evaluating seam-aware improvements.
"""

import numpy as np
from typing import Optional
import warnings


class PolynomialBaseline:
    """
    Polynomial regression baseline.

    Fits a polynomial of specified degree to the entire signal,
    assuming no regime changes.

    Args:
        degree: Polynomial degree (0 = constant, 1 = linear, etc.)

    Examples:
        >>> baseline = PolynomialBaseline(degree=2)
        >>> t = np.linspace(0, 1, 100)
        >>> signal = t**2 + 0.1 * np.random.randn(100)
        >>> prediction = baseline.fit_predict(signal)
        >>> baseline.num_params()
        3
    """

    def __init__(self, degree: int = 2):
        """
        Initialize polynomial baseline.

        Args:
            degree: Polynomial degree

        Raises:
            ValueError: If degree < 0
        """
        if degree < 0:
            raise ValueError(f"Degree must be non-negative, got {degree}")

        self.degree = degree
        self.coeffs: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        """
        Fit polynomial to data.

        Args:
            data: Input signal
        """
        t = np.arange(len(data))
        self.coeffs = np.polyfit(t, data, self.degree)

    def predict(self, length: Optional[int] = None) -> np.ndarray:
        """
        Generate predictions.

        Args:
            length: Prediction length (default: use fitted length)

        Returns:
            Polynomial predictions

        Raises:
            RuntimeError: If fit() hasn't been called
        """
        if self.coeffs is None:
            raise RuntimeError("Must call fit() before predict()")

        if length is None:
            # Use degree to infer reasonable length
            length = 100

        t = np.arange(length)
        return np.polyval(self.coeffs, t)

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step.

        Args:
            data: Input signal

        Returns:
            Fitted values (same length as data)
        """
        self.fit(data)
        t = np.arange(len(data))
        return np.polyval(self.coeffs, t)

    def num_params(self) -> int:
        """
        Return number of parameters.

        Returns:
            degree + 1 (polynomial coefficients)
        """
        return self.degree + 1

    def __repr__(self) -> str:
        """String representation."""
        return f"PolynomialBaseline(degree={self.degree})"


class FourierBaseline:
    """
    Fourier series baseline (sum of sinusoids).

    Models signal as sum of K harmonics:
        f(t) = a₀ + Σₖ [aₖ·cos(2πkt/N) + bₖ·sin(2πkt/N)]

    Assumes signal is periodic or smooth, with no seams.

    Args:
        K: Number of Fourier components (harmonics)

    Examples:
        >>> baseline = FourierBaseline(K=5)
        >>> t = np.linspace(0, 4*np.pi, 200)
        >>> signal = np.sin(t) + 0.5*np.sin(2*t)
        >>> prediction = baseline.fit_predict(signal)
        >>> baseline.num_params()
        11  # 1 DC + 2*5 harmonics
    """

    def __init__(self, K: int = 10):
        """
        Initialize Fourier baseline.

        Args:
            K: Number of harmonics

        Raises:
            ValueError: If K < 1
        """
        if K < 1:
            raise ValueError(f"K must be positive, got {K}")

        self.K = K
        self.coeffs: Optional[np.ndarray] = None
        self.n: Optional[int] = None

    def fit(self, data: np.ndarray) -> None:
        """
        Fit Fourier series using FFT.

        Args:
            data: Input signal
        """
        n = len(data)
        self.n = n

        # Compute FFT
        fft_result = np.fft.fft(data)

        # Keep only first K components (plus DC)
        fft_truncated = np.zeros_like(fft_result)
        fft_truncated[0] = fft_result[0]  # DC component
        fft_truncated[1 : self.K + 1] = fft_result[1 : self.K + 1]  # Positive freqs
        fft_truncated[-self.K :] = fft_result[-self.K :]  # Negative freqs

        self.coeffs = fft_truncated

    def predict(self, length: Optional[int] = None) -> np.ndarray:
        """
        Generate predictions.

        Args:
            length: Prediction length (default: use fitted length)

        Returns:
            Fourier predictions

        Raises:
            RuntimeError: If fit() hasn't been called
        """
        if self.coeffs is None:
            raise RuntimeError("Must call fit() before predict()")

        if length is None:
            length = self.n if self.n is not None else 100

        # Inverse FFT
        if length == self.n:
            return np.real(np.fft.ifft(self.coeffs))
        else:
            # For different length, reconstruct from coefficients
            # This is approximate and may not be exact
            warnings.warn("Predicting at different length than fitted, results approximate")
            return np.real(np.fft.ifft(self.coeffs[:length]))

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step.

        Args:
            data: Input signal

        Returns:
            Fitted values (same length as data)
        """
        self.fit(data)
        return np.real(np.fft.ifft(self.coeffs))

    def num_params(self) -> int:
        """
        Return number of parameters.

        Returns:
            1 + 2*K (DC + K sine/cosine pairs)
        """
        return 1 + 2 * self.K

    def __repr__(self) -> str:
        """String representation."""
        return f"FourierBaseline(K={self.K})"


class ARBaseline:
    """
    Autoregressive (AR) baseline.

    Models signal as linear combination of past p values:
        x(t) = c + Σᵢ φᵢ·x(t-i) + ε(t)

    Requires statsmodels package.

    Args:
        order: AR order p (number of lags)

    Examples:
        >>> baseline = ARBaseline(order=5)
        >>> signal = np.cumsum(np.random.randn(200))  # Random walk
        >>> prediction = baseline.fit_predict(signal)
        >>> baseline.num_params()
        6  # p coefficients + intercept
    """

    def __init__(self, order: int = 5):
        """
        Initialize AR baseline.

        Args:
            order: AR order (number of lags)

        Raises:
            ValueError: If order < 1
        """
        if order < 1:
            raise ValueError(f"Order must be positive, got {order}")

        self.order = order
        self.model = None
        self._fitted_values = None

    def fit(self, data: np.ndarray) -> None:
        """
        Fit AR model.

        Args:
            data: Input signal

        Raises:
            ImportError: If statsmodels not installed
        """
        try:
            from statsmodels.tsa.ar_model import AutoReg
        except ImportError:
            raise ImportError(
                "ARBaseline requires statsmodels. "
                "Install with: pip install statsmodels"
            )

        # Fit AR model
        self.model = AutoReg(data, lags=self.order).fit()
        self._fitted_values = self.model.fittedvalues

    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Generate out-of-sample predictions.

        Args:
            steps: Number of steps to forecast

        Returns:
            Forecasted values

        Raises:
            RuntimeError: If fit() hasn't been called
        """
        if self.model is None:
            raise RuntimeError("Must call fit() before predict()")

        return self.model.forecast(steps=steps)

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and return in-sample predictions.

        Args:
            data: Input signal

        Returns:
            Fitted values

        Notes:
            AR models lose the first p values to initialization.
            We pad with the mean to match input length.
        """
        self.fit(data)

        # Pad fitted values to match input length
        # (AR loses first p values)
        fitted = np.copy(self._fitted_values)

        # Prepend mean for first p values
        if len(fitted) < len(data):
            n_missing = len(data) - len(fitted)
            padding = np.full(n_missing, np.mean(data[:self.order]))
            fitted = np.concatenate([padding, fitted])

        return fitted

    def num_params(self) -> int:
        """
        Return number of parameters.

        Returns:
            order + 1 (AR coefficients + intercept)
        """
        return self.order + 1

    def __repr__(self) -> str:
        """String representation."""
        return f"ARBaseline(order={self.order})"


class ConstantBaseline:
    """
    Constant (mean) baseline.

    Simplest possible model: f(t) = mean(x).

    Useful for testing and as a lower bound.

    Examples:
        >>> baseline = ConstantBaseline()
        >>> signal = np.random.randn(100) + 5.0
        >>> prediction = baseline.fit_predict(signal)
        >>> np.allclose(prediction, 5.0, atol=0.5)
        True
    """

    def __init__(self) -> None:
        """Initialize constant baseline."""
        self.mean_value: Optional[float] = None

    def fit(self, data: np.ndarray) -> None:
        """
        Fit constant (compute mean).

        Args:
            data: Input signal
        """
        self.mean_value = float(np.mean(data))

    def predict(self, length: int = 100) -> np.ndarray:
        """
        Generate predictions.

        Args:
            length: Prediction length

        Returns:
            Array of constant values

        Raises:
            RuntimeError: If fit() hasn't been called
        """
        if self.mean_value is None:
            raise RuntimeError("Must call fit() before predict()")

        return np.full(length, self.mean_value)

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and predict.

        Args:
            data: Input signal

        Returns:
            Array of mean value (same length as data)
        """
        self.fit(data)
        return np.full(len(data), self.mean_value)

    def num_params(self) -> int:
        """
        Return number of parameters.

        Returns:
            1 (mean value)
        """
        return 1

    def __repr__(self) -> str:
        """String representation."""
        if self.mean_value is not None:
            return f"ConstantBaseline(mean={self.mean_value:.3f})"
        return "ConstantBaseline()"
