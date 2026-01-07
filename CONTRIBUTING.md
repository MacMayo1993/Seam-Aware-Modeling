# Contributing to SeamAware

Thank you for your interest in contributing to SeamAware! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MacMayo1993/Seam-Aware-Modeling.git
   cd Seam-Aware-Modeling
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup:**
   ```bash
   pytest tests/ -v
   ```

5. **Set up pre-commit hooks (recommended):**
   ```bash
   pip install pre-commit
   pre-commit install

   # Run hooks manually on all files
   pre-commit run --all-files
   ```

## Code Style

We follow standard Python conventions:

- **Formatting**: [Black](https://black.readthedocs.io/) (line length 88)
- **Import sorting**: [isort](https://pycqa.github.io/isort/)
- **Linting**: [flake8](https://flake8.pycqa.org/)
- **Type hints**: Python 3.9+ type annotations throughout

Format your code before committing:
```bash
black seamaware tests
isort seamaware tests
flake8 seamaware tests --max-line-length=88
```

**Pre-commit hooks** will automatically run these checks before each commit. If a check fails, the commit is blocked until you fix the issue.

### Type Checking with mypy

We use **gradual typing** - new and refactored code should include type hints:

```bash
# Check types on strict modules
mypy seamaware/core/detection.py --python-version=3.9
```

**Currently enforced modules:**
- `seamaware/core/detection.py`
- `tests/test_performance.py`

When adding type hints:
- Use `from typing import Any, List, Optional, Tuple`
- For numpy: `signal: np.ndarray`
- For returns: `def function() -> ReturnType:`
- For flexible kwargs: `**kwargs: Any`

## Testing

All new features must include tests:

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test complete workflows
- **Validation tests**: Verify mathematical properties (e.g., k* convergence)

Run tests:
```bash
pytest tests/ -v                    # All tests
pytest tests/ -m "not slow"         # Skip slow tests
pytest tests/test_mdl.py -v         # Specific module
```

## Documentation

- **Docstrings**: Use NumPy-style docstrings for all public functions
- **Type hints**: Required for all function signatures
- **Examples**: Include examples in docstrings when helpful

Example:
```python
def compute_k_star() -> float:
    """
    Compute the universal seam-aware modeling constant.

    k* = 1 / (2·ln 2) ≈ 0.7213

    Returns:
        float: The k* constant

    Examples:
        >>> k_star = compute_k_star()
        >>> 0.721 < k_star < 0.722
        True
    """
    return 1.0 / (2.0 * np.log(2))
```

## Pull Request Process

1. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, atomic commits

3. **Add tests** for new functionality

4. **Run the test suite** and ensure all tests pass

5. **Format your code:**
   ```bash
   black seamaware tests
   isort seamaware tests
   ```

6. **Push your branch** and open a pull request

7. **Write a clear PR description** explaining:
   - What problem does this solve?
   - How does it solve it?
   - Are there any breaking changes?

## Commit Messages

Use clear, descriptive commit messages:

```
Add k* validation with Monte Carlo sampling

- Implement validate_k_star_convergence() function
- Add comprehensive tests for SNR threshold detection
- Document theoretical derivation in docs/theory.md

Closes #42
```

## Areas for Contribution

We welcome contributions in:

- **Core algorithms**: Improve seam detection, add new flip atoms
- **Validation**: More rigorous tests of theoretical predictions
- **Performance**: Optimize for large signals, parallelize detection
- **Applications**: Real-world examples (EEG, finance, etc.)
- **Documentation**: Tutorials, examples, theory explanations
- **Visualizations**: Better plotting utilities

## Questions?

- Open an [issue](https://github.com/MacMayo1993/Seam-Aware-Modeling/issues) for bugs or feature requests
- Start a [discussion](https://github.com/MacMayo1993/Seam-Aware-Modeling/discussions) for questions or ideas

## Code of Conduct

Be respectful and constructive in all interactions. We're here to advance mathematical understanding and build useful tools together.
