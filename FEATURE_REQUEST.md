# Feature Request: Expose Minuit migrad() kwargs in Fitter.fit()

## Summary
Add a `migrad_kws` parameter to `Fitter.fit()` to allow users to pass keyword arguments to `Minuit.migrad()` for better control over convergence behavior.

## Motivation
When fitting challenging multiplet peaks (e.g., closely spaced peaks in gamma spectroscopy), Minuit's default settings may not achieve sufficient convergence, resulting in EDM (Estimated Distance to Minimum) values above the strict threshold. This causes `fitter.success` to be `False` even though the fit visually appears good and is scientifically useful.

Currently, there is no way to adjust Minuit's convergence parameters without modifying becquerel's source code.

## Proposed Solution
Add `migrad_kws` parameter to `Fitter.fit()`:

```python
def fit(self, backend="lmfit", guess=None, limits=None, migrad_kws=None):
    """
    ...
    migrad_kws : dict, optional
        Keyword arguments to pass to Minuit.migrad() for Minuit backends.
        Common options:
        - ncall (int): Maximum number of function calls (default ~10000)
        - iterate (int): Number of iteration cycles (default 5)
        - precision (float): Convergence tolerance. EDM goal ≈ 0.002 × precision
        Only used for minuit-pml backend.
    """
```

## Use Case Example
```python
# Fit multiplet with tighter convergence requirements
fitter.fit(
    backend='minuit-pml',
    guess=initial_params,
    limits=param_limits,
    migrad_kws={'ncall': 100000, 'iterate': 10, 'precision': 5.0}
)
# Now achieves EDM < 0.01 and fitter.success is True
```

## Benefits
1. **Better convergence**: Users can increase iteration limits for difficult fits
2. **Adjustable EDM tolerance**: Control convergence criteria via `precision` parameter
3. **Backward compatible**: Default behavior unchanged when `migrad_kws=None`
4. **Follows existing pattern**: Similar to how `guess` and `limits` are already exposed

## Implementation
Minimal change required (5 lines):
1. Add `migrad_kws=None` parameter to `fit()` signature
2. Pass to `self.result.migrad(**migrad_kws)` after handling None case

See PR #XXX for implementation.

## Related
- iminuit documentation: https://iminuit.readthedocs.io/en/stable/reference.html#iminuit.Minuit.migrad
- Addresses issue where visual fits are good but `success=False` due to EDM > goal

