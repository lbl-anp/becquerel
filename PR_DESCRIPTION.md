# Expose migrad kwargs in Fitter.fit() method

## Description
This PR adds a `migrad_kws` parameter to `Fitter.fit()` to allow users to pass keyword arguments to `Minuit.migrad()` for better control over convergence behavior when using the Minuit backend.

## Motivation
When fitting challenging multiplet peaks (e.g., closely spaced peaks in gamma spectroscopy), Minuit's default convergence settings may be insufficient. This results in:
- EDM (Estimated Distance to Minimum) above the strict threshold
- `fitter.success = False` even though the fit is visually good and scientifically useful
- No way for users to adjust convergence parameters without modifying source code

## Changes
- Added `migrad_kws` parameter to `Fitter.fit()` method signature
- Pass kwargs to `Minuit.migrad()` when backend is `minuit-pml`
- Added comprehensive docstring explaining common parameters:
  - `ncall`: Maximum function calls
  - `iterate`: Number of iteration cycles
  - `precision`: Convergence tolerance (EDM goal ≈ 0.002 × precision)

## Example Usage
```python
# Before: fit fails with EDM above threshold
fitter.fit(backend='minuit-pml', guess=params)
# fitter.success = False, EDM = 0.005

# After: achieve tighter convergence
fitter.fit(
    backend='minuit-pml',
    guess=params,
    migrad_kws={'ncall': 100000, 'iterate': 10, 'precision': 5.0}
)
# fitter.success = True, EDM = 0.0008
```

## Testing
Included demo script (`test_migrad_kws_demo.py`) showing:
- Default fit with EDM above threshold
- Custom settings achieving better convergence
- Quantified improvement in EDM

## Backward Compatibility
✅ Fully backward compatible
- Default behavior unchanged when `migrad_kws=None` or not provided
- Existing code continues to work without modification

## Checklist
- [x] Code follows project style guidelines
- [x] Added comprehensive docstrings
- [x] Included example/demo script
- [x] Backward compatible
- [ ] Added unit tests (if required by maintainers)
- [ ] Updated CHANGELOG.md (if applicable)

## Related Issues
Addresses feature request for exposing Minuit convergence parameters.

## Implementation Details
The change is minimal (5 lines) and follows the existing pattern used for `guess` and `limits` parameters:

```python
# Add parameter
def fit(self, backend="lmfit", guess=None, limits=None, migrad_kws=None):
    ...
    
# Use it
if migrad_kws is None:
    migrad_kws = {}
self.result.migrad(**migrad_kws)
```

This implementation:
- Is consistent with existing API design
- Requires minimal code changes
- Has no performance impact when not used
- Provides maximum flexibility for users

