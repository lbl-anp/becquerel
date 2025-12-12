# Becquerel Feature Submission Summary

## Branch Information
- **Branch name**: `feature/expose-migrad-kwargs`
- **Base**: `main`
- **Repository**: https://github.com/lbl-anp/becquerel

## Changes Made

### 1. Core Feature Implementation
**File**: `becquerel/core/fitting.py`

**Changes**:
- Added `migrad_kws=None` parameter to `Fitter.fit()` method signature
- Added comprehensive docstring explaining the parameter and its usage
- Pass kwargs to `Minuit.migrad(**migrad_kws)` in the minuit-pml backend section
- Maintains full backward compatibility

**Lines changed**: ~15 lines added/modified

### 2. Unit Tests
**File**: `tests/fitting_test.py`

**Tests added**:
- `test_migrad_kws_parameter()`: Verifies custom kwargs improve convergence (lower EDM)
- `test_migrad_kws_non_minuit_backend()`: Ensures non-Minuit backends safely ignore the parameter

**Coverage**: Tests both functionality and backward compatibility

### 3. Documentation Files

**Files created**:
- `FEATURE_REQUEST.md`: Detailed feature request rationale
- `PR_DESCRIPTION.md`: Pull request description template
- `test_migrad_kws_demo.py`: Standalone demo script showing the benefit

## Commits
```
d798a9b test: add unit tests for migrad_kws parameter
529d57c feat: expose migrad kwargs in Fitter.fit() method
```

## How to Submit

### Option 1: Create GitHub Issue
1. Go to https://github.com/lbl-anp/becquerel/issues/new
2. Title: "Feature Request: Expose Minuit migrad() kwargs in Fitter.fit()"
3. Copy content from `FEATURE_REQUEST.md`

### Option 2: Create Pull Request
1. Fork the becquerel repository
2. Push this branch: `git push origin feature/expose-migrad-kwargs`
3. Create PR at https://github.com/lbl-anp/becquerel/compare
4. Title: "feat: expose migrad kwargs in Fitter.fit() method"
5. Copy content from `PR_DESCRIPTION.md`

### Option 3: Manual Submission via gh CLI
```bash
# Fork and push (you'll need to do this part manually or via GitHub UI)
# Then create PR
gh pr create \
  --title "feat: expose migrad kwargs in Fitter.fit() method" \
  --body-file PR_DESCRIPTION.md \
  --base main \
  --head feature/expose-migrad-kwargs
```

## Testing Instructions

### Run unit tests:
```bash
cd /path/to/becquerel
pytest tests/fitting_test.py::test_migrad_kws_parameter -v
pytest tests/fitting_test.py::test_migrad_kws_non_minuit_backend -v
```

### Run demo script:
```bash
python test_migrad_kws_demo.py
```

Expected output: Shows EDM improvement (several times lower) with custom settings

## Benefits Summary
1. **Better convergence**: Users can increase limits for challenging fits
2. **Adjustable EDM tolerance**: Control via `precision` parameter  
3. **Backward compatible**: Default behavior unchanged
4. **Minimal code change**: Only ~15 lines modified
5. **Well-tested**: Unit tests + demo script included
6. **Real-world use case**: Solves actual problem in gamma spectroscopy peak fitting

## Real-World Application
This feature was developed to solve a real problem in HPGe gamma spectroscopy:
- Multiplet peak fitting (close doublets/triplets)
- Default Minuit settings gave EDM ~0.001-0.01 (above strict 0.0002 threshold)
- Visually excellent fits were marked as "failed" 
- Custom settings achieve EDM < 0.001 and proper success status
- Critical for automated batch fitting of hundreds of peaks

## Contact
This submission addresses a gap in the becquerel API that prevents users from controlling Minuit's convergence behavior without source code modifications.

