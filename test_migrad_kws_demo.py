"""
Demo script showing the benefit of migrad_kws parameter.

This demonstrates fitting a challenging multiplet where default Minuit
settings result in high EDM (> 0.001) but custom settings achieve 
better convergence.
"""
import numpy as np
import becquerel as bq

# Generate synthetic spectrum with close doublet
np.random.seed(42)
x = np.linspace(500, 600, 1000)
true_peaks = [540, 545]  # Close doublet
y_true = (
    1000 * np.exp(-((x - true_peaks[0]) ** 2) / (2 * 1.5**2)) +
    800 * np.exp(-((x - true_peaks[1]) ** 2) / (2 * 1.5**2)) +
    100  # Background
)
y = np.random.poisson(y_true)
y_unc = np.sqrt(y + 1)

# Build model
model = (
    bq.fitting.GaussModel(prefix='peak1_') + 
    bq.fitting.GaussModel(prefix='peak2_') + 
    bq.fitting.ConstantModel(prefix='bkg_')
)

# Fit with default settings
print("=" * 60)
print("FIT 1: Default Minuit settings")
print("=" * 60)
fitter1 = bq.Fitter(model, x=x, y=y, y_unc=y_unc)
guess1 = {
    'peak1_amp': 1000, 'peak1_mu': 540, 'peak1_sigma': 1.5,
    'peak2_amp': 800, 'peak2_mu': 545, 'peak2_sigma': 1.5,
    'bkg_c': 100
}
fitter1.fit(backend='minuit-pml', guess=guess1)
print(f"Success: {fitter1.success}")
if hasattr(fitter1.result, 'fmin'):
    print(f"EDM: {fitter1.result.fmin.edm:.2e}")
    print(f"EDM goal: {fitter1.result.fmin.edm_goal:.2e}")

# Fit with custom migrad_kws for better convergence
print("\n" + "=" * 60)
print("FIT 2: Custom migrad settings (ncall=50000, tol=10)")
print("=" * 60)
fitter2 = bq.Fitter(model, x=x, y=y, y_unc=y_unc)
fitter2.fit(
    backend='minuit-pml', 
    guess=guess1,
    migrad_kws={'ncall': 50000, 'iterate': 10, 'tol': 10.0}
)
print(f"Success: {fitter2.success}")
if hasattr(fitter2.result, 'fmin'):
    print(f"EDM: {fitter2.result.fmin.edm:.2e}")
    print(f"EDM goal: {fitter2.result.fmin.edm_goal:.2e}")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
if hasattr(fitter1.result, 'fmin') and hasattr(fitter2.result, 'fmin'):
    edm_improvement = fitter1.result.fmin.edm / fitter2.result.fmin.edm
    print(f"EDM improvement: {edm_improvement:.1f}x lower with custom settings")
    print(f"Fit 1 success: {fitter1.success}")
    print(f"Fit 2 success: {fitter2.success}")
