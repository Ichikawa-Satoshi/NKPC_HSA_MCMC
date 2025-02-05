import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math
import warnings
warnings.simplefilter('ignore')


def plot_z(year,z_mean,z_lower,z_upper,Y):
    plt.figure(figsize=(6, 6))
    plt.plot(year, z_mean, label="Mean of z_t", color="blue", marker="o")
    plt.fill_between(year, z_lower, z_upper, color="blue", alpha=0.2, label="95% Credible Interval")
    plt.title("Estimated Latent Variable z_t with 95% Credible Interval")
    plt.xlabel("Time")
    plt.ylabel("z_t")
    plt.legend()
    plt.grid(True)
    plt.show()
    corrcoef = np.corrcoef(z_mean, Y)[0, 1]
    plt.figure(figsize=(5, 4))
    plt.scatter(Y, z_mean, alpha=0.7)
    plt.xlabel("Y (Output Gap)")
    plt.ylabel("z_mean (Latent Variable)")
    plt.title(f"Correlation: {corrcoef:.3f}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()