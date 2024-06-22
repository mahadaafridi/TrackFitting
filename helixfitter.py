import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt    

class HelixFitter:
    """class for generating hits along helical trajectory"""
    def __init__(self):
        self.hits = None
        self.phi_values = None
        self.original_params = None
        self.recovered_params = None
        self.chi_sq = None
        self.last_phi = 0

    def simplified_helix_params(self, phi, alpha, kappa, tan_lambda):
        """Calculate helix parameters based on input

        Args:
            phi (array): The parameter along the helix.
            alpha (float): Scaling factor related to the particle's momentum.
            kappa (float): Curvature parameter.
            tan_lambda (float): Tangent of the dip angle.

        Returns:
            np.array: Numpy array of helix coordinates.
        """
        x = (alpha / kappa) * (np.cos(phi) - 1)
        y = (alpha / kappa) * np.sin(phi)
        z = (alpha / kappa) * tan_lambda * phi
        return np.array([x, y, z]).T

    def find_phi_for_r(self, r, alpha, kappa, tan_lambda, max_iter=1000):
        """
        Find the phi value that corresponds to a given radius.

        Args:
            r (float): Desired radius.
            alpha (float): Scaling factor related to the particle's momentum.
            kappa (float): Curvature parameter.
            tan_lambda (float): Tangent of the dip angle.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.

        Returns:
            float: The best phi value for the given radius.
        """
        best_phi = None
        smallest_diff = float('inf')

        phi_values = np.linspace(self.last_phi, 100, max_iter)
        for phi in phi_values:
            x, y, _ = self.simplified_helix_params(phi, alpha, kappa, tan_lambda)
            r_calculated = np.sqrt(x**2 + y**2)
            diff = np.abs(r_calculated - r)

            if diff < smallest_diff:
                smallest_diff = diff
                best_phi = phi

        self.last_phi = best_phi
        return best_phi

    def generate_hits(self):
        """Generates the hits along helical trajectory."""
        alpha = np.random.uniform(0.1, 2)
        kappa = np.random.uniform(0.01, .05)
        tan_lambda = np.random.uniform(0.1, 1)

        self.original_params = [alpha, kappa, tan_lambda]
        radii = np.linspace(1, 10, 10)
        self.phi_values = []
        for r in radii:
            phi = self.find_phi_for_r(r, alpha, kappa, tan_lambda, max_iter=1000)
            self.phi_values.append(phi)
        self.phi_values = np.array(self.phi_values)
        self.hits = self.simplified_helix_params(self.phi_values, alpha, kappa, tan_lambda)
        
    def simplified_helix_fitting_func(self, phi, alpha, kappa, tan_lambda):
        """Helix fitting function used in curve_fit.

        Args:
            phi (array): The parameter along the helix.
            alpha (float): Scaling factor related to the particle's momentum.
            kappa (float): Curvature parameter.
            tan_lambda (float): Tangent of the dip angle.

        Returns:
            np.array: Numpy array.
        """
        return self.simplified_helix_params(phi, alpha, kappa, tan_lambda).flatten()

    def fit_helix(self):
        """Fits helix to generated hits. Calculates the parameters."""
        popt, _ = curve_fit(self.simplified_helix_fitting_func, self.phi_values, self.hits.flatten(), p0=self.original_params)
        self.recovered_params = popt

    def calc_chi_squared(self, actual_hits):
        """Calculates and returns chi squared of the generated hits and actual hits.

        Args:
            actual_hits (np.array): Array of actual hits.

        Returns:
            float: Chi-squared value.
        """
        residuals = self.hits.flatten() - actual_hits.flatten()
        self.chi_sq = np.sum((residuals)**2)
        return self.chi_sq

    def recovered_paramaters(self):
        """Returns recovered parameters.

        Returns:
            np.array: Recovered parameters.
        """
        return self.recovered_params
    
    def og_params(self):
        """Returns original parameters.

        Returns:
            np.array: Original parameters.
        """       
        return self.original_params

if __name__ == "__main__":
    file = open("Output.txt", "w")
   
    hf = HelixFitter()

    hf.generate_hits()

    hf.fit_helix()
    file.write(f'Original Params {hf.og_params()}\n')
    file.write(f'Recovered Params {hf.recovered_paramaters()}\n')
    file.write("Generated Hits\n")
    
    for i, hit in enumerate(hf.hits):
        file.write((f"Hit {i} : {hit}\n"))
        # file.write(f'Phi: {hf.phi_values[i]}\n')
        x_squared = hit[0] ** 2
        y_squared  = hit[1] ** 2
        z_squared = hit[2] ** 2
        file.write(f"R: {(x_squared + y_squared) ** .5}\n")

    file.write("\n")
    
    file.close()
