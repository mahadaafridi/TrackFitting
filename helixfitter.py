import numpy as np
from scipy.optimize import curve_fit

class HelixFitter:
    def __init__(self):
        self.hits = None
        self.t_values = None
        self.original_params = None
        self.recovered_params = None
        self.chi_sq = None

    def helix_params(self, t, d0, phi0, omega, z0, tan_lambda):
        x = (d0 + 1/omega) * np.cos(phi0 + omega * t)
        y = (d0 + 1/omega) * np.sin(phi0 + omega * t)
        z = z0 + t * tan_lambda
        return np.array([x, y, z]).T

    def generate_hits(self):
        # randomly generate helix parameters
        # seed for constant results 
        # np.random.seed(1)
        d0 = np.random.uniform(-1, 1)
        phi0 = np.random.uniform(0, 2 * np.pi)
        omega = np.random.uniform(0.1, 1)
        z0 = np.random.uniform(-1, 1)
        tan_lambda = np.random.uniform(0.1, 1)
        
        self.original_params = [d0, phi0, omega, z0, tan_lambda]
        
        #generate angles for hits
        radii = np.arange(1, 11)
        self.t_values = radii / omega
        
        # generate hits
        self.hits = self.helix_params(self.t_values, d0, phi0, omega, z0, tan_lambda)
        
    def helix_fitting_func(self, t, d0, phi0, omega, z0, tan_lambda):
        return self.helix_params(t, d0, phi0, omega, z0, tan_lambda).flatten()

    def fit_helix(self):
        # curve fit    
        popt, _ = curve_fit(self.helix_fitting_func, self.t_values, self.hits.flatten(), p0=self.original_params)
        self.recovered_params = popt

        # calc chi squared
        residuals = self.hits.flatten() - self.helix_fitting_func(self.t_values, *self.recovered_params)

        self.chi_sq = np.sum((residuals)**2)

    def chi_squared(self):
        return self.chi_sq

    def recovered_paramaters(self):
        return self.recovered_params
    
    def og_params(self):
        return self.original_params

if __name__ == "__main__":
    # Create an instance of the HelixFitter class
    file = open("output.txt", "w")
   
    for i in range(5):
        file.write(f"Loop {i}\n")
        hf = HelixFitter()

        hf.generate_hits()

        hf.fit_helix()
        file.write(f'Original Params {hf.og_params()}\n')
        file.write(f'recovered Params {hf.recovered_paramaters()}\n')
        file.write(f'chi squared {hf.chi_squared()}\n')
        file.write("\n")