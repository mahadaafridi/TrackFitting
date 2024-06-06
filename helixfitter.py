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
        
        d0 = np.random.uniform(-1, 1)
        phi0 = np.random.uniform(0, 2 * np.pi)
        omega = np.random.uniform(0.1, 1)
        z0 = np.random.uniform(-1, 1)
        tan_lambda = np.random.uniform(0.1, 1)
        
        self.original_params = [d0, phi0, omega, z0, tan_lambda]
        
        radii = np.linspace(1, 10, 10)
        self.t_values = []
        for r in radii:
            t, best_omega = self.find_t_and_omega_for_r(r, d0, phi0, omega)
            self.t_values.append(t)
            omega = best_omega  

        self.t_values = np.array(self.t_values)
        self.hits = self.helix_params(self.t_values, d0, phi0, omega, z0, tan_lambda)

    def find_t_and_omega_for_r(self, r, d0, phi0, omega, max_iter=1000):
        #returns best_t and best_omega, which will return values that will give the closest "r" value
        best_t = None
        best_omega = None
        smallest_diff = float('inf')
        
        omega_values = np.linspace(0.1, 1, max_iter)
        #iterates throguh 1,000,000 times, need to reduce this 
        for omega in omega_values:
            t_values = np.linspace(0, 10, max_iter)  
            for t in t_values:
                x = (d0 + 1/omega) * np.cos(phi0 + omega * t)
                y = (d0 + 1/omega) * np.sin(phi0 + omega * t)
                r_calculated = np.sqrt(x**2 + y**2)
                diff = np.abs(r_calculated - r)

                if diff < smallest_diff:
                    smallest_diff = diff
                    best_t = t
                    best_omega = omega

        return best_t, best_omega

    def helix_fitting_func(self, t, d0, phi0, omega, z0, tan_lambda):
        return self.helix_params(t, d0, phi0, omega, z0, tan_lambda).flatten()

    def fit_helix(self):

        popt, _ = curve_fit(self.helix_fitting_func, self.t_values, self.hits.flatten(), p0=self.original_params)
        self.recovered_params = popt


    def calc_chi_squared(self, actual_hits):
        #calcs the chi squared comparing the simulation to the actual mesurements 
        
        estimated_hits = self.helix_params(self.t_values, **self.recovered_parameters)
        residuals = estimated_hits.flatten() - actual_hits.flatten()
        self.chi_sq = np.sum((residuals) ** 2)
        return self.chi_sq


    def recovered_parameters(self):
        return self.recovered_params
    
    def og_params(self):
        return self.original_params
if __name__ == "__main__":
    # Create an instance of the HelixFitter class
    file = open("output.txt", "w")
   
    for i in range(1):
        file.write(f"Loop {i}\n")
        hf = HelixFitter()

        hf.generate_hits()

        hf.fit_helix()
        file.write(f'Original Params {hf.og_params()}\n')
        file.write(f'recovered Params {hf.recovered_parameters()}\n')
        file.write("\n")
    
    file.close()