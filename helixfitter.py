import numpy as np
from scipy.optimize import curve_fit

class HelixFitter:
    """Class for generating hits along helical tragectory"""
    def __init__(self):
        self.hits = None
        self.t_values = None
        self.original_params = None
        self.recovered_params = None
        self.chi_sq = None

    def helix_params(self, t, d0, phi0, omega, z0, tan_lambda):
        """calc helix parameters based on input

        Args:
            t (array) 
            d0 (float)
            phi0 (float)
            omega (float)
            z0 (float)
            tan_lambda (float)

        Returns:
            _type_: numpy array of helix coordinates
        """
        x = (d0 + 1/omega) * np.cos(phi0 + omega * t)
        y = (d0 + 1/omega) * np.sin(phi0 + omega * t)
        z = z0 + t * tan_lambda
        return np.array([x, y, z]).T

    def find_t_for_r(self, r, d0, phi0, max_iter=100):
        """Find and return time param based on a given radial distance

        Args:
            r (float)
            d0 (float)
            phi0 (float)
            max_iter (int, optional)

        Returns:
            _type_: float
        """
        best_t = None
        smallest_diff = float('inf')

        omega_values = np.linspace(0.1, 1, max_iter)
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

        return best_t

    def generate_hits(self):
        """generates the hits along helical trajectory.
        """
        d0 = np.random.uniform(-1, 1)
        phi0 = np.random.uniform(0, 2 * np.pi)
        omega = np.random.uniform(0.1, 1)
        z0 = np.random.uniform(-1, 1)
        tan_lambda = np.random.uniform(0.1, 1)
        
        self.original_params = [d0, phi0, omega, z0, tan_lambda]
        
        radii = np.linspace(1, 10, 10)
        self.t_values = []
        for r in radii:
            t = self.find_t_for_r(r, d0, phi0)
            self.t_values.append(t)

        self.t_values = np.array(self.t_values)
        self.hits = self.helix_params(self.t_values, d0, phi0, omega, z0, tan_lambda)
        
   
    def helix_fitting_func(self, t, d0, phi0, omega, z0, tan_lambda):
        """helix fitting function used in curve_fit

        Args:
            t (array) 
            d0 (float)
            phi0 (float)
            omega (float)
            z0 (float)
            tan_lambda (float)

        Returns:
            _type_: numpy array
        """
        return self.helix_params(t, d0, phi0, omega, z0, tan_lambda).flatten()

    def fit_helix(self):
        """fits helix to generated hits. Calculates the parameters
        """
        popt, _ = curve_fit(self.helix_fitting_func, self.t_values, self.hits.flatten(), p0=self.original_params)
        self.recovered_params = popt
    def calc_chi_squared(self, actual_hits):
        """calcs and returns chi squared of the generated hits and actual hits 

        Args:
            actual_hits (np.array)

        Returns:
            _type_: float 
        """
        residuals = self.hits.flatten() - actual_hits.flatten()
        self.chi_sq = np.sum((residuals)**2)

        return self.chi_sq

    def recovered_paramaters(self):
        """returns recovered params

        Returns:
            _type_: numpy array 
        """
        return self.recovered_params
    
    def og_params(self):
        """returns original params

        Returns:
            _type_: numpy array 
        """       
        return self.original_params

if __name__ == "__main__":
    file = open("output.txt", "w")
   
    for i in range(5):
        file.write(f"Loop {i}\n")
        hf = HelixFitter()

        hf.generate_hits()

        hf.fit_helix()
        file.write(f'Original Params {hf.og_params()}\n')
        file.write(f'recovered Params {hf.recovered_paramaters()}\n')
        file.write("generated hits\n")
        for i, hit in enumerate(hf.hits):
            r_calculated = np.sqrt(hit[0]**2 + hit[1]**2)
            file.write(f"hit {i}: x = {hit[0]}, y = {hit[1]}, z = {hit[2]}, r = {r_calculated}\n")
 
        file.write("\n")
    
    file.close()