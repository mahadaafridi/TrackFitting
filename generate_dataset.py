from helixfitter import HelixFitter
import pandas as pd
import numpy as np

def generate_data(num_helices):
    data = []

    for i in range(num_helices):
        hf = HelixFitter()  
        hf.generate_hits()
        
        chi_squared = hf.fit_helix()
        
        for hit in hf.hits:
            data.append({
                'x': hit[0],
                'y': hit[1],
                'z': hit[2],
                'alpha': hf.original_params[0],
                'kappa': hf.original_params[1],
                'tan_lambda': hf.original_params[2],
                'r': np.sqrt(hit[0]**2 + hit[1]**2),
                'helix_id': i,
                'original_params': hf.og_params(),
                'curve_fit_params': hf.recovered_parameters(),
                'curve_fit_chi_sq': chi_squared 
            })
    
    return pd.DataFrame(data)