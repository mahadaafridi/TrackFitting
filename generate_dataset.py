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
                'helix_id': i,
                'x': hit[0],
                'y': hit[1],
                'z': hit[2],
                'true_alpha': hf.original_params[0],
                'true_kappa': hf.original_params[1],
                'true_tan_lambda': hf.original_params[2],
                'r': np.sqrt(hit[0]**2 + hit[1]**2),
                'curve_fit_alpha': hf.recovered_params[0],
                'curve_fit_kappa': hf.recovered_params[1],
                'curve_fit_tan_lambda': hf.recovered_params[2],
                'curve_fit_chi_sq': chi_squared 
            })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    
    dataset = generate_data(10000)
    dataset.to_csv("hit_data.csv")
