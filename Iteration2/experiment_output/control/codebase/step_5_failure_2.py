# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np

def min_image(dx):
    """
    Applies the minimum image convention for a periodic domain of size 1.0
    centered at 0 (i.e., [-0.5, 0.5]).
    
    Parameters:
        dx (np.ndarray): The displacement vector.
        
    Returns:
        np.ndarray: The shortest path displacement vector.
    """
    return dx - np.round(dx)

if __name__ == '__main__':
    data_dir = 'data/'
    
    print("Loading tracer data...")
    data = np.load(os.path.join(data_dir, 'tracer_data.npz'))
    trajectories = data['trajectories'][:5000]
    V_LS = data['V_LS'][:5000]
    
    num_tracers, num_snapshots, _ = trajectories.shape
    
    V_LS_mag = np.linalg.norm(V_LS, axis=2)
    mean_V_LS_mag = np.mean(V_LS_mag)
    print("Mean magnitude of V_LS: " + str(np.round(mean_V_LS_mag, 6)))
    
    pos_0 = trajectories[:, 0, :]
    disp = min_image(trajectories - pos_0[:, None, :])
    
    V_LS_mag_safe = np.where(V_LS_mag == 0, 1e-10, V_LS_mag)
    
    dot_product = np.sum(disp * V_LS, axis=2)
    d_parallel = dot_product / V_LS_mag_safe
    d_parallel_sq = d_parallel**2
    
    disp_sq = np.sum(disp**2, axis=2)
    d_perp_sq = disp_sq - d_parallel_sq
    d_perp_sq = np.maximum(d_perp_sq, 0.0)
    
    MSD_parallel = np.mean(d_parallel_sq, axis=0)
    MSD_perp = np.mean(d_perp_sq, axis=0)
    
    MSD_perp_safe = np.where(MSD_perp == 0, 1e-10, MSD_perp)
    lambda_t = MSD_parallel / MSD_perp_safe
    
    print("\n--- Dynamic Anisotropy Assessment Results ---")
    print("Final MSD_parallel: " + str(np.round(MSD_parallel[-1], 6)))
    print("Final MSD_perp: " + str(np.round(MSD_perp[-1], 6)))
    print("Final lambda(t): " + str(np.round(lambda_t[-1], 6)))
    print("Max lambda(t): " + str(np.round(np.max(lambda_t), 6)))
    print("Min lambda(t) (t>0): " + str(np.round(np.min(lambda_t[1:]), 6)))
    print("---------------------------------------------\n")
    
    out_file = os.path.join(data_dir, 'step_5_results.npz')
    np.savez(out_file, 
             MSD_parallel=MSD_parallel, 
             MSD_perp=MSD_perp, 
             lambda_t=lambda_t)
    print("Results saved to " + out_file)