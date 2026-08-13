# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:22:14 2026
- Run topolevel cluster TFCE statistics

@author: Rahul Venugopal
"""

#%% Import libraries
import numpy as np
import mne
from mne.channels import find_ch_adjacency
from limo_tfce_replicate import limo_tfce_python
import os
import matplotlib.pyplot as plt
import matplotlib

# Custom functions
def compute_mne_adjacency(channel_names):
    """Compute adjacency matrix using MNE's Delaunay triangulation."""
    info = mne.create_info(ch_names=channel_names, sfreq=100, ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'))
    adjacency, _ = find_ch_adjacency(info, ch_type='eeg')
    return adjacency.toarray().astype(float)

def yuen_t_test_2samp_vectorized(X, Y, tr=0.1):
    """
    Vectorized Robust Yuen's t-test returning t-values for all channels.
    X and Y should be 2D arrays of shape (n_subjects, n_channels).
    """
    # Ensure inputs are 2D arrays
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    
    nx, n_channels = X.shape
    ny, _ = Y.shape
    
    gx = int(tr * nx)
    gy = int(tr * ny)
    
    # Sort along the subjects axis (axis=0)
    X_sort = np.sort(X, axis=0)
    Y_sort = np.sort(Y, axis=0)
    
    # --- 1. Trimmed Means ---
    # Handle the case where trimming might result in an empty slice
    X_trim = X_sort[gx:nx-gx, :] if gx > 0 else X_sort
    Y_trim = Y_sort[gy:ny-gy, :] if gy > 0 else Y_sort
    mx = np.mean(X_trim, axis=0)
    my = np.mean(Y_trim, axis=0)
    
    # --- 2. Winsorized Variances ---
    X_winsor = X_sort.copy()
    if gx > 0:
        X_winsor[:gx+1, :] = X_winsor[gx, :]
        X_winsor[nx-gx-1:, :] = X_winsor[nx-gx-1, :]
        
    Y_winsor = Y_sort.copy()
    if gy > 0:
        Y_winsor[:gy+1, :] = Y_winsor[gy, :]
        Y_winsor[ny-gy-1:, :] = Y_winsor[ny-gy-1, :]
        
    # np.var with ddof=1 is equivalent to your * N / (N-1) logic
    vx = np.var(X_winsor, axis=0, ddof=1)
    vy = np.var(Y_winsor, axis=0, ddof=1)
    
    # --- 3. Standard Error Calculation ---
    hx = nx - 2 * gx
    hy = ny - 2 * gy
    
    da = vx * (nx - 1) / (hx * (hx - 1))
    db = vy * (ny - 1) / (hy * (hy - 1))
    
    se = np.sqrt(da + db)
    
    # --- 4. T-Value Output with Zero-Division Protection ---
    t_values = np.zeros(n_channels)
    mask = se > 0  # Identify channels where standard error is not exactly zero
    t_values[mask] = (mx[mask] - my[mask]) / se[mask]
    
    return t_values
   
# Create the colour palette
colors = [(0.2706, 0.4588, 0.7059),
          (0.5686, 0.7490, 0.8588),
          (0.8784, 0.9529, 0.9725),
          (1.0000, 1.0000, 0.7490),
          (0.9961, 0.8784, 0.5647),
          (0.9882, 0.5529, 0.3490),
          (0.8431, 0.1882, 0.1529)]

from matplotlib.colors import ListedColormap

cmap_becp = ListedColormap(colors)

#%% See the neighbours

def interactive_adjacency_plot(info, adjacency, channels):
    """
    Creates an interactive MNE topomap where hovering over an electrode
    dynamically draws lines to its spatial neighbors.
    """
    # 1. Generate the base MNE sensor plot
    fig = mne.viz.plot_sensors(info, kind='topomap', show_names=True, show=False)
    ax = fig.axes[0]
    
    # Extract the 2D display coordinates MNE generated for the sensors
    coords = ax.collections[0].get_offsets()
    
    # Dictionary to keep track of the lines and dots we draw so we can delete them when the mouse moves
    dynamic_elements = {'lines': [], 'seed': None, 'neighbors': None}
    
    # 2. Define the hover event function
    def on_hover(event):
        # Only proceed if the mouse is inside the plot area
        if event.inaxes != ax: 
            return
            
        # Check if the mouse hit one of the sensor dots
        hit, props = ax.collections[0].contains(event)
        if not hit: 
            return
            
        # Get the index and name of the hovered point
        idx = props["ind"][0]
        seed_ch = info.ch_names[idx]
        
        # Ensure the channel exists in your adjacency matrix list
        if seed_ch not in channels: 
            return
            
        # --- Clear previous drawings ---
        for line in dynamic_elements['lines']: 
            line.remove()
        dynamic_elements['lines'].clear()
        
        if dynamic_elements['seed']: 
            dynamic_elements['seed'].remove()
            dynamic_elements['seed'] = None
            
        if dynamic_elements['neighbors']: 
            dynamic_elements['neighbors'].remove()
            dynamic_elements['neighbors'] = None
            
        # --- Calculate and draw new neighbors ---
        adj_idx = channels.index(seed_ch)
        neighbor_indices = np.where(adjacency[adj_idx] > 0)[0]
        
        seed_x, seed_y = coords[idx]
        n_coords = []
        
        for n_idx in neighbor_indices:
            neighbor_ch = channels[n_idx]
            info_n_idx = info.ch_names.index(neighbor_ch)
            nx, ny = coords[info_n_idx]
            n_coords.append((nx, ny))
            
            # Draw the connecting line
            line, = ax.plot([seed_x, nx], [seed_y, ny], color='red', linewidth=1.5, zorder=1)
            dynamic_elements['lines'].append(line)
            
        # Highlight the neighbor dots and the seed dot
        if n_coords:
            nx_vals, ny_vals = zip(*n_coords)
            dynamic_elements['neighbors'] = ax.scatter(nx_vals, ny_vals, color='orange', s=60, zorder=2)
            
        dynamic_elements['seed'] = ax.scatter([seed_x], [seed_y], color='red', s=100, zorder=3)
        
        # Update title and refresh the canvas
        ax.set_title(f"Hovering: {seed_ch} ({len(neighbor_indices)} neighbors)")
        fig.canvas.draw_idle()

    # 3. Connect the hover event listener to the figure
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    
    print("Interactive plot generated! Hover over the electrodes to see their neighbors.")
    plt.show()
    
    return fig

#%% Load data
data = np.load('data_array_all.npy')
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

comparisons = [
    (0, 1, "ATT", "ATTMW"),
    (1, 2, "ATTMW", "MW"),
    (0, 2, "ATT", "MW")
]

N_PERMUTATIONS = 1000

channels = ['AF3','AF4','AF7','AF8','AFz','C1', 'C2', 'C3', 'C4',
 'C5', 'C6', 'CP1','CP2', 'CP3','CP4','CP5','CP6','CPz',
 'Cz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
'FC1', 'FC2', 'FC3', 'FC4','FC5','FC6', 'FT10','FT7','FT8',
'FT9','Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2',
 'P3', 'P4', 'P5', 'P6', 'P7', 'P8','PO3','PO4','PO7',
'PO8','POz', 'Pz', 'T7', 'T8', 'TP10','TP7','TP8','TP9']

feature_names = ['Delta_Irasa_Mean', 'Delta_Irasa_CV', 'Theta_Irasa_Mean', 'Theta_Irasa_CV',
'ThetaAlpha_Irasa_Mean', 'ThetaAlpha_Irasa_CV', 'Alpha_Irasa_Mean',
'Alpha_Irasa_CV', 'Beta1_Irasa_Mean', 'Beta1_Irasa_CV', 'Beta2_Irasa_Mean',
'Beta2_Irasa_CV', 'Gamma1_Irasa_Mean', 'Gamma1_Irasa_CV', 'intercept_Irasa_Mean',
'intercept_Irasa_CV', 'slope_Irasa_Mean', 'slope_Irasa_CV', 'ACW_Mean',
'ACW_CV']

adjacency = compute_mne_adjacency(channels)
info = mne.create_info(ch_names=channels, sfreq=100, ch_types='eeg')
info.set_montage(mne.channels.make_standard_montage('standard_1020'))

# See the interactive neighbours
# Force Matplotlib to use the interactive Qt backend
matplotlib.use('Qt5Agg')
fig = interactive_adjacency_plot(info, adjacency, channels)

# Get the min max values for all features

print("Pre-calculating global vmin and vmax for all features...")
num_features = len(feature_names)
num_groups = data.shape[0] 

global_vmins = np.zeros(num_features)
global_vmaxs = np.zeros(num_features)

for feature in range(num_features):
    feature_means = []
    
    # Loop through all groups to find the global min/max for this feature
    for grp_idx in range(num_groups):
        grp_data = data[grp_idx, :, :, feature]
        
        # Apply the exact same NaN masking we use later
        mask = ~np.isnan(grp_data).any(axis=1)
        grp_data_clean = grp_data[mask]
        
        if grp_data_clean.size > 0:
            grp_mean = grp_data_clean.mean(axis=0)
            feature_means.append(grp_mean)
            
    if feature_means:
        global_vmins[feature] = np.min(feature_means)
        global_vmaxs[feature] = np.max(feature_means)
    else:
        global_vmins[feature] = -1 # Fallback if entirely NaN
        global_vmaxs[feature] = 1

#%% Run the stats

main_fig = plt.figure(figsize=(15, 6))

for grp1_idx, grp2_idx, grp1_name, grp2_name in comparisons:

    print(f"\n>>> Starting {grp1_name} vs {grp2_name}")

    data_comparison = data[[grp1_idx, grp2_idx], :, :, :]
    # Shape: (2, 41, 32, 28)

    for feature in range(data_comparison.shape[3]):

        print(f"  Processing feature: {feature_names[feature]}...")

        X1 = data_comparison[0, :, :, feature]
        mask1 = ~np.isnan(X1).any(axis=1)
        X1 = X1[mask1]

        X2 = data_comparison[1, :, :, feature]
        mask2 = ~np.isnan(X2).any(axis=1)
        X2 = X2[mask2]

        if X1.size == 0 or X2.size == 0:
            print(f"  Skipping {grp1_name} vs {grp2_name}, feature {feature_names[feature]}: no data after NaN removal.")
            continue

        t_obs = yuen_t_test_2samp_vectorized(X1,X2)
        tfce_obs = limo_tfce_python(t_obs, adjacency)

        X_comb = np.concatenate([X1, X2], axis=0)
        n1 = len(X1)
        max_tfce_null = []

        for _ in range(N_PERMUTATIONS):
            idx = np.random.permutation(len(X_comb))
            
            # Slice the combined array for the two pseudo-groups and pass them directly
            X1_perm = X_comb[idx[:n1], :]
            X2_perm = X_comb[idx[n1:], :]
            
            t_perm = yuen_t_test_2samp_vectorized(X1_perm, X2_perm)
            max_tfce_null.append(np.max(np.abs(limo_tfce_python(t_perm, adjacency))))
        
        max_tfce_null_arr = np.array(max_tfce_null)
        p_corr = np.mean(max_tfce_null_arr >= np.abs(tfce_obs)[:, None], axis=1)
        sig_mask = p_corr <= 0.05
        
        # Clear the reused figure entirely
        main_fig.clf()
        
        # Re-create the subplots for this iteration
        ax1 = main_fig.add_subplot(1, 3, 1)
        ax2 = main_fig.add_subplot(1, 3, 2)
        ax3 = main_fig.add_subplot(1, 3, 3)
        axes = [ax1, ax2, ax3]

        X1_mean = X1.mean(axis=0)
        X2_mean = X2.mean(axis=0)
        
        # Lookup the global vmin and vmax per feature
        vmin = global_vmins[feature]
        vmax = global_vmaxs[feature]

        im1, _ = mne.viz.plot_topomap(X1_mean, info, axes=axes[0], cmap=cmap_becp,
                             vlim=(vmin, vmax), show=False, contours=0,
                             sensors=True, outlines='head', sphere=None)
        axes[0].set_title(f"{grp1_name} {feature_names[feature]}",
                     fontsize=12, fontweight='bold', pad=25)
        cb_ghost = main_fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cb_ghost.ax.set_visible(False)

        im2, _ = mne.viz.plot_topomap(X2_mean, info, axes=axes[1], cmap=cmap_becp,
                             vlim=(vmin, vmax), show=False, contours=0,
                             sensors=True, outlines='head', sphere=None)
        axes[1].set_title(f"{grp2_name} {feature_names[feature]}",
                     fontsize=12, fontweight='bold', pad=25)
        cb = main_fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cb.set_label('Feature values',
                     rotation=90, labelpad=20, fontsize=14, fontweight='bold')
        cb.ax.tick_params(labelsize=8)

        im, cn = mne.viz.plot_topomap(data=t_obs, pos=info, mask=sig_mask,
            mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                             linewidth=0, markersize=5),
            cmap=cmap_becp, contours=0, axes=axes[2], show=False)
        axes[2].set_title(f"{feature_names[feature]} (TFCE Statistics) {grp1_name} (vs) {grp2_name}",
                     fontsize=12, fontweight='bold', pad=25)
        cb = main_fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cb.set_label('T-values', rotation=90, labelpad=20, fontsize=14, fontweight='bold')
        cb.ax.tick_params(labelsize=8)

        main_fig.tight_layout()
        out_fn = os.path.join(output_dir, f"{grp1_name}_vs_{grp2_name}_{feature_names[feature]}.png")
        main_fig.savefig(out_fn, dpi=600, bbox_inches='tight')

        print(f"  Saved: {grp1_name}_vs_{grp2_name}_{feature_names[feature]}.png")

print("\n All comparisons complete!")
