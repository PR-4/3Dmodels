import numpy as np
import pandas as pd

# Model parameters from your code
xmin, xmax = 733233.64, 743883.64
ymin, ymax = 7229517.38, 7239317.38
zmin, zmax = -6800, -5500

# Create 10 random well locations within the model bounds
np.random.seed(42)  # For reproducibility
n_wells = 10
n_points_per_well = 10

# Generate well locations
well_x = np.random.uniform(xmin, xmax, n_wells)
well_y = np.random.uniform(ymin, ymax, n_wells)

# Create synthetic TOC data
toc_data = []
for i in range(n_wells):
    # Generate evenly spaced points between zmin and zmax for each well
    z_points = np.linspace(zmin, zmax, n_points_per_well)
    
    # Generate TOC values between 5 and 15 with some vertical trend
    # Here we'll create a slight trend where TOC generally increases with depth
    base_toc = np.random.uniform(5, 15, n_points_per_well)
    depth_trend = (z_points - zmin) / (zmax - zmin)  # Normalized depth
    toc_values = base_toc + depth_trend * 2  # Add depth trend
    toc_values = np.clip(toc_values, 5, 15)  # Ensure values stay within 5-15 range
    
    # Add data points for this well
    for j, z in enumerate(z_points):
        toc_data.append({
            'Well_ID': f'WELL_{i+1}',
            'X': well_x[i],
            'Y': well_y[i],
            'Z': z,
            'TOC': toc_values[j]
        })

# Create DataFrame with the synthetic data
toc_df = pd.DataFrame(toc_data)

# Save the synthetic data
toc_df.to_csv('synthetic_toc_data.csv', index=False)

# Display first few rows and basic statistics
print("\nSynthetic TOC Data Sample:")
print(toc_df.head(10))
print("\nTOC Statistics:")
print(toc_df['TOC'].describe())
print("\nWell Locations:")
print(toc_df.groupby('Well_ID')[['X', 'Y']].first())