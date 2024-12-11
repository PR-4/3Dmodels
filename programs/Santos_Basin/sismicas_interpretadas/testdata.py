import pandas as pd
import numpy as np

# Create sample well data with TOC values
well_data = {
    'WELL_ID': [
        'WELL-1', 'WELL-1', 'WELL-1',
        'WELL-2', 'WELL-2', 'WELL-2',
        'WELL-3', 'WELL-3', 'WELL-3',
        'WELL-4', 'WELL-4', 'WELL-4'
    ],
    'X': [
        431215.581, 431215.581, 431215.581,
        435015.751, 435015.751, 435015.751,
        440015.974, 440015.974, 440015.974,
        445016.198, 445016.198, 445016.198
    ],
    'Y': [
        7798428.01, 7798428.01, 7798428.01,
        7798428.211, 7798428.211, 7798428.211,
        7798428.477, 7798428.477, 7798428.477,
        7798428.742, 7798428.742, 7798428.742
    ],
    'Z': [
        -2784.95, -3728.01, -4126.05,
        -3020.15, -4145.45, -4400.62,
        -3299.72, -4623.05, -4619.65,
        -3442.88, -4727.90, -4683.26
    ],
    'FORMATION': [
        'MAASTRICHTIANO', 'CENOMANIANO', 'CENOMANIANO',
        'MAASTRICHTIANO', 'CENOMANIANO', 'CENOMANIANO',
        'MAASTRICHTIANO', 'CENOMANIANO', 'CENOMANIANO',
        'MAASTRICHTIANO', 'CENOMANIANO', 'CENOMANIANO'
    ],
    'TOC': [
        np.nan, 5.2, 3.8,
        np.nan, 8.4, 4.2,
        np.nan, 12.6, 6.8,
        np.nan, 9.8, 4.5
    ]
}

# Create DataFrame
df = pd.DataFrame(well_data)

# Save to CSV
df.to_csv('well_data_with_toc.csv', index=False)

print("Example well data format:")
print(df.head(6))