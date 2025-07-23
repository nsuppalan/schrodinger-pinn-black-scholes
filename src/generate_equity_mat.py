# generate_equity_mat.py

import pandas as pd
import numpy as np
from scipy.io import savemat

def create_equity_mat(input_file, expiry_str, output_mat_file, underlying_symbol='RELIANCE'):
    print(f"📥 Reading equity file: {input_file}")
    df = pd.read_csv(input_file)

    # Drop rows with missing LTP or BuyPrice
    df = df.dropna(subset=['LTP', 'BuyPrice'])

    # Ensure Timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.sort_values('Timestamp').dropna(subset=['Timestamp'])

    # ========== 1. Grids ==========
    # x: log-price vector
    x_vals = np.log(df['LTP'].astype(float)).values
    Nx = len(x_vals)

    # t: normalized time grid
    t_vals = np.linspace(0, 1, Nx)
    Nt = len(t_vals)

    # ========== 2. ψ = Re + i·Im ==========
    uu_re = df['LTP'].values

    # Smoothed rolling return (5-point)
    df['Return'] = df['LTP'].pct_change().rolling(5, min_periods=1).mean()
    uu_im = df['Return'].fillna(0).values

    uu = uu_re + 1j * uu_im
    uu = uu.reshape(Nt, 1).repeat(Nx, axis=1)  # Nt × Nx

    # ========== 3. Feature Engineering ==========
    # Entropy (local noise proxy)
    df['Entropy'] = np.log1p(df['Return'].rolling(5, min_periods=1).std())

    # Volatility Shock: Entropy spike above 85th percentile
    df['VolShock'] = (df['Entropy'] > df['Entropy'].quantile(0.85)).astype(int)

    # OpenGap detection: large LTP - BuyPrice
    df['OpenGap'] = ((df['LTP'] - df['BuyPrice']) > 5).astype(int)

    # EventDay: duplicated timestamps → macro regime
    df['EventDay'] = df.duplicated('Timestamp').astype(int)

    # Stack features into [Nt × F]
    features_matrix = np.stack([
        df['Entropy'].fillna(0).values,
        df['EventDay'].values,
        df['OpenGap'].values,
        df['VolShock'].values
    ], axis=1).astype(np.float32)  # Nt × F

    # Expand to 3D tensor: Nt × Nx × F
    features_tensor = np.repeat(features_matrix[:, np.newaxis, :], Nx, axis=1)

    # ========== 4. Valid Mask ==========
    mask = (df['LTP'].fillna(0).values > 0).astype(np.uint8)
    mask = mask.reshape(Nt, 1).repeat(Nx, axis=1)

    # ========== 5. Save to .mat ==========
    mat_data = {
        'x': x_vals.astype(np.float64),         # Nx vector
        'tt': t_vals.astype(np.float64),        # Nt vector
        'uu': uu.astype(np.complex64),          # Nt × Nx complex surface
        'features': features_tensor,            # Nt × Nx × 4
        'mask': mask,                           # Nt × Nx
        'symbol': underlying_symbol,
        'expiry': expiry_str
    }

    savemat(output_mat_file, mat_data)
    print(f"✅ Equity .mat file saved to {output_mat_file}.\n")


# Example usage (uncomment below to test)
# create_equity_mat("EQT_RELIANCE_20250626.csv", "20250626", "EQT_PINN_TRAIN_DATA_20250626.mat")
