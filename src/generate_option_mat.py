# generate_option_mat.py

import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime
from scipy.interpolate import CloughTocher2DInterpolator, griddata
from scipy.optimize import brentq
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type):
    """Compute BS price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-8)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'CE':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # PE
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(S, K, T, r, market_price, option_type='CE'):
    """Estimate IV using Brent's method."""
    try:
        return brentq(
            lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - market_price,
            1e-6, 5.0, maxiter=100, full_output=False
        )
    except Exception:
        return np.nan

def black_scholes_delta(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-8)
    return norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1)

def compute_entropy(signal, window=5):
    return np.log(1 + signal.rolling(window).std().fillna(0))

def create_options_mat(opt_file, expiry, output_mat_file, eqt_file=None):
    df = pd.read_csv(opt_file)
    df['LTP'] = pd.to_numeric(df['LTP'], errors='coerce')
    df['BuyPrice'] = pd.to_numeric(df['BuyPrice'], errors='coerce')
    df['StrikePrice'] = pd.to_numeric(df['StrikePrice'], errors='coerce')
    df['OpenInterest'] = pd.to_numeric(df['OpenInterest'], errors='coerce')
    df.dropna(subset=['LTP', 'StrikePrice'], inplace=True)

    # Parse timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)

    # Optional EQT merge
    if eqt_file:
        eqt = pd.read_csv(eqt_file)
        eqt['Timestamp'] = pd.to_datetime(eqt['Timestamp'], errors='coerce')
        eqt = eqt[['Timestamp', 'LTP']].rename(columns={'LTP': 'Spot'})
        df = pd.merge_asof(df.sort_values('Timestamp'), eqt.sort_values('Timestamp'), on='Timestamp')
        df['Spot'].fillna(method='ffill', inplace=True)
    else:
        df['Spot'] = df['StrikePrice']  # fallback

    df['Underlying'] = df['Spot']
    df['log_moneyness'] = np.log(df['StrikePrice'] / df['Spot'].replace(0, np.nan))
    df['DTE'] = pd.to_numeric(df['DTE'], errors='coerce')
    df['t_normalized'] = df['DTE'] / df['DTE'].max()

    df['Return'] = df['LTP'].pct_change().fillna(0)
    df['ImPsi'] = df['Return'].rolling(3).mean().fillna(0)
    df['Entropy'] = compute_entropy(df['LTP'])
    df['VolShock'] = (df['Entropy'] > df['Entropy'].quantile(0.85)).astype(int)
    df['OpenGap'] = ((df['LTP'] - df['BuyPrice']) > 5).astype(int)
    df['EventDay'] = df.duplicated(subset=['Timestamp']).astype(int)

    # Compute IV if not present
    if 'IV' not in df.columns:
        df['IV'] = df.apply(lambda row: implied_volatility(
            row['Spot'], row['StrikePrice'], row['DTE'] / 365,
            0.05, row['LTP'], row['OptionsType']
        ), axis=1)

    # Fill missing IVs via spline/grid
    iv_valid = df[['log_moneyness', 't_normalized', 'IV']].dropna()
    if len(iv_valid) > 0:
        iv_interp = CloughTocher2DInterpolator(
            np.vstack((iv_valid['log_moneyness'], iv_valid['t_normalized'])).T,
            iv_valid['IV']
        )
        df['IV_fitted'] = iv_interp(df['log_moneyness'], df['t_normalized'])
        df['IV'] = df['IV'].fillna(df['IV_fitted'])

    # Delta
    df['Delta'] = df.apply(lambda row: black_scholes_delta(
        row['Spot'], row['StrikePrice'], row['DTE'] / 365, 0.05, row['IV'], row['OptionsType']
    ), axis=1)

    # Space-time grids
    x_grid = np.sort(df['log_moneyness'].unique())
    t_grid = np.sort(df['t_normalized'].unique())
    grid_x, grid_t = np.meshgrid(x_grid, t_grid)

    def interpolate(field):
        d = df[['log_moneyness', 't_normalized', field]].dropna()
        f = CloughTocher2DInterpolator(np.vstack((d['log_moneyness'], d['t_normalized'])).T, d[field])
        return f(grid_x, grid_t)

    uu_re = interpolate('LTP')
    uu_im = interpolate('ImPsi')
    uu = uu_re + 1j * uu_im

    features = np.stack([
        interpolate('OpenInterest'),
        interpolate('Delta'),
        interpolate('Entropy'),
        interpolate('VolShock'),
        interpolate('OpenGap'),
        interpolate('EventDay'),
        interpolate('IV')
    ], axis=-1)

    mask = (uu_re > 0).astype(np.float32)

    mat_data = {
        'x': x_grid.astype(np.float32),
        'tt': t_grid.astype(np.float32),
        'uu': uu.astype(np.complex64),
        'features': features.astype(np.float32),
        'mask': mask,
        'symbol': 'RELIANCE',
        'expiry': expiry
    }

    sio.savemat(output_mat_file, mat_data)
    print(f"[✓] Saved enhanced options .mat: {output_mat_file}")
