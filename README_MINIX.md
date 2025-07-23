\# 📘 Schrödinger PINN for Options \& Equity Pricing  

\*\*Quantum-Inspired Option \& Equity Modeling Using Physics-Informed Neural Networks\*\*



---



\## 🧠 Project Overview



This project builds a \*\*production-ready data preparation pipeline\*\* to train \*\*Schrödinger-based Physics-Informed Neural Networks (PINNs)\*\* for modeling:



\- 📈 Option price surfaces (multi-strike, multi-expiry)

\- 📉 Equity price evolution and directional forecasting



All input data is \*\*real historical tick-level data\*\*. The PINN models are aligned with \*\*quantum finance principles\*\* and governed by a nonlinear Schrödinger equation to capture realistic, evolving market behaviors like spread, smile, regime shocks, and entropic effects.



---



\## 📂 Project Structure



project/

│

├── create\_option\_mat.py # Creates .mat file for options data (multi-expiry)

├── create\_equity\_mat.py # Creates .mat file for equity data

├── OPT\_RELIANCE\_20250626.csv # Real tick options data

├── EQT\_RELIANCE\_20250626.csv # Real tick equity data

├── README.md # This file



---



\## 🔧 Preprocessing Logic \& Features



\### ✅ `create\_option\_mat.py` (Options)



| Feature / Signal              | Included | Description |

|------------------------------|----------|-------------|

| `Re(ψ)` from LTP             | ✅       | Real part of complex wavefunction — option price surface |

| `Im(ψ)` from Return          | ✅       | Smoothed rolling return — proxy for bid-ask asymmetry |

| `x` = log-moneyness          | ✅       | log(S / K), normalized |

| `t` = DTE normalized         | ✅       | time-to-expiry / max(DTE) |

| `OI`                         | ✅       | Open interest from tick data |

| `Delta` (BS computed)        | ✅       | Fallback if not in CSV |

| `IV` (Black-Scholes implied) | ✅       | Brent method inversion |

| IV Spline Interpolation      | ✅       | For missing strikes/maturities |

| `Entropy`                    | ✅       | log(1 + std of rolling returns) |

| `VolShock`                  | ✅       | Binary flag if entropy > 85th percentile |

| `OpenGap`                   | ✅       | If (LTP - BuyPrice) > threshold |

| `EventDay`                  | ✅       | If timestamp duplicated (macro action proxy) |

| `features` tensor            | ✅       | 3D tensor (Nt × Nx × F) |

| `mask`                       | ✅       | Binary where LTP > 0 |

| `ψ = Re + i·Im`             | ✅       | Complex-valued option surface |

| Output Format                | ✅       | `.mat` file: `{'x', 'tt', 'uu', 'features', 'mask', 'symbol', 'expiry'}` |



---



\### ✅ `create\_equity\_mat.py` (Equity)



| Feature / Signal        | Included | Description |

|------------------------|----------|-------------|

| `Re(ψ)` from LTP       | ✅       | Equity price surface (real) |

| `Im(ψ)` from Return    | ✅       | Smoothed return captures directionality |

| `x` = log(S)           | ✅       | Log of LTP, normalized |

| `t` = linspace(0,1,N)  | ✅       | Artificial time grid |

| `Entropy`              | ✅       | Local speckle density |

| `VolShock`            | ✅       | Entropy spike detection |

| `OpenGap`             | ✅       | Jump flag if gap from BuyPrice |

| `EventDay`            | ✅       | Duplicate timestamp = event |

| `features` tensor      | ✅       | 2D tensor (Nt × F) |

| `mask`                 | ✅       | Binary where LTP > 0 |

| `ψ = Re + i·Im`       | ✅       | Equity surface for PINN forecasting |

| Output Format          | ✅       | `.mat` file: `{'x', 'tt', 'uu', 'features', 'mask', 'symbol', 'expiry'}` |



---



\## 🎯 Use Cases



\### ✅ Options Pricing via PINN



\- Compute \*\*current fair value surface\*\*

\- Forecast \*\*evolution over short timeframes\*\*

\- Capture \*\*bid-ask asymmetry\*\*, \*\*entropy\*\*, and \*\*volatility shocks\*\*

\- Model \*\*regime flags\*\* and \*\*event impact\*\*



\### ✅ Equity Forecasting via PINN



\- Learn \*\*price wavefunction over short horizon\*\*

\- Build \*\*standalone equity signals\*\*

\- Enable \*\*delta hedging\*\* and \*\*timing entry\*\*

\- Train on \*\*directional volatility and entropic regime shifts\*\*



---



\## 🧪 Output `.mat` Format (Both Scripts)



Each `.mat` file contains:



```matlab

{

&nbsp; 'x':        \[Nx]             # Spatial grid (log-moneyness or log-price)

&nbsp; 'tt':       \[Nt]             # Normalized time grid

&nbsp; 'uu':       \[Nt x Nx]        # Complex-valued wavefunction (Re + i·Im)

&nbsp; 'features': \[Nt x Nx x F]    # Feature tensor per point (3D options / 2D equity)

&nbsp; 'mask':     \[Nt x Nx]        # Valid data mask (LTP > 0)

&nbsp; 'symbol':   'RELIANCE'       # Asset

&nbsp; 'expiry':   '20250626'       # Expiry tag

}

📌 Notes \& Future Extensions

✅ Implied Volatility Surface Spline fitting added.



✅ Black-Scholes Delta \& IV computed if missing.



✅ Smooth grid interpolation and fillna(0) safeguard included.



🔁 Easily looped across multiple expiry dates.



🔮 Can be integrated with PINN training pipeline directly.



🧠 Aligned with Jack Sarkissian’s quantum microstructure research and FMI framework.



👨‍💻 Author \& Acknowledgements

Developed by Naveen, Quant AI Researcher

Guided by quantum-informed PDE theory, market microstructure research, and real-world deployment goals.

