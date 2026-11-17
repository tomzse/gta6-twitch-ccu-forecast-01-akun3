# GTA VI Twitch CCU forecast — akun 3 (tomzse / quiet-otter-43)

Competition: gta6-twitch-ccu-forecast-01. Forecast Twitch concurrent viewers in
the Grand Theft Auto VI category, hours 2-24 of release day (2026-11-19 UTC),
as five quantiles per hour.

Model: per-hour OLS log regression on rolling-midnight comparables, SHARP
spread. For each hour h, fit log(viewers_h) = a_h + b_h * log(viewers_1);
predict at GTA VI's hour-1 print; spread = residual std scaled by 0.75, then
clamped to [0.20, 1.00]. Quantiles: q_tau(h) = exp(pred_h + z_tau * s_h).

Leave-one-out backtest over the 12 rolling comparables: pinball loss 0.251 vs
the house baseline's 0.357 (30% better) and vs the plain OLS entry's 0.247.

Run: `MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py`
