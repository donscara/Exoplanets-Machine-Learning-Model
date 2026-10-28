"""Kepler's Third Law, revisited.

Companion to "Exoplanets Machine Learning Project.ipynb".
Compares the original regression (orbital period vs. the system's distance
from Earth) with the physically meaningful one (orbital period vs. the
planet's distance from its own star, on log scales).

Run from the repo folder:  python kepler_revisited.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("oec.csv")
exo = df[~df["ListsPlanetIsOn"].str.contains("Solar System", na=False)]

# 1) Original approach: distance from our Sun (parsecs) -> orbital period (days),
#    with missing values filled by linear interpolation, as in the notebook.
orig = df.drop(columns=["LongitudeDeg", "AscendingNodeDeg", "AgeGyr"])
num = orig.select_dtypes("number").columns
orig[num] = orig[num].interpolate(method="linear")
X, y = orig[["DistFromSunParsec"]], orig["PeriodDays"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
m1 = LinearRegression().fit(X_tr, y_tr)
p1 = m1.predict(X_te)
print("ORIGINAL  period ~ distance from Earth")
print(f"  RMSE = {mean_squared_error(y_te, p1) ** 0.5:,.0f} days   test R^2 = {r2_score(y_te, p1):.3f}")

# 2) Revisited: Kepler's Third Law says P^2 is proportional to a^3,
#    so log(P) = 1.5 * log(a) + c. Use only measured values (no imputation).
k = exo.dropna(subset=["PeriodDays", "SemiMajorAxisAU"])
k = k[(k["PeriodDays"] > 0) & (k["SemiMajorAxisAU"] > 0)]
X = np.log10(k[["SemiMajorAxisAU"]])
y = np.log10(k["PeriodDays"])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
m2 = LinearRegression().fit(X_tr, y_tr)
p2 = m2.predict(X_te)
pct_err = np.median(np.abs(10 ** p2 - 10 ** y_te) / 10 ** y_te) * 100
print("REVISITED log(period) ~ log(semi-major axis)")
print(f"  planets = {len(k):,}   slope = {m2.coef_[0]:.2f} (theory: 1.50)")
print(f"  test R^2 = {r2_score(y_te, p2):.3f}   median period error = {pct_err:.1f}%")

# 3) Adding the host star's mass (theory: P ~ a^1.5 * M^-0.5)
k2 = k.dropna(subset=["HostStarMassSlrMass"])
k2 = k2[k2["HostStarMassSlrMass"] > 0]
X = np.log10(k2[["SemiMajorAxisAU", "HostStarMassSlrMass"]])
y = np.log10(k2["PeriodDays"])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
m3 = LinearRegression().fit(X_tr, y_tr)
p3 = m3.predict(X_te)
pct_err = np.median(np.abs(10 ** p3 - 10 ** y_te) / 10 ** y_te) * 100
print("REVISITED + host star mass")
print(f"  planets = {len(k2):,}   coefficients = {m3.coef_[0]:.2f}, {m3.coef_[1]:.2f} (theory: 1.50, -0.50)")
print(f"  test R^2 = {r2_score(y_te, p3):.3f}   median period error = {pct_err:.1f}%")
