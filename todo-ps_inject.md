# ps_inject.py Bug Report

## Bug 1 (crash): `sigma_to_power` error path — line ~279–281

```python
if maxpower.all() == 0:
    return np.zeros(len(n_harm))
```

**Two problems:**

1. `n_harm` is an `int`, so `len(n_harm)` raises `TypeError: object of type 'int' has no len()`.
2. Even if that were fixed, this returns a **single array** while the normal path returns a **tuple** `(scaled_fft, phases)`. The call site at line ~575 unpacks into two variables: `scaled_prof_fft, phases = self.sigma_to_power(n_harm, df)` — this would raise `ValueError`.

**Fix:** `return np.zeros(n_harm), np.zeros(n_harm)`

---

## Bug 2 (wrong branch): `if not TPA_idx:` — line ~171

```python
if not TPA_idx:
    self.phase_prof = np.array(profile)
else:
    self.phase_prof = TPA_profiles[str(TPA_idx)]
```

If `TPA_idx` is `0` (a valid index), `not 0` is `True`, so it takes the wrong branch — it uses `profile` instead of `TPA_profiles["0"]`.

**Fix:** `if TPA_idx is None:`

---

## Bug 3 (fragile/potentially wrong): `np.roots(const)[1]` — line ~127–129

```python
const = np.array([F, E - x * F, D - C - x * E, 1 - B - x * D, -(x + A)])
t = np.roots(const)[1]
prob = np.exp(-(t**2) / 2)
```

`np.roots` returns 4 roots for a degree-4 polynomial, and their **order is not guaranteed** to be stable across numpy versions. Blindly picking index `[1]` may select a complex root, leading to a complex `t`, complex `prob`, and ultimately invalid input to `chdtri`. The code should select the appropriate real, positive root explicitly.

---

## Bug 4 (hardcoded test DB): `smear_fft` database connection — line ~225–226

```python
db = db_utils.connect(host="sps-archiver1", name="test")
```

This hardcodes the host `"sps-archiver1"` and database name `"test"`. This looks like a debug leftover — it will fail in any other environment, and in production it would connect to a test database unintentionally. Compare with `find_closest_pointing` which calls `db_utils.connect()` with no args (using defaults).

---

## Bug 5 (potential division by zero): `get_rednoise_normalisation` — line ~461–462

```python
day_medians = (
    rn_medians[day] / np.min(rn_medians[day], axis=1)[:, np.newaxis]
)
```

If any row in `rn_medians[day]` has a minimum value of `0`, this produces `inf` or `nan` values that propagate into the injection amplitudes.

---

## Bug 6 (logic error): `maxpower.all() == 0` — line ~279

While this accidentally works (because `.all()` on a zero scalar returns `False`, and `False == 0` is `True`), it's semantically wrong and extremely confusing. The intent is clearly `if maxpower == 0:`.

---

## Minor issues

- **Dead code** — `phis` (line ~23), `lorentzian` (line ~49), and `get_median` (line ~140) are defined but never used anywhere in the codebase.
- **Unused variable** — `true_dm_in_harms` at line ~597 is computed but never referenced.
- **Unused variable** — `phases` returned from `sigma_to_power` and `flux_to_power` is never used after assignment.
- **Database call inside a computation loop** — `smear_fft` makes a database query and `find_closest_pointing` call every time it's invoked, which could be cached in `__init__`.
