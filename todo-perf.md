# Performance Optimization TODO

## 1. `barycenter_timeseries`: Repeated `np.insert`/`np.delete` in a loop

**File:** `champss/sps-common/sps_common/barycenter.py` (line ~239–253)

**Impact:** Critical — O(n·k) copies of million-element arrays

Each `np.insert` or `np.delete` on a large array copies the entire array (typically ~1M+ samples). With hundreds to thousands of corrections, this is O(n·k) where both n (array length) and k (corrections) are large.

```python
for i in range(len(correction_index)):
    if correction_required[i] > 0:
        bary_ts = np.insert(bary_ts, correction_index[i] + idx_offset + 1, replace_val)
    else:
        bary_ts = np.delete(bary_ts, correction_index[i] + idx_offset)
```

**Fix:** Pre-compute the final index mapping and do a single array construction. Build a list of output indices, then do `bary_ts_out = bary_ts[output_indices]` in one vectorized operation.

---

## 2. `rednoise_normalise`: Per-element Python loop to build `scale`

**File:** `champss/ps-processes/ps_processes/utilities/utilities.py` (line ~36–46)

**Impact:** High — called per DM trial

Iterates potentially hundreds of thousands of times in Python to build `scale`, calling `np.sum(scale)` on every iteration (O(n²) total). Yet `scale` is typically only ~100–200 elements long because of the exponential growth of window sizes.

```python
for n in range(0, ps_len):
    new_window = np.exp(1 + n / 3) * b0 / np.exp(1)
    ...
    scale.append(window)
    if np.sum(scale) > ps_len:
        ...
```

**Fix:** Precompute the scale array vectorially. The window sizes grow as `b0 * exp(n/3)` — compute the cumulative sum analytically to find where it exceeds `ps_len`, then construct the array directly with `np.clip`.

---

## 3. `rednoise_normalise_runmed`: Per-bin Python loop with `np.median` each iteration

**File:** `champss/ps-processes/ps_processes/utilities/utilities.py` (line ~132–144)

**Impact:** High — 3000 median calls

3000 iterations (default `bmax`), each calling `np.median` on an overlapping window. The windows are largely overlapping so most work is redundant.

```python
for n in range(0, bmax):
    window_size = int(np.exp(1 + n / exp_fac) * w0 / np.exp(1))
    runmed[n] = np.median(power_spectrum[n - int(window_size/2) : n + int(window_size/2)])
    normalised_power_spectrum[n] = power_spectrum[n] / (runmed[n] / np.log(2))
```

**Fix:** Use `scipy.ndimage.median_filter` or a sorted data structure / sliding window approach. For the second loop (`bmax` to end with fixed `wmax`), use `np.median` on full chunks with reshape + `np.median(axis=1)`.

---

## 4. `Injection.get_tsky`: Regenerating the full Haslam sky map per injection

**File:** `champss/ps-processes/ps_processes/processes/ps_inject.py` (line ~189–203)

**Impact:** High — heavy I/O per injection

`HaslamSkyModel` downloads/loads and interpolates a full-sky map for every single `Injection` instance. When running multiple random injections, this is extremely wasteful.

```python
def get_tsky(self):
    haslam = HaslamSkyModel(freq_unit="MHz", spectral_index=-2.6)
    sky_map = haslam.generate(600)  # Generates a full-sky HEALPix map every time
```

**Fix:** Cache the sky model and generated map as a module-level singleton or pass it in.

---

## 5. `Injection.smear_fft`: Database query + pointing lookup per injection

**File:** `champss/ps-processes/ps_processes/processes/ps_inject.py` (line ~225–230)

**Impact:** Medium — network I/O per injection

A database connection + spatial query runs every time `smear_fft` is called. The result depends only on pointing RA/Dec which doesn't change.

```python
def smear_fft(self, scaled_fft):
    db = db_utils.connect(host="sps-archiver1", name="test")
    ap = find_closest_pointing(self.pspec_obj.ra, self.pspec_obj.dec, mode=mode)
```

**Fix:** Move this lookup into `__init__` and cache `nchan`.

---

## 6. `harm_sigma_curve`: Scalar loop calling `sigma_sum_powers` per harmonic

**File:** `champss/sps-common/sps_common/interfaces/single_pointing.py` (line ~315–318)

**Impact:** Medium — called per candidate

The code already has a comment noting this could use vectorized input. `sigma_sum_powers` accepts array inputs. This property is accessed multiple times per candidate (via `best_harmonic_sum` etc).

```python
for i in range(len(harm_sigma_curve)):
    # sigma_sum_powers also accepts array inputs which might be faster
    harm_sigma_curve[i] = sigma_sum_powers(
        raw_power_curve[: i + 1].sum(), self.num_days * (i + 1)
    )
```

**Fix:**

```python
cumulative_sums = np.cumsum(raw_power_curve)
nsums = self.num_days * np.arange(1, len(raw_power_curve) + 1)
harm_sigma_curve = sigma_sum_powers(cumulative_sums, nsums)
```

---

## 7. `scalloping`: Per-harmonic Python loop

**File:** `champss/ps-processes/ps_processes/processes/ps_inject.py` (line ~420–436)

**Impact:** Medium

Loops in Python per harmonic (up to 32), but can be fully vectorized.

```python
for i in range(n_harm):
    f_harm = (i + 1) * self.f
    bin_true = f_harm / df
    bin_below = np.floor(bin_true + 1e-8)
    ...
    bins[i * N : (i + 1) * N] = current_bins
    harmonics[i * N : (i + 1) * N] = np.abs(amplitude) ** 2
```

**Fix:**

```python
f_harms = np.arange(1, n_harm + 1) * self.f
bin_trues = f_harms / df
bin_belows = np.floor(bin_trues + 1e-8)
offsets = np.array([-1, 0, 1, 2])
all_bins = (bin_belows[:, None] + offsets).astype(int)
amplitudes = prof_fft[:, None] * np.sinc(bin_trues[:, None] - all_bins)
bins = all_bins.ravel()
harmonics = np.abs(amplitudes.ravel()) ** 2
```

---

## 8. `predict_sigma`: Python loop with `np.intersect1d` per bin per harmonic

**File:** `champss/ps-processes/ps_processes/processes/ps_inject.py` (line ~502–520)

**Impact:** Medium

`np.intersect1d` is called for each (harmonic × bin) combination. Pre-building a set/hash lookup of `bins` and using vectorized membership tests would be faster.

```python
for search_harmonic in used_search_harmonics:
    ...
    for bin in last_bins:
        bins_in_sum = self.full_harm_bins[:used_injection_harmonic, bin]
        _, _, bin_indices = np.intersect1d(bins_in_sum, bins, return_indices=True)
```

**Fix:** Pre-build a set/hash lookup of `bins` and use vectorized membership tests (e.g. `np.isin`).

---

## 9. Clustering: Dense N×N distance matrix

**File:** `champss/ps-processes/ps_processes/processes/clustering.py` (line ~1030)

**Impact:** Medium–High for large detection counts

The clustering builds a dense `metric_array` of shape `(N_detections, N_detections)`. With thousands of detections, this allocates GBs and is slow. The code has a `use_sparse` path but the harmonic metric loop still iterates row-by-row in Python.

```python
for i in range(metric_array.shape[0]):
    ...
```

**Fix:** Ensure the sparse path is enabled by default and avoid materializing the full dense matrix. The grouped harmonic metric computation already batches pairs — extend this to avoid the per-row loop.

---

## 10. `g_abergel_moisan`: Iterative convergence loop on full array

**File:** `champss/sps-common/sps_common/interfaces/utilities.py` (line ~310–360)

**Impact:** Low–Medium

The continued-fraction evaluation runs a `while np.isinf(g).any()` loop that keeps iterating even when most elements have converged. Each iteration operates on the full array.

**Fix:** Use masked computation or split converged elements out of the computation. After most elements converge (typical within 10 iterations), operating on only the remaining few unconverged values would save significant work for large arrays.

---

## Summary

| # | Location | Issue | Impact |
|---|----------|-------|--------|
| 1 | `barycenter_timeseries` | `np.insert`/`delete` in loop | **Critical** — O(n·k) copies of million-element arrays |
| 2 | `rednoise_normalise` scale loop | O(n²) cumulative sum | **High** — called per DM trial |
| 3 | `rednoise_normalise_runmed` | Per-bin `np.median` | **High** — 3000 median calls |
| 4 | `get_tsky` | Full sky map regeneration | **High** — heavy I/O per injection |
| 5 | `smear_fft` | DB query per call | **Medium** — network I/O per injection |
| 6 | `harm_sigma_curve` | Scalar loop, already noted by developer | **Medium** — called per candidate |
| 7–8 | `scalloping`, `predict_sigma` | Python loops over harmonics/bins | **Medium** |
| 9 | Clustering dense matrix | O(N²) memory + row loop | **Medium–High** for large detection counts |
| 10 | `g_abergel_moisan` | Full-array iteration after convergence | **Low–Medium** |
