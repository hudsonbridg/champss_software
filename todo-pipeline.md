# Pipeline Bug Report

## Bug 1 (crash): `closest_pointing` undefined when `--file` is used without DB

**File:** `champss/sps-pipeline/sps_pipeline/pipeline.py` (line ~1006)

When the database is unreachable (`db_connection = False`), `closest_pointing` is never assigned (only `closest_pointing_id = None` is set at line ~941). But the `to_stack or to_search` branch references it, causing a `NameError`. This is the primary use case for `--file` ("Allows processing without database access"), yet `--file` offers no protection against this path.

```python
if to_stack or to_search:
    ps_detections, power_spectra = ps_cumul_stack.run(
        closest_pointing,  # ← NameError if db_connection is False
        ...
    )
```

---

## Bug 2 (crash): Default `all` components enable incompatible paths with `--file`

**File:** `champss/sps-pipeline/sps_pipeline/pipeline.py` (line ~949)

The `--file` help says "Only works with search-monthly", but if no components are specified (or `all` is), `stack` and `search` are also added. Both require `closest_pointing` and DB access — guaranteed crash when `--file` is used with no DB. There's no guard to skip or disable `stack`/`search` when using `--file`.

```python
if not components or "all" in components:
    components = set(components) | {"search-monthly", "stack", "search"}
```

---

## Bug 3 (crash): Inconsistent return in `load_and_search_monthly`

**File:** `champss/ps-processes/ps_processes/ps_pipeline.py` (line ~387)

A bare `return` returns `None`, but the caller at `pipeline.py` line ~976–978 unpacks into two variables:

```python
(ps_detections_monthly, power_spectra_monthly,) = ...load_and_search_monthly(...)
```

This raises `TypeError: cannot unpack non-iterable NoneType object`. Should be `return None, None`.

```python
else:
    log.error(
        "Monthly stack file for pointing id"
        f" {ps_stack_db.pointing_id} does not exist. Exiting"
    )
    return  # ← returns None, not a tuple
```

---

## Bug 4 (crash): `ps_detections` undefined in `ps_cumul_stack.run` when not searching

**File:** `champss/sps-pipeline/sps_pipeline/ps_cumul_stack.py` (line ~74)

When `to_stack=True, to_search=False`, `run_ps_search` is `False`, so `ps_detections` and `power_spectra` are never assigned, but the function unconditionally returns them.

```python
def run(pointing, ps_cumul_stack_processor, ...):
    if ps_cumul_stack_processor.pipeline.run_ps_stack:
        log.info(...)
    if ps_cumul_stack_processor.pipeline.run_ps_search:
        (ps_detections, power_spectra,) = ...stack_and_search(...)
    return ps_detections, power_spectra  # ← UnboundLocalError if run_ps_search is False
```

---

## Bug 5 (silent data bug): Missing f-string prefix

**File:** `champss/ps-processes/ps_processes/ps_pipeline.py` (line ~393)

Missing `f` prefix. Logs the literal text `{pointing_id}` and `{file}` instead of their values.

```python
log.error(
    "Need either pointing id or file but got {pointing_id} and {file}"
)
```

---

## Bug 6 (wrong path): `log_path` concatenation

**File:** `champss/sps-pipeline/sps_pipeline/pipeline.py` (line ~915)

When `cand_path` is non-empty (e.g., `/data/cands`), this produces `/data/cands./stack_logs/2026/03/11/` — note the stray `.` before `/stack_logs`. Should use `os.path.join` or remove the `.`.

```python
log_path = str(cand_path) + f"./stack_logs/{now.strftime('%Y/%m/%d')}/"
```

---

## Bug 7 (crash): `insufficient_data` undefined in `main`

**File:** `champss/sps-pipeline/sps_pipeline/pipeline.py` (line ~672)

`insufficient_data` is only assigned inside the `if "beamform" in components:` block at line ~548. If beamform is not in the requested components (e.g., running only `ps search`), this raises `NameError`.

```python
if insufficient_data:
```

---

## Bug 8: `num_threads` not recomputed across `active_pointing` iterations

**File:** `champss/sps-pipeline/sps_pipeline/pipeline.py` (line ~524)

After the first loop iteration sets `num_threads`, the `is None` check prevents recomputation on subsequent iterations even though `nchan_factor` may differ across active pointings (different `nchan`). It should either be recomputed each iteration or checked against the CLI-provided value.

```python
if num_threads is None:
    num_threads = int(config.threads.thread_per_1024_chan * nchan_factor)
```

---

## Summary of `--file` path issues

The `--file` codepath is broadly broken because:

1. Only `search-monthly` handles it, but nothing prevents `stack`/`search` from running
2. The no-DB path leaves `closest_pointing` undefined, which both `stack` and `search` require
3. `ra`/`dec` are required click arguments even though `--file` says it overrules them
4. A downstream function has a bare `return` instead of `return None, None`

A proper fix would guard the `to_stack or to_search` block with a `db_connection` check and/or force `to_stack = to_search = False` when `file` is provided.
