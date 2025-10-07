#!/usr/bin/env python
"""
Script to create and save an incoherent beam from CHIME SPS data.

The incoherent beam is formed by incoherently summing multiple beams across
a specified time range. This can be used for RFI mitigation and noise removal.
"""
import click
import numpy as np
import glob
import os
import logging
from astropy.time import Time
from spshuff import l1_io
from multiprocessing import Pool, set_start_method, shared_memory
from functools import partial

log = logging.getLogger(__name__)


def extract_data_trange(file_list, tstart, nsamp):
    """
    Extract data across multiple files for a given time range.

    Parameters
    ----------
    file_list : list[str]
        List of files belonging to one beam (in time order).
    tstart : float
        Requested start time (Unix seconds).
    nsamp : int
        Number of samples to extract.

    Returns
    -------
    data : np.ndarray
        Array with shape (1024, nsamp). Returns None if the interval cannot be satisfied.
    """
    tsamp = 0.00098304
    tend = tstart + nsamp * tsamp

    # Gather file coverage ranges
    file_ranges = []
    for fname in file_list:
        with open(fname, "rb") as f:
            int_file = l1_io.IntensityFile.from_file(
                f, shape=(1024, None)
            )
            fh = int_file.fh
            file_ranges.append((fname, fh.start, fh.end))

    # Find files overlapping the requested interval
    overlap_files = [fname for fname, fstart, fend in file_ranges if not (fend <= tstart or fstart >= tend)]
    if not overlap_files:
        log.warning("No files contain requested range")
        return None

    data_blocks = []
    for fname, fstart, fend in file_ranges:
        if fname not in overlap_files:
            continue
        with open(fname, "rb") as f:
            int_file = l1_io.IntensityFile.from_file(
                f, shape=(1024, None)
            )

            chunks = int_file.get_chunks()
            file_data = []
            for chunk in chunks:
                file_data.append(chunk.get_data(apply_mask=True))
            file_data = np.concatenate(file_data, axis=1)

            # build time axis for this file
            taxis = fstart + np.arange(file_data.shape[1]) * tsamp

            # restrict to relevant section
            mask = (taxis >= tstart) & (taxis < tend)
            if np.any(mask):
                data_blocks.append(file_data[:, mask])

    if not data_blocks:
        log.warning("Files overlap but no samples extracted")
        return None

    # Concatenate in time
    data = np.concatenate(data_blocks, axis=1)

    # Ensure we have exactly nsamp samples
    if data.shape[1] < nsamp:
        log.warning(f"Requested {nsamp} samples, but only {data.shape[1]} available")
        return data
    else:
        return data[:, :nsamp]


def group_files_by_beam(file_list):
    """
    Group CHIME files by beam ID (extracted from path).

    Parameters
    ----------
    file_list : list[str]
        List of filenames.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping beam ID -> sorted list of files.
    """
    beam_groups = {}
    for fname in file_list:
        # beam is the directory just before the file
        beam = os.path.basename(os.path.dirname(fname))
        beam_groups.setdefault(beam, []).append(fname)

    # sort files in time order by extracting leading integer from filename
    for beam in beam_groups:
        beam_groups[beam].sort(key=lambda fn: int(os.path.basename(fn).split("_")[0]))
    return beam_groups


def process_beam_to_shared(
    data_sum_shared_name,
    count_shared_name,
    shape,
    beam,
    files,
    tstart,
    nsamp,
    nchan,
):
    """
    Process a single beam and accumulate to shared memory arrays.

    Parameters
    ----------
    data_sum_shared_name : str
        Name of shared memory for data sum
    count_shared_name : str
        Name of shared memory for count
    shape : tuple
        Shape of the arrays (nchan, nsamp)
    beam : str
        Beam ID
    files : list[str]
        List of files for this beam
    tstart : float
        Start time (Unix seconds)
    nsamp : int
        Number of samples to extract
    nchan : int
        Number of channels
    """
    log.info(f"Extracting beam {beam} from {len(files)} files")
    arr = extract_data_trange(files, tstart, nsamp)

    if arr is None:
        log.warning(f"Beam {beam} has no data for requested range")
        return

    # Convert zeros (masked values) to NaNs
    arr[arr < 1] = np.nan

    # Access shared memory
    shared_data_sum = shared_memory.SharedMemory(name=data_sum_shared_name)
    data_sum = np.ndarray(shape, dtype=np.float32, buffer=shared_data_sum.buf)

    shared_count = shared_memory.SharedMemory(name=count_shared_name)
    count = np.ndarray(shape, dtype=np.float32, buffer=shared_count.buf)

    # Handle arrays that may be shorter than nsamp
    ntime = arr.shape[1]
    if ntime < nsamp:
        log.warning(f"Beam {beam} has only {ntime}/{nsamp} samples, padding with NaNs")
        # Only write the available data, leave the rest as zeros (will not affect sum/count)
        arr_slice = arr[:, :ntime]
        mask = ~np.isnan(arr_slice)
        data_sum[:, :ntime][mask] += arr_slice[mask]
        count[:, :ntime][mask] += 1
    else:
        # No padding needed - write directly to shared memory
        mask = ~np.isnan(arr)
        data_sum[mask] += arr[mask]
        count[mask] += 1

    shared_data_sum.close()
    shared_count.close()


def extract_data_allbeams(file_list, tstart, nsamp, beam_range=None, num_processes=1):
    """
    Extract data for a time range across all beams using parallel processing.

    Parameters
    ----------
    file_list : list[str]
        List of filenames across multiple beams.
    tstart : float
        Requested start time (Unix seconds).
    nsamp : int
        Number of samples to extract.
    beam_range : tuple of (int, int), optional
        Beam ID range to include (min_beam, max_beam). If None, use all beams.
    num_processes : int
        Number of parallel processes to use.

    Returns
    -------
    data_sum : np.ndarray
        Shape [1024, ntime]. Sum of all beam data (NaNs excluded).
    count : np.ndarray
        Shape [1024, ntime]. Number of non-NaN samples at each bin.
    beam_ids : list[str]
        Beam IDs that were processed.
    """
    # Fixes some unexpected behaviour with shared memory
    set_start_method("forkserver", force=True)

    tsamp = 0.00098304
    tend = tstart + nsamp * tsamp
    nchan = 1024

    # group files by beam
    grouped = group_files_by_beam(file_list)
    beam_ids = sorted(grouped.keys())

    # Filter by beam range if specified
    if beam_range is not None:
        min_beam, max_beam = beam_range
        beam_ids = [b for b in beam_ids if min_beam <= int(b) <= max_beam]
        log.info(f"Filtering to beams {min_beam}-{max_beam}: {len(beam_ids)} beams")

    # Determine number of processes
    n_processes = min(len(beam_ids), num_processes)
    log.info(f"Using {n_processes} processes for parallel processing")
    log.info(f"Processing {len(beam_ids)} beams total")

    # Create shared memory arrays
    shape = (nchan, nsamp)
    buffer_size = nchan * nsamp * 4  # 4 bytes for float32

    data_sum_shared = shared_memory.SharedMemory(create=True, size=buffer_size)
    data_sum = np.ndarray(shape, dtype=np.float32, buffer=data_sum_shared.buf)
    data_sum[:] = 0

    count_shared = shared_memory.SharedMemory(create=True, size=buffer_size)
    count = np.ndarray(shape, dtype=np.float32, buffer=count_shared.buf)
    count[:] = 0

    # Process beams in parallel
    pool = Pool(n_processes)
    pool.starmap(
        partial(
            process_beam_to_shared,
            data_sum_shared.name,
            count_shared.name,
            shape,
        ),
        [(beam, grouped[beam], tstart, nsamp, nchan) for beam in beam_ids],
    )
    pool.close()
    pool.join()

    log.info(f"Finished processing all {len(beam_ids)} beams")

    # Check if any data was collected
    if np.all(count == 0):
        log.error("No beams contained data in requested range")
        data_sum_shared.close()
        data_sum_shared.unlink()
        count_shared.close()
        count_shared.unlink()
        return None, None, []

    # Copy to regular arrays before cleaning up shared memory
    data_sum_copy = np.array(data_sum)
    count_copy = np.array(count)

    # Clean up shared memory
    data_sum_shared.close()
    data_sum_shared.unlink()
    count_shared.close()
    count_shared.unlink()

    return data_sum_copy, count_copy, beam_ids


def get_datfiles_for_date_and_time(date_str, unix_start, unix_end, datpath='/data/chime/sps/raw'):
    """
    Get list of .dat files for a given date and time range.

    Parameters
    ----------
    date_str : str
        Date in format YYYY/MM/DD
    unix_start : float
        Start time in Unix seconds
    unix_end : float
        End time in Unix seconds
    datpath : str
        Root directory for raw data

    Returns
    -------
    list[str]
        List of .dat files in the time range
    """
    datfiles = np.sort(glob.glob(f'{datpath}/{date_str}/*/*dat'))

    datfiles_range = []
    for datfile in datfiles:
        fname = datfile.split('/')[-1]
        t_start = int(fname.split('_')[0])
        t_end = int(fname.split('_')[1].split('.')[0])
        # Include files with some overlap (with buffer of ~45-37 seconds)
        if (t_start > unix_start - 45) and (t_end < unix_end + 37):
            datfiles_range.append(datfile)

    return datfiles_range


@click.command()
@click.option('--date', type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]), required=True, help='Date in format YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD')
@click.option('--unix-start', type=float, required=True, help='Start time in Unix seconds')
@click.option('--unix-end', type=float, required=True, help='End time in Unix seconds')
@click.option('--nsamp', type=int, default=None, help='Number of samples to extract (overrides unix-end if specified)')
@click.option('--beam-min', type=int, default=None, help='Minimum beam ID to include (optional)')
@click.option('--beam-max', type=int, default=None, help='Maximum beam ID to include (optional)')
@click.option('--output', '-o', required=True, help='Output file path for incoherent beam (.npz)')
@click.option('--datpath', default='/data/chime/sps/raw', help='Root directory for raw data')
@click.option('--num-processes', type=int, default=32, help='Number of parallel processes (default: 32)')
def create_incoherent_beam(date, unix_start, unix_end, nsamp, beam_min, beam_max,
                          output, datpath, num_processes):
    """
    Create and save an incoherent beam from CHIME SPS data.

    The incoherent beam is formed by incoherently averaging multiple beams across
    a specified time range. This can be used for RFI mitigation and noise removal.

    Example usage:
        python create_incoherent_beam.py --date 20250806 --unix-start 1754438500
               --unix-end 1754438600 --beam-min 0 --beam-max 100 -o incoh_beam.npz
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    log.info("=" * 60)
    log.info("Creating Incoherent Beam")
    log.info("=" * 60)

    # Calculate number of samples if not specified
    tsamp = 0.00098304
    if nsamp is None:
        nsamp = int((unix_end - unix_start) / tsamp)
        log.info(f"Calculated nsamp = {nsamp} from time range")
    else:
        log.info(f"Using specified nsamp = {nsamp}")

    # Get beam range
    beam_range = None
    if beam_min is not None and beam_max is not None:
        beam_range = (beam_min, beam_max)
        log.info(f"Beam range: {beam_min} to {beam_max}")
    elif beam_min is not None or beam_max is not None:
        raise click.BadParameter("Must specify both --beam-min and --beam-max or neither")
    else:
        log.info("Using all available beams")

    # Convert datetime object to date string in YYYY/MM/DD format
    date_str = date.strftime("%Y/%m/%d")

    log.info(f"Date: {date_str}")
    log.info(f"Time range: {unix_start} to {unix_end} (Unix seconds)")
    log.info(f"Number of samples: {nsamp}")
    log.info(f"Output file: {output}")

    # Get list of data files
    log.info("Finding data files...")
    datfiles_range = get_datfiles_for_date_and_time(date_str, unix_start, unix_end, datpath)
    log.info(f"Found {len(datfiles_range)} data files")

    if len(datfiles_range) == 0:
        log.error("No data files found for specified date and time range")
        return

    # Extract data from all beams
    log.info("Extracting data from beams...")
    data_sum, count, beam_ids = extract_data_allbeams(
        datfiles_range,
        tstart=unix_start,
        nsamp=nsamp,
        beam_range=beam_range,
        num_processes=num_processes
    )

    if data_sum is None:
        log.error("Failed to extract data")
        return

    log.info(f"Data sum shape: {data_sum.shape} (nchan, ntime)")
    log.info(f"Beam IDs: {beam_ids}")

    # Create incoherent beam by dividing sum by count
    log.info("Creating incoherent beam (normalizing by count)...")
    data_incoh = np.divide(data_sum, count, where=count > 0)
    data_incoh[count == 0] = np.nan
    log.info(f"Incoherent beam shape: {data_incoh.shape} (nchan, ntime)")

    # Save to file
    log.info(f"Saving incoherent beam to {output}...")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log.info(f"Created directory: {output_dir}")

    # Save with metadata
    np.savez(
        output,
        data=data_incoh,
        unix_start=unix_start,
        unix_end=unix_end,
        nsamp=nsamp,
        beam_ids=beam_ids,
        beam_range=beam_range,
        date=date_str,
        nchan=data_incoh.shape[0],
        ntime=data_incoh.shape[1],
    )

    log.info("=" * 60)
    log.info("SUCCESS: Incoherent beam created and saved")
    log.info("=" * 60)
    log.info(f"File: {output}")
    log.info(f"Shape: {data_incoh.shape}")
    log.info(f"Number of beams used: {len(beam_ids)}")

    # Print some statistics
    log.info("Statistics:")
    log.info(f"  Mean: {np.nanmean(data_incoh):.2f}")
    log.info(f"  Std: {np.nanstd(data_incoh):.2f}")
    log.info(f"  Min: {np.nanmin(data_incoh):.2f}")
    log.info(f"  Max: {np.nanmax(data_incoh):.2f}")
    log.info(f"  NaN fraction: {np.isnan(data_incoh).mean():.4f}")


if __name__ == '__main__':
    create_incoherent_beam()
