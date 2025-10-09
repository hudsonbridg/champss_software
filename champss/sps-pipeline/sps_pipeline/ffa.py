"""FFA (Fast Folding Algorithm) search module for the SPS pipeline."""

import logging
import datetime
import os
import numpy as np
from omegaconf import OmegaConf

from FFA import (
    periodogram_form,
    periodogram_form_dm_0,
    gaussian_model,
    SinglePointingCandidate_FFA,
    SearchAlgorithm_FFA,
    Clusterer_FFA,
    Cluster_FFA,
)
import riptide
from scipy.optimize import curve_fit
from sps_common.constants import TSAMP

log = logging.getLogger(__name__)


class FFAProcessor:
    """
    FFA search processor for the SPS pipeline.

    This class handles FFA search on dedispersed time series, including:
    - Periodogram creation across DM range
    - Peak detection and filtering
    - Candidate clustering
    - Writing detections and candidates to disk

    Parameters
    ----------
    run_ffa_search : bool
        Whether to run FFA search
    create_periodogram_plots : bool
        Whether to create periodogram plots
    create_profile_plots : bool
        Whether to create pulse profile plots
    write_ffa_detections : bool
        Whether to write raw detections to file
    write_ffa_candidates : bool
        Whether to write clustered candidates to file
    write_ffa_stack : bool
        Whether to write stacked periodograms
    dm_downsample : int
        DM downsampling factor (default: 8)
    dm_min : float
        Minimum DM to search (default: 0)
    dm_max : float
        Maximum DM to search (default: None, uses max from dedisp_ts)
    min_rfi_sigma : float
        Minimum sigma for RFI detection (default: 5)
    min_detection_sigma : float
        Minimum sigma for valid detections (default: 7)
    birdie_tolerance : float
        Tolerance for birdie removal in seconds (default: 0.005)
    num_threads : int
        Number of threads for parallel processing (default: 8)
    """

    def __init__(
        self,
        run_ffa_search=True,
        create_periodogram_plots=False,
        create_profile_plots=False,
        write_ffa_detections=True,
        write_ffa_candidates=True,
        write_ffa_stack=True,
        dm_downsample=8,
        dm_min=0,
        dm_max=None,
        min_rfi_sigma=5,
        min_detection_sigma=7,
        birdie_tolerance=0.005,
        num_threads=8,
    ):
        self.run_ffa_search = run_ffa_search
        self.create_periodogram_plots = create_periodogram_plots
        self.create_profile_plots = create_profile_plots
        self.write_ffa_detections = write_ffa_detections
        self.write_ffa_candidates = write_ffa_candidates
        self.write_ffa_stack = write_ffa_stack
        self.dm_downsample = dm_downsample
        self.dm_min = dm_min
        self.dm_max = dm_max
        self.min_rfi_sigma = min_rfi_sigma
        self.min_detection_sigma = min_detection_sigma
        self.birdie_tolerance = birdie_tolerance
        self.num_threads = num_threads

    def run_ffa_search_on_dedisp(
        self,
        dedisp_ts,
        ra,
        dec,
        date,
        basepath="./",
        birdies=None,
    ):
        """
        Run FFA search on dedispersed time series.

        Parameters
        ----------
        dedisp_ts : DedispersedTimeSeries
            Dedispersed time series object
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        date : datetime.datetime
            Observation date
        basepath : str
            Base path for output files
        birdies : list, optional
            List of known birdie periods to filter

        Returns
        -------
        ffa_detections : list
            List of raw FFA detections
        ffa_candidates : list
            List of clustered FFA candidates
        """
        if not self.run_ffa_search:
            log.info("FFA search is disabled")
            return None, None

        log.info(f"Starting FFA search on RA={ra:.2f}, DEC={dec:.2f}, Date={date}")

        # Get DM range
        dm_max = self.dm_max if self.dm_max is not None else dedisp_ts.dms.max()

        # Downsample DM trials
        dm_indices = np.arange(0, len(dedisp_ts.dms), self.dm_downsample)
        dm_trials = dedisp_ts.dms[dm_indices]

        # Filter DM range
        dm_mask = (dm_trials >= self.dm_min) & (dm_trials <= dm_max)
        dm_trials = dm_trials[dm_mask]
        dm_indices = dm_indices[dm_mask]

        log.info(f"Searching {len(dm_trials)} DM trials from {dm_trials[0]:.1f} to {dm_trials[-1]:.1f}")

        # Get RFI peaks from DM=0
        log.debug("Finding RFI peaks at DM=0")
        if birdies is None:
            try:
                rfi_peaks = periodogram_form_dm_0(
                    dedisp_ts.dedisp_ts[0],
                    period_min=2.0,
                    period_max=20.0,
                    min_rfi_sigma=self.min_rfi_sigma,
                )
                birdies = [peak.period for peak in rfi_peaks]
                log.info(f"Found {len(birdies)} RFI birdies at DM=0")
            except Exception as e:
                log.warning(f"Could not determine RFI peaks: {e}")
                birdies = []

        # Run FFA on each DM trial
        all_detections = []

        for idx, dm in zip(dm_indices, dm_trials):
            try:
                ts, pgram, peaks = periodogram_form(
                    dedisp_ts.dedisp_ts[idx],
                    dm=dm,
                    birdies=birdies,
                    min_detection_sigma=self.min_detection_sigma,
                    min_rfi_sigma=self.min_rfi_sigma,
                    birdie_tolerance=self.birdie_tolerance,
                    period_min=2.0,
                    period_max=20.0,
                )

                # Store detections
                for peak in peaks:
                    all_detections.append({
                        'dm': dm,
                        'freq': 1.0 / peak.period,
                        'period': peak.period,
                        'sigma': peak.snr,
                        'width': peak.width,
                    })

                log.debug(f"DM={dm:.1f}: {len(peaks)} detections")

            except Exception as e:
                log.error(f"Error processing DM={dm:.1f}: {e}")
                continue

        log.info(f"Total raw detections: {len(all_detections)}")

        if len(all_detections) == 0:
            log.warning("No FFA detections found")
            return None, None

        # Cluster detections into candidates
        ffa_candidates = self._cluster_detections(all_detections, ra, dec)

        # Write outputs
        if self.write_ffa_detections or self.write_ffa_candidates:
            self._write_outputs(
                all_detections,
                ffa_candidates,
                ra,
                dec,
                date,
                basepath,
            )

        return all_detections, ffa_candidates

    def _cluster_detections(self, detections, ra, dec):
        """
        Cluster raw detections into candidates.

        Parameters
        ----------
        detections : list
            List of detection dictionaries
        ra : float
            Right ascension
        dec : float
            Declination

        Returns
        -------
        candidates : list
            List of SinglePointingCandidate_FFA objects
        """
        if len(detections) == 0:
            return []

        log.info("Clustering FFA detections into candidates")

        # Convert to structured array
        detection_array = np.array(
            [(d['dm'], d['freq'], d['sigma'], d['width']) for d in detections],
            dtype=[('dm', float), ('freq', float), ('sigma', float), ('width', int)]
        )

        # Initialize clusterer
        clusterer = Clusterer_FFA()

        # Cluster detections
        try:
            clusters = clusterer.cluster(detection_array)
            log.info(f"Created {len(clusters)} candidate clusters")

            # Create candidate objects
            candidates = []
            for cluster in clusters:
                candidate = SinglePointingCandidate_FFA(
                    freq=cluster.freq,
                    dm=cluster.dm,
                    ra=ra,
                    dec=dec,
                    sigma=cluster.sigma,
                    summary=cluster.summary,
                    features=cluster.features,
                    detections=cluster.detections,
                    detection_statistic=SearchAlgorithm_FFA.periodogram_form,
                )
                candidates.append(candidate)

            return candidates

        except Exception as e:
            log.error(f"Error clustering detections: {e}")
            return []

    def _write_outputs(self, detections, candidates, ra, dec, date, basepath):
        """
        Write FFA detections and candidates to disk.

        Parameters
        ----------
        detections : list
            Raw detections
        candidates : list
            Clustered candidates
        ra : float
            Right ascension
        dec : float
            Declination
        date : datetime.datetime
            Observation date
        basepath : str
            Base output path
        """
        # Create output directory
        date_str = date.strftime("%Y/%m/%d")
        output_dir = os.path.join(basepath, "ffa_output", date_str)
        os.makedirs(output_dir, exist_ok=True)

        pointing_str = f"{ra:.2f}_{dec:.2f}"

        # Write detections
        if self.write_ffa_detections and detections:
            det_file = os.path.join(output_dir, f"{pointing_str}_FFA_detections.npz")
            detection_array = np.array(
                [(d['dm'], d['freq'], d['sigma'], d['width']) for d in detections],
                dtype=[('dm', float), ('freq', float), ('sigma', float), ('width', int)]
            )
            np.savez(det_file, detections=detection_array)
            log.info(f"Wrote {len(detections)} detections to {det_file}")

        # Write candidates
        if self.write_ffa_candidates and candidates:
            cand_file = os.path.join(output_dir, f"{pointing_str}_FFA_candidates.npz")

            # Serialize candidates
            cand_data = {
                'num_candidates': len(candidates),
                'ras': np.array([c.ra for c in candidates]),
                'decs': np.array([c.dec for c in candidates]),
                'freqs': np.array([c.freq for c in candidates]),
                'dms': np.array([c.dm for c in candidates]),
                'sigmas': np.array([c.sigma for c in candidates]),
            }

            np.savez(cand_file, **cand_data)
            log.info(f"Wrote {len(candidates)} candidates to {cand_file}")


def initialise(config, num_threads=8):
    """
    Initialize FFA processor from configuration.

    Parameters
    ----------
    config : OmegaConf
        Configuration object with 'ffa' section
    num_threads : int
        Number of threads for processing

    Returns
    -------
    FFAProcessor
        Initialized FFA processor
    """
    ffa_config = config.get("ffa", {})

    if isinstance(ffa_config, OmegaConf):
        ffa_config = OmegaConf.to_container(ffa_config)

    # Add num_threads to config
    ffa_config['num_threads'] = num_threads

    log.info("Initializing FFA processor")
    processor = FFAProcessor(**ffa_config)

    return processor


def run(active_pointing, dedisp_ts, ffa_processor, basepath, date, birdies=None):
    """
    Run FFA search on an observation.

    Parameters
    ----------
    active_pointing : Pointing
        Active pointing object with ra, dec
    dedisp_ts : DedispersedTimeSeries
        Dedispersed time series
    ffa_processor : FFAProcessor
        Initialized FFA processor
    basepath : str
        Base path for outputs
    date : datetime.datetime
        Observation date
    birdies : list, optional
        Known RFI birdies

    Returns
    -------
    ffa_detections : list
        Raw detections
    ffa_candidates : list
        Clustered candidates
    """
    log.info(
        f"FFA Search ({active_pointing.ra:.2f} {active_pointing.dec:.2f}) @ {date:%Y-%m-%d}"
    )

    ffa_detections, ffa_candidates = ffa_processor.run_ffa_search_on_dedisp(
        dedisp_ts=dedisp_ts,
        ra=active_pointing.ra,
        dec=active_pointing.dec,
        date=date,
        basepath=basepath,
        birdies=birdies,
    )

    if ffa_candidates:
        log.info(f"FFA search complete: {len(ffa_candidates)} candidates found")
    else:
        log.info("FFA search complete: no candidates found")

    return ffa_detections, ffa_candidates
