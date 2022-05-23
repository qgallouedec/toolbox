from typing import List, Optional, Tuple

import numpy as np
import rliable.library as rly
from rliable import metrics
from scipy.interpolate import interp1d


def load_eval(file: str, key: str = "results") -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads evaluations.npz and returns timesteps and results

    :param file: File path.
    :type file: str
    :return: Timesteps and results as an array of shape (num_evals x num_timesteps)
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    timesteps = np.load(file)["timesteps"]
    values = np.load(file)[key]
    return timesteps, values


def load_evals(files: List[str], key: str = "results") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the evaluations generated by SB3 learning.

    :param files: A list of path to the numpy results files
    :return: Timesteps and results as an array of shape (num_runs x num_evals x num_timesteps)
    """
    all_timesteps_values = [load_eval(file, key) for file in files]
    timesteps = all_timesteps_values[0][0]
    values = [timesteps_values[1] for timesteps_values in all_timesteps_values]
    values = np.stack(values)
    return timesteps, values


def iqm(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the interquartile mean across runs and the 95% confidence interval.

    :param values: The values as a matrix of the shape (num_runs x num_timesteps)
    :type values: np.ndarray
    :return: The IQM, the lower bound confidence and the upper bound confidence
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    scores_dict = {"algo": values}
    func = lambda scores: np.array([metrics.aggregate_iqm(s) for s in scores.T])
    iqm_scores, iqm_cis = rly.get_interval_estimates(scores_dict, func, reps=5000)
    return iqm_scores["algo"], iqm_cis["algo"][0], iqm_cis["algo"][1]


def performance_profile(values: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the performance profile under the thresholds.

    :param values: The values as a matrix of the shape (num_runs x num_evals)
    :type values: np.ndarray
    :param thresholds: The thresholds on which the values is evaluated
    :type thresholds: np.ndarray
    :return: The performance profile, the upper confidence bound and the lower confidence bound
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    scores_dict = {"algo": values}
    score_distributions, score_distributions_cis = rly.create_performance_profile(scores_dict, thresholds)
    return score_distributions["algo"], score_distributions_cis["algo"][0], score_distributions_cis["algo"][1]


def rescale(values: np.ndarray, timesteps: np.ndarray, target_length: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rescale the values with a target lenght by interpolation.

    :param values: Inital values
    :type values: np.ndarray
    :param timesteps: Intial timesteps
    :type timesteps: np.ndarray
    :param target_length: The target number of timesteps
    :type target_length: np.ndarray
    :return: The new values and the new timesteps with the new lenght
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    old_idx = np.arange(values.shape[1])
    f = interp1d(old_idx, old_idx, kind="nearest")
    new_idx = np.arange(target_length) * values.shape[1] / target_length
    new_idx = f(new_idx).astype(np.int64)
    return values[:, new_idx], timesteps[new_idx]


def save_iqm(
    values: np.ndarray,
    timesteps: np.ndarray = None,
    step: int = 1,
    filename: str = "result.dat",
    target_length: Optional[int] = None,
) -> None:
    """
    Save the interquartile mean across runs and the 95% confidence interval.

    :param values: The values as a matrix of the shape (num_runs x num_timesteps)
    :type values: np.ndarray
    :param step: Number of timesteps between values, defaults to 1.
    :type step: int, optional
    :param filename: Filename for saving, defaults to "result.dat".
    :type filename: str, optional
    :param target_length: The target number of elements in the output file. Downsample if necessary.
        If None, stores every inputs elements.
    :type target_length: int or None, optional
    :param quantiles: The quantiles to compute; values must be between 0 and 1
    :type quantiles: List or None, optional

    ```
    timestep med lowq highq
    25000 0.080 0.070 0.090
    50000 0.060 0.030 0.090
    75000 0.140 0.100 0.160
    ```
    """
    timesteps = timesteps if timesteps is not None else np.arange(values.shape[1]) * step
    if target_length is not None:
        values, timesteps = rescale(values, timesteps, target_length)
    med, lowq, highq = iqm(values)
    out = np.vstack((timesteps, med, lowq, highq)).transpose()
    header = " ".join(("timestep", "iqm", "lowq", "highq"))
    fmt = " ".join(("%d", "%.3f", "%.3f", "%.3f"))
    np.savetxt(filename, out, fmt=fmt, header=header, comments="")


def save_performance_profile(values: np.ndarray, min_val: float, max_val: float, filename: str = "result.dat") -> None:
    """
    Save the performance profile and the 95% confidence interval.

    :param values: The values as a matrix of the shape (num_runs x num_evals)
    :type values: np.ndarray
    :param min_val: Minimum value
    :type min_val: float
    :param max_val: Maximum value
    :type max_val: float
    :param filename: Filename for saving, defaults to "result.dat".
    :type filename: str, optional

    ```
    timestep med lowq highq
    -2.957 0.995 0.985 1.000
    -2.866 0.995 0.985 1.000
    -2.775 0.995 0.985 1.000
    ```
    """
    thresholds = np.linspace(min_val, max_val, 50)
    med, lowq, highq = performance_profile(thresholds, values)
    out = np.vstack((thresholds, med, lowq, highq)).transpose()
    header = " ".join(("thresholds", "med", "lowq", "highq"))
    fmt = " ".join(("%.3f", "%.3f", "%.3f", "%.3f"))
    np.savetxt(filename, out, fmt=fmt, header=header, comments="")


def save_median(
    values: np.ndarray,
    timesteps: np.ndarray = None,
    step: int = 1,
    filename: str = "result.dat",
    target_length: Optional[int] = None,
    quantiles: Optional[List] = None,
) -> None:
    """
    Save the median score and optionnally quantiles.

    :param values: The values as a matrix of the shape (num_runs x num_timesteps)
    :type values: np.ndarray
    :param timesteps: Timesteps as a 1D array of size num_timesteps, defaults to None
    :type timesteps: np.ndarray, optional
    :param step: When timesteps is None, use this number as number timesteps between values, defaults to 1
    :type step: int, optional
    :param filename: Filename for saving, defaults to "result.dat".
    :type filename: str, optional
    :param target_length: The target number of elements in the output file. Downsample if necessary.
        If None, stores every inputs elements.
    :type target_length: Optional[int], optional
    :param quantiles: The quantiles to compute; values must be between 0 and 1
    :type quantiles: List or None, optional

    When quantiles is set to [0.05, 0.95], the output file looks like

    ```
    timestep med q0.05 q0.95
    25000 0.080 0.070 0.090
    50000 0.060 0.030 0.090
    75000 0.140 0.100 0.160
    ```
    """
    timesteps = timesteps if timesteps is not None else np.arange(values.shape[1]) * step
    if target_length is not None:
        values, timesteps = rescale(values, timesteps, target_length)
    med = np.median(values, axis=0)
    quantiles = [] if quantiles is None else quantiles  # when quantile is None, turn it into []
    qs = [np.quantile(values, q, axis=0) for q in quantiles]
    out = np.vstack((timesteps, med, *qs)).transpose()
    header = " ".join(("timestep", "med", *["q" + str(q) for q in quantiles]))
    fmt = " ".join(("%d", "%.3f", *["%.3f" for _ in quantiles]))
    np.savetxt(filename, out, fmt=fmt, header=header, comments="")
