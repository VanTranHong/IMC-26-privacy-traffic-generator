from typing import Optional, Dict, List
from collections import Counter, OrderedDict
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def get_frequencies(real: List, synthetic: List, categorical_mapping: bool):
    """Get percentual frequencies for each possible real categorical value.
    Given two iterators containing categorical data, this transforms it into
    real/synthetic frequencies which can be used for statistical tests. It
    adds a regularization term (zero) to handle cases where the synthetic data contains values that don't exist in the real data.

    Args:
        real (list):
            A list of hashable objects.
        synthetic (list):
            A list of hashable objects.
        categorical_mapping (bool): 
            search exact mapping of values
    Yields:
        tuple[list, list]:
            The real and synthetic frequencies (as a percent).

    Reference: https://github.com/netsharecmu/SDMetrics_timeseries/blob/6983bc075073a794aabb4a9d48badfbbf0f6ec70/sdmetrics/utils.py#L45-L74
    """
    if categorical_mapping:
        f_real, f_syn = [], []
        real, synthetic = Counter(real), Counter(synthetic)
        for value in synthetic:
            if value not in real:
                # warnings.warn(f'Unexpected value {value} in synthetic data.')
                real[value] = 0  # Regularization to prevent NaN.

        for value in real:
            f_syn.append(synthetic[value] / sum(synthetic.values()))  # noqa: PD011
            f_real.append(real[value] / sum(real.values()))  # noqa: PD011
    else:
        c_real, c_syn = \
            list(OrderedDict(Counter(real).most_common()).values()), \
            list(OrderedDict(
                Counter(synthetic).most_common()).values())
        # pad c_real and c_syn to be the same length with zeros
        real_syn_max_len = max(len(c_real), len(c_syn))
        c_real += [0]*(real_syn_max_len - len(c_real))
        c_syn += [0]*(real_syn_max_len - len(c_syn))
        f_real, f_syn = [i/sum(c_real) for i in c_real], \
            [i/sum(c_syn) for i in c_syn]

    # assert sum(f_real) == 1.0 and sum(
    #     f_syn) == 1.0, f"Relative frequency should sum up to 1.0. f_real: {sum(f_real)}, f_syn: {sum(f_syn)}"
    assert len(f_real) == len(
        f_syn), "length of real data support is not equal to length of synthetic data support"

    return f_real, f_syn


def sort_dict(
    d: Dict,
    by=Literal['key', 'value']
):
    """Sort Python dictionary by key or value"""
    if by == 'key':
        return dict(sorted(d.items()))
    elif by == 'value':
        return dict(sorted(d.items(), key=lambda item: item[1]))
    else:
        raise ValueError(
            "Python dictionary can only be sorted by `key` or `value`!")
