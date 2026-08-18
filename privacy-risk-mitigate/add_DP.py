#!/usr/bin/env python3
"""Differential-privacy sanitization for pcap packet headers.

Adds calibrated Laplace-mechanism noise to a handful of packet-header
fields -- packet timestamp, IP TTL, IP identification, IP ToS, and TCP
window size -- governed by a total privacy budget `epsilon`, split evenly
across fields (`PcapDPSanitizer.epsilon_allocation`). This is a local,
per-file/per-packet noise-addition mitigation: there is no dataset-wide
sensitivity analysis and no formal (epsilon, delta)-DP accounting across a
release, so treat `--epsilon` as a per-file noise-budget knob rather than a
certified privacy parameter.

Packet-size perturbation is allocated 1/6 of the budget
(`epsilon_allocation['packet_size']`) but intentionally never spent:
changing a packet's *size* means changing its payload, which risks an
invalid or semantically different packet, so this implementation only
perturbs header fields that can be modified in place safely. The
allocation entry is kept anyway so the remaining fields still split epsilon
6 ways, matching the original script's behavior.

Adapted from `add_DP.py` in the local shared harness at
`/net/scratch/vantran/code/private_sharing_network/privacy_mitigation/`
(cleaned up, parameterized via `argparse`, generalized from one hardcoded
input/output folder pair to a recursive `--input_root`/`--output_root`
glob that preserves arbitrary subfolder structure, and the six near-
identical range calculations in `generate_sensitivity()` deduplicated into
one per-field helper). No functional change to the DP-noise math itself.

## Usage

    python add_DP.py \\
        --input_root /path/to/<dataset>/test \\
        --output_root /path/to/<dataset>/test_epsilon_01 \\
        --epsilon 0.1 --sensitivity_mode data_driven

Run `python add_DP.py --help` for the full flag list.
"""

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scapy.all import IP, TCP, rdpcap, wrpcap

DEFAULT_EPSILON_ALLOCATION = {
    "packet_size": 1 / 6,  # allocated, never spent -- see module docstring
    "timestamp": 1 / 6,
    "tcp_window": 1 / 6,
    "ttl": 1 / 6,
    "ip_id": 1 / 6,
    "ip_tos": 1 / 6,
}

DEFAULT_SENSITIVITIES = {
    "packet_size": 50,      # +/-50 bytes: keeps traffic patterns without breaking MTU
    "timestamp": 0.001,     # +/-1ms: preserves flow timing characteristics
    "tcp_window": 1000,     # small fraction of typical windows (16K-64K)
    "ttl": 5,                # obscures exact hop count without breaking routing
    "ip_id": 1000,           # wide range; used for fragmentation, low analytical value
    "ip_tos": 8,              # ToS/DSCP is meaningful -- keep noise small
}


# =============================================================================
# Core: Laplace-mechanism sanitizer
# =============================================================================

class PcapDPSanitizer:
    """Adds Laplace-mechanism noise to a fixed set of packet-header fields
    in a pcap, spending a total privacy budget `epsilon_total` split across
    fields according to `epsilon_allocation`."""

    def __init__(
        self,
        epsilon_total: float = 1.0,
        sensitivities: Optional[Dict[str, float]] = None,
        epsilon_allocation: Optional[Dict[str, float]] = None,
    ):
        self.epsilon_total = epsilon_total
        self.epsilon_allocation = epsilon_allocation or dict(DEFAULT_EPSILON_ALLOCATION)
        self.sensitivities = sensitivities or dict(DEFAULT_SENSITIVITIES)

    def _laplace_noise(self, sensitivity: float, epsilon: float, size: int = 1) -> np.ndarray:
        scale = sensitivity / (epsilon + 1e-12)
        return np.random.laplace(0.0, scale, size=size)

    def sanitize_pcap(self, in_pcap: str, out_pcap: str) -> Optional[str]:
        """Reads `in_pcap`, perturbs timestamp/TTL/IP-ID/ToS/TCP-window on
        every packet, and writes the result to `out_pcap`. Returns
        `out_pcap`, or None if `in_pcap` couldn't be read."""
        try:
            packets = rdpcap(in_pcap)
        except Exception as e:
            print(f"[skip] {in_pcap}: {e}")
            return None

        eps_used = {field: self.epsilon_total * weight for field, weight in self.epsilon_allocation.items()}

        for pkt in packets:
            if hasattr(pkt, "time"):
                pkt.time += float(self._laplace_noise(self.sensitivities["timestamp"], eps_used["timestamp"]))

            if IP in pkt:
                ip = pkt[IP]
                ip.ttl = int(np.clip(ip.ttl + self._laplace_noise(self.sensitivities["ttl"], eps_used["ttl"]), 1, 255))
                ip.id = int(np.clip(ip.id + self._laplace_noise(self.sensitivities["ip_id"], eps_used["ip_id"]), 0, 65535))
                ip.tos = int(np.clip(ip.tos + self._laplace_noise(self.sensitivities["ip_tos"], eps_used["ip_tos"]), 0, 255))

            if TCP in pkt:
                tcp = pkt[TCP]
                tcp.window = int(np.clip(
                    tcp.window + self._laplace_noise(self.sensitivities["tcp_window"], eps_used["tcp_window"]),
                    0, 65535,
                ))

        wrpcap(out_pcap, packets)
        return out_pcap


def compute_data_driven_sensitivity(
    pcap_path: str, fallback: Optional[Dict[str, float]] = None
) -> Optional[Dict[str, float]]:
    """Derives per-field sensitivities from one pcap's own value ranges
    (max - min) instead of using fixed guesses. Falls back to `fallback`
    (default: DEFAULT_SENSITIVITIES) for any field with too few
    observations to produce a nonzero range. Returns None if the pcap
    can't be read."""
    fallback = fallback or DEFAULT_SENSITIVITIES
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"[skip] {pcap_path}: {e}")
        return None

    packet_sizes, timestamps, tcp_windows, ttls, ip_ids, ip_tos = [], [], [], [], [], []
    for pkt in packets:
        packet_sizes.append(len(pkt))
        if hasattr(pkt, "time"):
            timestamps.append(pkt.time)
        if IP in pkt:
            ttls.append(pkt[IP].ttl)
            ip_ids.append(pkt[IP].id)
            ip_tos.append(pkt[IP].tos)
        if TCP in pkt:
            tcp_windows.append(pkt[TCP].window)

    def _range_or_fallback(values: List[float], field: str) -> float:
        return (max(values) - min(values)) if values and max(values) - min(values) > 0 else fallback[field]

    return {
        "packet_size": _range_or_fallback(packet_sizes, "packet_size"),
        "timestamp": _range_or_fallback(timestamps, "timestamp"),
        "tcp_window": _range_or_fallback(tcp_windows, "tcp_window"),
        "ttl": _range_or_fallback(ttls, "ttl"),
        "ip_id": _range_or_fallback(ip_ids, "ip_id"),
        "ip_tos": _range_or_fallback(ip_tos, "ip_tos"),
    }


# =============================================================================
# CLI entry point
# =============================================================================

def iter_pcap_files(input_root: str) -> List[Tuple[str, str]]:
    """Every .pcap file under input_root, recursively, as (absolute_path,
    path_relative_to_input_root) pairs. The relative path is reused
    verbatim under --output_root so subfolder structure (train/test/
    otheractivity, per-app-class folders, ...) is preserved."""
    pattern = os.path.join(input_root, "**", "*.pcap")
    return [(p, os.path.relpath(p, input_root)) for p in sorted(glob.glob(pattern, recursive=True))]


def run_add_dp(args: argparse.Namespace) -> None:
    pairs = iter_pcap_files(args.input_root)
    print(f"Found {len(pairs)} pcap files under {args.input_root}")

    for abs_path, rel_path in pairs:
        out_path = os.path.join(args.output_root, rel_path)
        if os.path.exists(out_path) and not args.overwrite:
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if args.sensitivity_mode == "data_driven":
            sensitivities = compute_data_driven_sensitivity(abs_path)
            if sensitivities is None:
                continue  # unreadable pcap; already reported above
        else:
            sensitivities = None  # PcapDPSanitizer falls back to DEFAULT_SENSITIVITIES

        sanitizer = PcapDPSanitizer(epsilon_total=args.epsilon, sensitivities=sensitivities)
        print(f"[dp eps={args.epsilon}] {rel_path}")
        sanitizer.sanitize_pcap(abs_path, out_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add Laplace-mechanism differential-privacy noise to pcap packet headers "
        "(timestamp, TTL, IP ID, IP ToS, TCP window).",
    )
    parser.add_argument("--input_root", required=True, help="Root folder to recursively search for .pcap files.")
    parser.add_argument(
        "--output_root", required=True,
        help="Output root; each input file's path relative to --input_root is preserved underneath it.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1,
        help="Total privacy budget per pcap, split evenly across the 6 allocated fields (see module docstring). "
        "Smaller = more noise. Default: 0.1.",
    )
    parser.add_argument(
        "--sensitivity_mode", choices=["data_driven", "fixed"], default="data_driven",
        help="'data_driven' (default): derive each field's sensitivity from that pcap's own value range. "
        "'fixed': use DEFAULT_SENSITIVITIES for every file.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Reprocess files whose output already exists (default: skip, so a run can be resumed).",
    )
    parser.set_defaults(func=run_add_dp)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
