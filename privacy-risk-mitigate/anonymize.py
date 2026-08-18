#!/usr/bin/env python3
"""Identifier anonymization for pcap files.

Three selectable strategies replace or obscure IP and MAC addresses in
pcap packets (recomputing IP/TCP/UDP checksums afterward, since editing
address fields in place leaves the originals stale):

  - `complete`: every packet's source/destination IP and MAC, across every
    file processed in one run, is overwritten with the SAME randomly
    generated identity (one src IP/MAC, one dst IP/MAC for the whole run).
    Maximum anonymization -- also destroys per-flow and per-file identity,
    since every flow in the run becomes indistinguishable from every other.
  - `subset`: coarsens each IP to its /16 (zeroes the last two octets) and
    each MAC to its vendor OUI (zeroes the last three bytes) -- keeps
    coarse-grained structure (rough network/vendor) while dropping the
    host-identifying part. Deterministic, no randomness involved.
  - `identity-masked`: consistently pseudonymizes each distinct IP/MAC to
    the same random replacement everywhere it appears in the run (the
    mapping is kept in memory for the run's duration) -- hides true
    identities while preserving which packets/flows share an endpoint,
    unlike `complete`.

A fourth subcommand, `analyze-frequency`, is not an anonymization step: it
counts how often each IP/MAC address appears across a set of pcaps and
saves the counts to CSV, as a way to see which addresses dominate a
dataset (and are therefore the highest-value anonymization targets)
before or after applying one of the three strategies above.


## Usage

    # Complete anonymization: one shared random identity for the whole run
    python anonymize.py complete \\
        --input_root /path/to/<dataset>/test --output_root /path/to/<dataset>/test_completeanonymized

    # Subset masking: /16 IPs, OUI-only MACs
    python anonymize.py subset \\
        --input_root /path/to/<dataset>/test --output_root /path/to/<dataset>/test_subnet

    # Identity masking: consistent per-address pseudonyms
    python anonymize.py identity-masked \\
        --input_root /path/to/<dataset>/test --output_root /path/to/<dataset>/test_identitymasked

    # Address-frequency analysis (diagnostic, not an anonymization step)
    python anonymize.py analyze-frequency \\
        --input_glob "/path/to/<dataset>/train/**/*.pcap" --dataset <dataset> --output_dir .

Run `python anonymize.py <command> --help` for each subcommand's full flag list.
"""

import argparse
import glob
import os
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from scapy.all import IP, TCP, UDP, Ether, rdpcap, wrpcap


# =============================================================================
# Shared helpers
# =============================================================================

def random_ip() -> str:
    return ".".join(str(random.randint(0, 255)) for _ in range(4))


def random_mac() -> str:
    return ":".join(f"{random.randint(0, 255):02x}" for _ in range(6))


def _recompute_checksums(pkt) -> None:
    """Drops IP/TCP/UDP checksums so scapy recalculates them on write --
    required after editing address fields in place, or the written pcap
    would carry stale checksums for the new addresses."""
    if IP in pkt:
        del pkt[IP].chksum
    if TCP in pkt:
        del pkt[TCP].chksum
    elif UDP in pkt:
        del pkt[UDP].chksum


def iter_pcap_files(input_root: str) -> List[Tuple[str, str]]:
    """Every .pcap file under input_root, recursively, as (absolute_path,
    path_relative_to_input_root) pairs. The relative path is reused
    verbatim under --output_root so subfolder structure (train/test/
    otheractivity, per-app-class folders, ...) is preserved."""
    pattern = os.path.join(input_root, "**", "*.pcap")
    return [(p, os.path.relpath(p, input_root)) for p in sorted(glob.glob(pattern, recursive=True))]


# =============================================================================
# Strategy: complete anonymization
# =============================================================================

class CompleteAnonymizer:
    """Replaces every packet's src/dst IP and MAC with one shared randomly
    generated identity, fixed for the lifetime of this object (i.e. for
    one run of the CLI)."""

    def __init__(self):
        self.source_ip = random_ip()
        self.destination_ip = random_ip()
        self.source_mac = random_mac()
        self.destination_mac = random_mac()

    def anonymize_pcap(self, input_pcap: str, output_pcap: str) -> Optional[str]:
        try:
            packets = rdpcap(input_pcap)
        except Exception as e:
            print(f"[skip] {input_pcap}: {e}")
            return None
        for pkt in packets:
            if IP in pkt:
                pkt[IP].src = self.source_ip
                pkt[IP].dst = self.destination_ip
            if Ether in pkt:
                pkt[Ether].src = self.source_mac
                pkt[Ether].dst = self.destination_mac
            _recompute_checksums(pkt)
        wrpcap(output_pcap, packets)
        return output_pcap


# =============================================================================
# Strategy: subset anonymization
# =============================================================================

def subset_ip(ip_address: Optional[str]) -> Optional[str]:
    """Zeroes the last two octets of an IPv4 address (/16 mask); returns
    the input unchanged if it isn't a 4-part address."""
    if ip_address is None:
        return ip_address
    parts = ip_address.split(".")
    if len(parts) != 4:
        return ip_address
    return ".".join(parts[:2] + ["0", "0"])


def subset_mac(mac_address: Optional[str]) -> Optional[str]:
    """Zeroes the last three bytes of a MAC address, keeping only the
    vendor OUI; returns the input unchanged if it isn't a 6-part address."""
    if mac_address is None:
        return mac_address
    parts = mac_address.split(":")
    if len(parts) != 6:
        return mac_address
    return ":".join(parts[:3] + ["00", "00", "00"])


def subset_anonymize_pcap(input_pcap: str, output_pcap: str) -> Optional[str]:
    try:
        packets = rdpcap(input_pcap)
    except Exception as e:
        print(f"[skip] {input_pcap}: {e}")
        return None
    for pkt in packets:
        if IP in pkt:
            pkt[IP].src = subset_ip(pkt[IP].src)
            pkt[IP].dst = subset_ip(pkt[IP].dst)
        if Ether in pkt:
            pkt[Ether].src = subset_mac(pkt[Ether].src)
            pkt[Ether].dst = subset_mac(pkt[Ether].dst)
        _recompute_checksums(pkt)
    wrpcap(output_pcap, packets)
    return output_pcap


# =============================================================================
# Strategy: identity masking
# =============================================================================

class IdentityMasker:
    """Consistently pseudonymizes each distinct IP/MAC address to the same
    random replacement everywhere it appears, for the lifetime of this
    object (i.e. for one run of the CLI) -- preserves which packets/flows
    share an endpoint without revealing the true address."""

    def __init__(self):
        self.ip_mapping: Dict[str, str] = {}
        self.mac_mapping: Dict[str, str] = {}

    def _masked_ip(self, ip: str) -> str:
        if ip not in self.ip_mapping:
            self.ip_mapping[ip] = random_ip()
        return self.ip_mapping[ip]

    def _masked_mac(self, mac: str) -> str:
        if mac not in self.mac_mapping:
            self.mac_mapping[mac] = random_mac()
        return self.mac_mapping[mac]

    def anonymize_pcap(self, input_pcap: str, output_pcap: str) -> Optional[str]:
        try:
            packets = rdpcap(input_pcap)
        except Exception as e:
            print(f"[skip] {input_pcap}: {e}")
            return None
        for pkt in packets:
            if IP in pkt:
                pkt[IP].src = self._masked_ip(pkt[IP].src)
                pkt[IP].dst = self._masked_ip(pkt[IP].dst)
            if Ether in pkt:
                pkt[Ether].src = self._masked_mac(pkt[Ether].src)
                pkt[Ether].dst = self._masked_mac(pkt[Ether].dst)
            _recompute_checksums(pkt)
        wrpcap(output_pcap, packets)
        return output_pcap


# =============================================================================
# Diagnostic: address-frequency analysis (not an anonymization step)
# =============================================================================

def count_address_frequency(
    pcap_glob: str, dataset: str, output_dir: str = ".", top_n: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Counts how often each IP/MAC address appears (as src or dst) across
    every pcap matched by pcap_glob, saves the full counts to
    `<output_dir>/<dataset>_ip_frequency.csv` and `..._mac_frequency.csv`,
    and prints the top_n most frequent of each."""
    ip_counts: Dict[str, int] = defaultdict(int)
    mac_counts: Dict[str, int] = defaultdict(int)

    for fn in glob.glob(pcap_glob, recursive=True):
        try:
            packets = rdpcap(fn)
        except Exception as e:
            print(f"[skip] {fn}: {e}")
            continue
        for pkt in packets:
            if IP in pkt:
                ip_counts[pkt[IP].src] += 1
                ip_counts[pkt[IP].dst] += 1
            if Ether in pkt:
                mac_counts[pkt[Ether].src] += 1
                mac_counts[pkt[Ether].dst] += 1

    ip_df = pd.DataFrame(ip_counts.items(), columns=["IP Address", "Frequency"]).sort_values("Frequency", ascending=False)
    mac_df = pd.DataFrame(mac_counts.items(), columns=["MAC Address", "Frequency"]).sort_values("Frequency", ascending=False)

    print(f"Top {top_n} IP addresses by frequency:\n{ip_df.head(top_n)}")
    print(f"\nTop {top_n} MAC addresses by frequency:\n{mac_df.head(top_n)}")

    os.makedirs(output_dir, exist_ok=True)
    ip_csv = os.path.join(output_dir, f"{dataset}_ip_frequency.csv")
    mac_csv = os.path.join(output_dir, f"{dataset}_mac_frequency.csv")
    ip_df.to_csv(ip_csv, index=False)
    mac_df.to_csv(mac_csv, index=False)
    print(f"\nSaved: {ip_csv}\nSaved: {mac_csv}")

    return ip_df, mac_df


# =============================================================================
# CLI entry point
# =============================================================================

def _process_all(args: argparse.Namespace, anonymize_fn: Callable[[str, str], Optional[str]]) -> None:
    """Shared driver for the three pcap-rewriting strategies: walks
    --input_root, skips files already present under --output_root (unless
    --overwrite), and calls anonymize_fn(in_path, out_path) on the rest."""
    pairs = iter_pcap_files(args.input_root)
    print(f"Found {len(pairs)} pcap files under {args.input_root}")
    for abs_path, rel_path in pairs:
        out_path = os.path.join(args.output_root, rel_path)
        if os.path.exists(out_path) and not args.overwrite:
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if anonymize_fn(abs_path, out_path):
            print(f"[anonymized] {rel_path}")


def run_complete(args: argparse.Namespace) -> None:
    anonymizer = CompleteAnonymizer()
    print(
        f"Shared identity for this run -- "
        f"src: {anonymizer.source_ip} ({anonymizer.source_mac})  "
        f"dst: {anonymizer.destination_ip} ({anonymizer.destination_mac})"
    )
    _process_all(args, anonymizer.anonymize_pcap)


def run_subset(args: argparse.Namespace) -> None:
    _process_all(args, subset_anonymize_pcap)


def run_identity_masked(args: argparse.Namespace) -> None:
    masker = IdentityMasker()
    _process_all(args, masker.anonymize_pcap)
    print(f"Distinct IPs pseudonymized: {len(masker.ip_mapping)}  Distinct MACs pseudonymized: {len(masker.mac_mapping)}")


def run_analyze_frequency(args: argparse.Namespace) -> None:
    count_address_frequency(args.input_glob, args.dataset, args.output_dir, args.top_n)


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input_root", required=True, help="Root folder to recursively search for .pcap files.")
    parser.add_argument(
        "--output_root", required=True,
        help="Output root; each input file's path relative to --input_root is preserved underneath it.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Reprocess files whose output already exists (default: skip, so a run can be resumed).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pcap identifier anonymization: complete/subset/identity-masked strategies, "
        "plus an address-frequency analysis utility.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Anonymization strategy or analysis to run.")

    p_complete = subparsers.add_parser(
        "complete",
        help="Replace every IP/MAC with one shared randomly generated identity for the whole run "
        "(maximum anonymization; destroys per-flow/per-file identity).",
    )
    _add_io_args(p_complete)
    p_complete.set_defaults(func=run_complete)

    p_subset = subparsers.add_parser(
        "subset",
        help="Zero the host part of each IP (/16) and device part of each MAC (keep vendor OUI) -- "
        "coarsens identifiers without randomizing them.",
    )
    _add_io_args(p_subset)
    p_subset.set_defaults(func=run_subset)

    p_identity = subparsers.add_parser(
        "identity-masked",
        help="Consistently pseudonymize each distinct IP/MAC to the same random replacement everywhere "
        "in the run -- preserves flow/topology structure while hiding true identities.",
    )
    _add_io_args(p_identity)
    p_identity.set_defaults(func=run_identity_masked)

    p_freq = subparsers.add_parser(
        "analyze-frequency",
        help="Count how often each IP/MAC address appears across a set of pcaps; save the counts to CSV "
        "(diagnostic -- not an anonymization step).",
    )
    p_freq.add_argument(
        "--input_glob", required=True,
        help="Glob for .pcap files to scan, e.g. '/path/to/train/**/*.pcap' (use '**' for nested subfolders).",
    )
    p_freq.add_argument("--dataset", required=True, help="Dataset name; used as the output CSV filename prefix.")
    p_freq.add_argument(
        "--output_dir", default=".",
        help="Directory to write <dataset>_ip_frequency.csv / <dataset>_mac_frequency.csv into. Default: '.'.",
    )
    p_freq.add_argument("--top_n", type=int, default=10, help="How many top addresses to print to stdout. Default: 10.")
    p_freq.set_defaults(func=run_analyze_frequency)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
