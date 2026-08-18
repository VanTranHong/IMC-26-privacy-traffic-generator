# IMC-26 Privacy Traffic Generator

Research code for evaluating the **privacy risk of synthetic network-traffic generators**: does a
model trained on real pcaps leak information about its training data, and how much does mitigating
that risk cost in downstream utility? The repo bundles (1) real pcap datasets, (2) four vendored
synthetic traffic generators, (3) privacy attacks against those generators, (4) mitigation
techniques applied to the traffic itself, and (5) utility evaluation to measure the resulting
fidelity/utility trade-off.

## Repository layout

```
.
├── data/                       # Real pcap datasets (train/test/otheractivity splits), via Git LFS
├── synthetic-data-generation/  # Vendored third-party traffic generators
│   ├── NetDiffusion/           # Stable Diffusion + ControlNet + LoRA (image-encoded flows)
│   ├── NetShare/               # DoppelGANger-style GAN (CMU, SIGCOMM 2022)
│   ├── NetSSM/                 # Fine-tuned Mamba state-space language model
│   └── TrafficLLM/             # P-Tuned / LoRA ChatGLM2
├── privacy-evaluation/         # Attacks against the generators above
│   ├── MIA/                    # Membership inference (black-box + white-box)
│   ├── data-extraction/        # Training-value recovery (precision/recall/F1)
│   └── network-attack/         # Header/topology/property re-identification attacks
├── privacy-risk-mitigate/      # Mitigation applied directly to pcaps (DP noise, anonymization)
├── utility-evaluation/         # Fidelity (JSD/EMD) and TRTR-vs-TSTR downstream classification
└── requirements.txt            # Deps for this repo's own code (not the vendored generators)
```

Every subdirectory under `synthetic-data-generation/`, `privacy-evaluation/`, and
`utility-evaluation/` has its own README with detailed usage instructions — this file is the map;
follow the links below for the details of any one piece.

## Datasets (`data/`)

All datasets follow a `train/ test/ otheractivity` split and are stored as `.pcap` files tracked
with **Git LFS** — see [Setup](#setup) for cloning instructions.

| Dataset | Description | Layout |
|---|---|---|
| `VNAT` | VPN vs. non-VPN traffic (sftp, ssh, netflix, rdp, scp, ...) | flat |
| `SR` | Multi-app service recognition traffic | flat |
| `single_SR` | Service recognition, one app isolated per flow (twitter, facebook, zoom, teams, instagram, amazon, netflix, meet, twitch) | per-app subfolders |
| `IoT` | Smart-home device traffic (alexa, nestcam, refrigerator, ...) | flat |
| `single_IoT` | IoT device fingerprinting, one device isolated per flow (google home, nvidia jetson nano, samsung galaxy j3, GE washer/dryer, nest camera, smartthings dishwasher, samsung fridge, LG nexus5, amazon echo, raspberry pi 3, philips lightbulb, ...) | per-device subfolders |
| `CIC` | CIRA-CIC-DoHBrw-2020 — benign vs. malicious DNS-over-HTTPS (dns2tcp tunneling vs. benign Chrome/Cloudflare DoH) | flat |
| `USTC` | USTC-TFC2016 — benign apps (BitTorrent, Facetime, FTP, Gmail, MySQL, Outlook, SMB, Skype, WorldOfWarcraft) + malware/botnet traffic (Neris, Zeus, Virut, Nsis-ay) | flat |

## Setup

This repo needs **Git LFS** to pull the pcap datasets:

```bash
git lfs install
git clone https://github.com/VanTranHong/IMC-26-privacy-traffic-generator.git
```

Dependencies for this repo's own code (`privacy-evaluation/`, `privacy-risk-mitigate/`,
`utility-evaluation/`) are in `requirements.txt`. The vendored generators each have their own,
separate dependency sets (heavier, CUDA-specific in places) — install those from within each
generator's folder as needed, ideally in a **separate virtual environment** per generator:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # this repo's analysis/mitigation/utility code
# e.g. for NetSSM:
pip install -r synthetic-data-generation/NetSSM/requirements.txt
```

Python ≥3.9 is recommended for this repo's own code; check each generator's README for its own
version/toolchain requirements (e.g. NetSSM needs Go ≥1.21.4 and libpcap in addition to Python
≥3.10).

## Workflow

The pieces are meant to be used together as an end-to-end pipeline:

1. **Generate synthetic traffic** — pick a generator under `synthetic-data-generation/`, train it
   on a `data/<dataset>/train` split, and produce synthetic pcaps/flows. See each generator's
   README for its entry point (e.g. `NetShare/examples/pcap/driver.py`, `NetSSM/train.py` +
   `generation/generate.py`, `TrafficLLM/inference.py`, NetDiffusion's
   `nprint_to_png.py` → `train_dreambooth_lora_sd3_miniature.py` → `image_to_nprint.py` pipeline).
2. **Attack it** — run the relevant scripts under `privacy-evaluation/` against the trained model
   and its outputs to measure membership inference risk (`MIA/`), training-value leakage
   (`data-extraction/`), and network-level re-identification risk (`network-attack/`), using the
   dataset's `train`/`test`/`otheractivity` splits as the membership signal.
3. **Mitigate** — apply `privacy-risk-mitigate/add_DP.py` (Laplace-noise DP on header fields:
   timestamp, TTL, IP ID, ToS, TCP window) and/or `privacy-risk-mitigate/anonymize.py` (IP/MAC
   anonymization, with `complete`/`subset`/`identity-masked` strategies) to the generator's
   training data or outputs, then re-run step 2 to see how much the attack risk drops.
4. **Measure the utility cost** — run `utility-evaluation/utility_evaluation.py` to score
   distributional fidelity (JSD/EMD) and TRTR-vs-TSTR downstream classification accuracy
   (`--classifier xgboost/catboost/random_forest`, `--feature_mode nprint/flow_stats`) before and
   after mitigation, to quantify the privacy/utility trade-off.

Attack and mitigation coverage varies by generator (e.g. NetDiffusion's preprocessing drops IP
headers entirely, so it has no network-identifier attack; TrafficLLM isn't yet wired into the
network-topology or sensitive-property attacks) — check each subfolder's README for exactly which
generator/attack combinations are implemented.

## Notes

- Scripts in `privacy-evaluation/` and `privacy-risk-mitigate/` were adapted from an internal
  research harness not included in this repo, then cleaned up and parameterized via `argparse` —
  no hardcoded personal paths remain.
- The vendored generators (`NetDiffusion`, `NetShare`, `NetSSM`, `TrafficLLM`) are unmodified or
  lightly modified third-party code; see each folder's own README for attribution and the original
  paper citation.
