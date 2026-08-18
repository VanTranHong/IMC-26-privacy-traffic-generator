# NetShare: Practical GAN-Based Synthetic IP Header Trace Generation

## 📌 Attribution

This directory documents and mirrors example usage of **NetShare**, developed by researchers at CMU:

- 📄 Paper (SIGCOMM 2022): [Practical GAN-Based Synthetic IP Header Trace Generation Using NetShare](https://dl.acm.org/doi/abs/10.1145/3544216.3544251)
- 💻 Original repository: [github.com/netsharecmu/NetShare](https://github.com/netsharecmu/NetShare)
- 🎥 Talks: [SIGCOMM 2022](https://www.youtube.com/watch?v=mWnFIncjtWg) · [ZeekWeek 2022](https://www.youtube.com/watch?v=MN_fa-FBOHg) · [FloCon 2023](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=890917)

**Authors:** Yucheng Yin, Zinan Lin, Minhao Jin, Giulia Fanti, Vyas Sekar.

All credit for the method, framework, and implementation goes to the original authors. **This folder vendors the code needed to actually run NetShare end-to-end**, copied unmodified (`git archive` of the exact upstream commits, no local edits) from two upstream repositories:

| In this folder | Upstream source | License |
|---|---|---|
| `netshare/`, `examples/`, `traces/`, `util/`, `setup.py` | [github.com/netsharecmu/NetShare](https://github.com/netsharecmu/NetShare) @ `af02603` | Clear BSD ([`LICENSE`](./LICENSE)), Copyright © 2022 Carnegie Mellon University |
| `SDMetrics_timeseries/` | [github.com/netsharecmu/SDMetrics_timeseries](https://github.com/netsharecmu/SDMetrics_timeseries) @ `2f3aadb` | MIT ([`SDMetrics_timeseries/LICENSE`](./SDMetrics_timeseries/LICENSE)), Copyright © 2020 MIT Data To AI Lab |

See the [Citing NetShare](#-citing-netshare) section below for the full citation. NetShare itself builds on and adapts code from [DoppelGANger](https://github.com/fjxmlzn/DoppelGANger), [GPUTaskScheduler](https://github.com/fjxmlzn/GPUTaskScheduler), [BSN](https://github.com/fjxmlzn/BSN), [Ray](https://github.com/ray-project/ray), and [config_io](https://github.com/fjxmlzn/config_io); `SDMetrics_timeseries` is CMU's timeseries fork of [SDMetrics](https://github.com/sdv-dev/SDMetrics).

---

## 📘 Introduction

NetShare uses **Generative Adversarial Networks (GANs)** — specifically a [DoppelGANger](https://github.com/fjxmlzn/DoppelGANger)-based time-series GAN — to learn generative models directly from real packet- or flow-header traces, and then sample new, statistically similar **synthetic** traces from them. It targets networking tasks such as telemetry, anomaly detection, and provisioning, where sharing real traffic data raises privacy concerns.

Given a trace (PCAP or NetFlow/CSV), NetShare:
1. Splits it into metadata fields (e.g., `srcip`, `dstip`, `proto`) and timeseries fields (e.g., packet size, TTL, inter-arrival time).
2. Encodes each field per its declared type (bit / word2vec / categorical / continuous).
3. Trains a DoppelGANger-style GAN on the encoded data.
4. Samples synthetic sequences from the trained GAN and decodes them back into trace rows, optionally with differential privacy (DP-SGD).

---

## 🛠 Setup

### Step 0: `libpcap` (optional, only needed for PCAP input)
```bash
# macOS
brew install libpcap
# Debian/Ubuntu
sudo apt install libpcap-dev
```

### Step 1: Install the vendored packages
Both packages live in this folder already — no need to `git clone` anything.
```bash
conda create --name NetShare python=3.9
conda activate NetShare

cd synthetic-data-generation/NetShare
pip3 install -e .                      # installs the `netshare` package (setup.py at this folder's root)
pip3 install -e ./SDMetrics_timeseries # required for generator.visualize()
```
`requirements.txt` in this folder is a full pinned snapshot (Python 3.9 + CUDA 12.8) of a known-working environment, if you'd rather install everything in one shot with `pip install -r requirements.txt`.

### Step 2: Start Ray (strongly recommended)
NetShare uses [Ray](https://www.ray.io/) to parallelize/distribute training and generation.

Single machine (demo/dev only):
```bash
ray start --head --port=6379 --include-dashboard=True --dashboard-host=0.0.0.0 --dashboard-port=8265
```
Dashboard: [http://localhost:8265](http://localhost:8265)

For a multi-machine cluster, see [`util/README.md`](./util/README.md) (also vendored here).

---

## ⚙️ Configuration file schema

Each dataset is described by a JSON config with:
- `global_config`: path to the raw trace, dataset type (`"pcap"` or `"netflow"`), number of chunks, DP on/off.
- `pre_post_processor.config.timestamp`: how the time column is encoded (e.g., inter-arrival, zero-one normalized).
- `pre_post_processor.config.metadata`: per-row constant fields (e.g., IPs, ports, protocol) — each field is one of:
  - **Bit field** — integer encoded as a bit string (e.g., IPs as 32-bit):
    ```json
    {"column": "srcip", "type": "integer", "encoding": "bit", "n_bits": 32}
    ```
  - **Word2Vec field** — integer encoded via a learned embedding (e.g., ports):
    ```json
    {"column": "srcport", "type": "integer", "encoding": "word2vec_port"}
    ```
  - **Categorical field** — one-hot encoded string:
    ```json
    {"column": "type", "type": "string", "encoding": "categorical"}
    ```
  - **Continuous field** — normalized float:
    ```json
    {"column": "pkt", "type": "float", "normalization": "ZERO_ONE", "log1p_norm": true}
    ```
- `pre_post_processor.config.timeseries`: per-packet/per-flow-record varying fields, using the same field types as above.
- `model.config`: GAN training hyperparameters (`batch_size`, `sample_len`, `epochs`, checkpoint frequency, etc.).

Full worked examples: [`examples/netflow/config_example_netflow_nodp.json`](./examples/netflow/config_example_netflow_nodp.json) and [`examples/pcap/config_example_pcap_nodp.json`](./examples/pcap/config_example_pcap_nodp.json).

> Note: in both example configs, `epochs` is deliberately small for a quick end-to-end smoke test. For production-quality synthetic data, increase `epochs` based on your dataset size and compute budget.

---

## ▶️ How to generate synthetic traffic

### NetFlow example
```bash
cd examples/netflow
python driver.py
```
[`driver.py`](./examples/netflow/driver.py):
```python
import netshare.ray as ray
from netshare import Generator

ray.config.enabled = False
ray.init(address="auto")

generator = Generator(config="config_example_netflow_nodp.json")

# work_folder must NOT already exist, and should be an absolute path
# when using Ray on a multi-machine cluster.
generator.train(work_folder='../../results/test-ugr16')
generator.generate(work_folder='../../results/test-ugr16')
generator.visualize(work_folder='../../results/test-ugr16')

ray.shutdown()
```
This trains on `traces/ugr16-small/raw.csv` (referenced by `config_example_netflow_nodp.json`), then generates and visualizes synthetic NetFlow records under `results/test-ugr16/`.

### PCAP example
```bash
cd examples/pcap
python driver.py
```
[`driver.py`](./examples/pcap/driver.py) follows the same `train` → `generate` → `visualize` pattern, but reads `traces/caida-small/raw.pcap` and writes results to `results/test-caida/`.

### General pattern for your own data
1. Pick the "nearest match" example config (`netflow` for CSV/tabular flow records, `pcap` for raw packet captures) and adapt the `metadata`/`timeseries` field list to your columns.
2. Point `global_config.original_data_file` at your trace, under `traces/<your-dataset>/`.
3. Call, in order: `generator.train(work_folder=...)`, `generator.generate(work_folder=...)`, `generator.visualize(work_folder=...)`.
4. `visualize()` opens a dashboard with a side-by-side real-vs-synthetic comparison (via SDMetrics).

The small `caida-small` and `ugr16-small` demo traces used by the two example configs are already included under [`traces/`](./traces/). The full six public traces used in the original paper are larger and available separately [here](https://drive.google.com/drive/folders/1FOl1VMr0tXhzKEOupxnJE9YQ2GwfX2FD?usp=sharing) — download and place them under `traces/` if you want to reproduce the paper's full results.

---

## 🗂 Codebase structure (this folder)

```
├── README.md                 # This file
├── LICENSE                   # NetShare's Clear BSD license (CMU)
├── setup.py                  # `pip install -e .` installs the `netshare` package below
├── requirements.txt          # Pinned full env snapshot (optional, alternative to setup.py)
├── examples                  # Driver scripts + configs
│   ├── netflow                # NetFlow/CSV example (ugr16-small)
│   └── pcap                   # PCAP example (caida-small)
├── netshare                  # NetShare source code (the installable package)
│   ├── configs                # Default configurations
│   ├── generators              # Generator class (train/generate/visualize entry point)
│   ├── model_managers           # Core train/generate orchestration
│   ├── models                   # Timeseries GAN models (e.g., DoppelGANger)
│   ├── pre_post_processors      # Field encoding/decoding
│   ├── ray                      # Ray function overloading
│   └── utils                    # Utility functions
├── traces                    # Demo datasets (caida-small, ugr16-small)
├── util                      # Ray cluster setup scripts
└── SDMetrics_timeseries      # Vendored dependency, needed for generator.visualize()
```

---

## 📚 Citing NetShare

```bibtex
@inproceedings{netshare-sigcomm2022,
  author = {Yin, Yucheng and Lin, Zinan and Jin, Minhao and Fanti, Giulia and Sekar, Vyas},
  title = {Practical GAN-Based Synthetic IP Header Trace Generation Using NetShare},
  year = {2022},
  isbn = {9781450394208},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3544216.3544251},
  doi = {10.1145/3544216.3544251},
  booktitle = {Proceedings of the ACM SIGCOMM 2022 Conference},
  pages = {458--472},
  numpages = {15},
  keywords = {privacy, synthetic data generation, network packets, network flows, generative adversarial networks},
  location = {Amsterdam, Netherlands},
  series = {SIGCOMM '22}
}
```
