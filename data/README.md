# Datasets

All four evaluation datasets are **public, real measurement data**. The small inputs required by the
headline experiments are committed to the repository; the bulky raw files (traffic speed series,
Beijing raw archives) are **not committed** (see the repository `.gitignore`) and are fetched from the
canonical public sources below. Every loader lives in `src/data_*.py` and documents the exact columns
it reads.

| Dataset | Domain | Loader | Committed? | Source |
|---|---|---|---|---|
| Delhi air quality (CPCB, 37 stations, daily PM2.5) | environmental | `src/data_real.py` | yes (`data/real_delhi/*.csv`, ~1.4 MB) | E-STGCN repository: `github.com/mad-stat/E_STGCN` (`Dataset/`) |
| Beijing air quality (UCI PRSA, 12 stations) | environmental | `src/data_beijing.py` | no (raw ~40 MB) | UCI ML Repository: *Beijing Multi-Site Air-Quality Data* (`PRSA_Data_*.csv`) |
| Western-US weather (NOAA GHCN-Daily, 30 stations, TMAX) | meteorology | `src/data_weather.py` | yes (`data/real_weather/ca_weather.npz`, ~0.5 MB) | NOAA GHCN-Daily (public, no authentication) |
| PeMSD7-M traffic (Caltrans, 228 sensors) | transportation | `src/data_traffic_real.py` | adjacency only (`pemsd7_adj.npz`) | Caltrans PeMS District 7 / STGCN benchmark `github.com/VeritasYin/STGCN_IJCAI-18` |
| PeMS-BAY traffic (Caltrans, 325 sensors; large-N) | transportation | `src/data_pemsbay.py` | adjacency only (`pemsbay_adj.npz`) | `github.com/hazdzz/STGCN` (`data/pems-bay/vel.csv`) |

## Fetching the non-committed files

- **Beijing (UCI PRSA):** download the *Beijing Multi-Site Air-Quality Data Set* archive from the UCI
  ML Repository and extract the per-station `PRSA_Data_*.csv` files into
  `data/real_beijing/PRSA_Data_20130301-20170228/`. The loader auto-discovers them via glob.
- **PeMSD7-M speed series** (`data/real_pems/pemsd7_vel.csv`, ~69 MB): from the STGCN benchmark repo.
- **PeMS-BAY speed series** (`data/real_pems/pemsbay_vel.csv`, ~81 MB): `vel.csv` from
  `hazdzz/STGCN` at `data/pems-bay/`, saved to that path.

Coordinates for the traffic datasets are reconstructed via classical MDS from the released distance
adjacency (the committed `*_adj.npz`); the speed time series are the original real data.

## Notes

- The Delhi dataset is the primary accuracy benchmark and ships in full so `experiments/06`,
  `experiments/B6`, and the monthly table reproduce with no download.
- After placing the fetched files at the paths above, every experiment in `experiments/` runs
  unchanged; see the top-level `RESULTS-INDEX.md` for which script produces which reported number.
