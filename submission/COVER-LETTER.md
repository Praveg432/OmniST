# Cover Letter

[Date]

To the Editor-in-Chief,
[Journal name — e.g. Neurocomputing / IEEE Internet of Things Journal]

Dear Editor,

Please consider our manuscript, "OmniST: Topology-Invariant Spatiotemporal Forecasting with
Guaranteed-Coverage Uncertainty at Unseen Sensor Locations," for publication in [journal name].

Spatiotemporal graph convolutional networks are tied to a fixed adjacency matrix and therefore to a
fixed set of sensors. When a sensor network changes, by a station being added, failing, or going
offline for maintenance, these models cannot produce a prediction at a location they did not train on.
This is a routine situation in air-quality, traffic, and weather monitoring, and it is the gap our work
addresses.

We present OmniST, a forecaster whose computation has no structural dependence on the number of sensors.
Composed with inductive split-conformal calibration, it produces distribution-free, coverage-guaranteed
prediction intervals at sensor locations never seen during training, and it preserves that guarantee as
sensors are added or removed. We support the design with a locality proposition (each node's
nonconformity score is exactly invariant to non-neighbor nodes, so the guarantee is recalibration-free
under far-field topology change) and with experiments on four real-world datasets across three physical
domains (Delhi and Beijing air quality, NOAA GHCN weather, and Caltrans traffic, up to 325 real sensors).

We want to be explicit about scope and attribution:

- The contribution is a capability, namely guaranteed inductive uncertainty under dynamic topology, and
  not a point-accuracy state of the art. On the standard fixed-topology benchmark OmniST matches rather
  than beats a comparable GCN+LSTM, and we say so throughout.
- Our two learned comparison methods, E-STGCN and IGNNK, are faithful reimplementations of the published
  architectures, not the authors' original code. We state this in the manuscript and in every relevant
  table caption. The controlled comparison is run under identical data, seeds, preprocessing, and
  training budget so that the two methods differ only in the spatial backbone.
- Every reported number is reproducible from a committed result file in the public repository, via a
  one-command reproduction manifest.

The work is original, has not been published elsewhere, and is not under review at any other venue. A
preliminary version is available as an arXiv preprint (arXiv:2611.04256); this submission extends and
refines that preprint, and we are happy to provide the overlap statement your policy requires. The
authors declare no conflicts of interest. All datasets are public, and the code is openly available at
https://github.com/Praveg432/OmniST.

We believe the problem is of practical importance to [journal]'s readership and that the combination of
topology invariance with a distribution-free coverage guarantee at unseen locations is a genuinely new
contribution to spatiotemporal forecasting. Thank you for your consideration.

Sincerely,

Praveg Vashishtha
Department of Computer Science and Engineering, Indian Institute of Technology Patna, India
Sopra Banking Software India
praveg_pa2503mth259@iitp.ac.in
