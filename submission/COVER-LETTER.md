# Cover Letter

November 6, 2026

To the Editor-in-Chief,
IEEE Sensors Journal

Dear Editor,

Please consider our manuscript, "OmniST: Topology-Invariant Forecasting for Dynamic Sensor Networks
With Guaranteed Coverage at Unseen Sensors," for publication as a regular paper in the IEEE Sensors
Journal.

Operational sensor networks rarely keep the layout they were trained on. Air-quality stations are
added each year, traffic detectors fail, and weather stations go offline for maintenance. Graph
forecasters built on a fixed adjacency matrix cannot produce a forecast at a newly installed sensor
and break when a trained sensor is removed. Even where a forecast exists, an operator needs to know how
far to trust it at a location with no measurement history.

The manuscript presents OmniST, a forecaster with no structural dependence on the number of sensors.
Combined with inductive split-conformal calibration, it returns prediction intervals with a
finite-sample coverage guarantee at sensor locations never seen in training, and the guarantee holds
as sensors are added or removed. We prove that each sensor's score is unaffected by topology changes
outside its neighborhood, so no recalibration is needed in that regime. On four real sensor datasets
(Delhi and Beijing air quality, NOAA weather, and Caltrans traffic, up to 325 sensors), coverage at
unseen sensors stays at or above 90% under 30 to 50% sensor dropout, where model-free interpolators
fall short and a fixed-topology baseline cannot produce an output at all.

We would like to be clear about scope:

- The contribution is reliable forecasting and uncertainty for changing sensor networks, not a
  point-accuracy state of the art. On a fixed network, OmniST is on par with a comparable GCN+LSTM
  model, and the manuscript says so.
- The two learned baselines, E-STGCN and IGNNK, are our reimplementations of the published
  architectures, not the authors' code. This is stated in the text and in the table captions.
- Every number is reproducible from a committed result file in the public repository
  (https://github.com/Praveg432/OmniST). All datasets are public.

A longer version of this work is available as an arXiv preprint (arXiv:2611.04256) under a different
title. This submission is a condensed version of that preprint, focused on sensor networks;
supporting results that did not fit the page budget are provided as supplementary material. The
manuscript has not been published elsewhere and is not under review at any other venue. The author
declares no competing interests. AI-assisted tools were used in preparing the code and text, as
disclosed in the Acknowledgment.

Thank you for your consideration.

Sincerely,

Praveg Vashishtha
Department of Computer Science and Engineering, Indian Institute of Technology Patna, India
Sopra Banking Software India
praveg_pa2503mth259@iitp.ac.in
