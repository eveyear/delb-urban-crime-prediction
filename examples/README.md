# Minimal DELB/CMI Simulation

This example is a small, deterministic entry point to the same population
model, CMI estimators, and exact integer-envelope inversion used by E12. It
does not read city data and does not reproduce the full E12 grid.

Run from the project root:

```bash
python examples/minimal_delb_cmi_simulation.py
```

Optional arguments:

```bash
python examples/minimal_delb_cmi_simulation.py \
  --seed 20260723 \
  --sample-size 365 \
  --output examples/minimal_delb_cmi_results.csv
```

The program compares two known populations:

- `population_null`: the added state has exactly zero population CMI;
- `positive_cmi`: the added state changes the count distribution and has
  positive population CMI.

For each scenario it reports exact population CMI, one finite-sample plug-in
and Miller--Madow estimate, the exact baseline and augmented DELBs, their
population reduction, and the MSE of a rounded oracle predictor. The internal
assertions check zero reduction under the null, positive ordering under the
alternative, and that the predictor MSE does not fall below the augmented
population DELB.

The single finite-sample estimates are illustrative rather than inferential.
Use `empirical/run_e12_cmi_simulation.py` for the registered repeated
simulation and randomization study.
