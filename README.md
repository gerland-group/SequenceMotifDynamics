Code
--------------------------

Source code for all simulation, computations and plotting routines
of [Harth-Kitzerow et al: Sequence motif dynamics in RNA pools](https://doi.org/10.1101/2024.12.10.627702).
The data is available here [^1].
For different parameter sets specify one of the following system arguments:

- `zebra_0`
- `zebra_1`
- `zebra_2`
- `zebra_3`
- `zebra_4`
- `zebra_2_2fl` (for different initial conditions in the second parameter set)

like

```bash
python3 1_infer_motif_rates_from_strand_reactor_parameters.py zebra_0
```

[^1] Harth-Kitzerow, Johannes; Tobias Göppel; Burger, Ludwig; Torsten A. Enßlin; Gerland, Ulrich, 2025, "Sequence motif dynamics data", https://doi.org/10.17617/3.L2WUFG, Edmond
