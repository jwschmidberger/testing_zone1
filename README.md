# testing_zone1

This repository contains miscellaneous experiments. To try the advection-diffusion example you will need `numpy`, `matplotlib` and `plotly` installed. A `requirements.txt` file is provided for convenience.

```bash
pip install -r requirements.txt
```

Run the basic example by executing `gas_transport_modelling/advection_diffusion.py` from the repository root:

```bash
python gas_transport_modelling/advection_diffusion.py
```

A window should appear displaying the final concentration field.

An interactive version using Plotly is available in `advection_diffusion2.py`:

```bash
python - <<'PY'
from gas_transport_modelling.advection_diffusion2 import interactive_wind_example
interactive_wind_example()
PY
```
