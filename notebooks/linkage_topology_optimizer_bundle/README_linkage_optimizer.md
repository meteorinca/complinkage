# Linkage Topology Optimizer

## Files

- `linkage_topology_optimizer.py`: reusable implementation.
- `linkage_topology_optimization_jupyter.ipynb`: guided JupyterLab workflow with plots, sliders, CMA-ES, shaping, layering, and export.
- `linkage_requirements.txt`: Python dependencies.

## Start

Place the notebook and Python module in the same folder, open the notebook in JupyterLab, and run from the top.

Terminal installation alternative:

```bash
python -m pip install -r linkage_requirements.txt
```

## Main entry points

```python
import linkage_topology_optimizer as lto

result = lto.run_demo(
    n_samples=121,
    n_starts=80,
    n_candidates=8,
    gamma=0.1,
    make_plots=True,
)
```

For real animation data, provide body states as an array shaped:

```text
(n_time_samples, n_rigid_components, 3)
```

with each state stored as:

```text
[theta_radians, world_x, world_y]
```

The notebook shows how to replace the demonstration motion with imported data.

## Numerical notes

- `area_mode="barrier"` is the recommended setting. It prevents zero moment arms without rewarding anchors that move indefinitely far from the character.
- `area_mode="paper"` reproduces the literal negative-log area expression and must be used with finite coordinate bounds.
- The forward solver uses sparse damped Gauss-Newton steps and warm-starts each phase from the preceding solution.
- The singularity term attempts sparse partial SVD first and falls back to dense SVD for small or difficult Jacobians.
- Intersecting planar links require graph coloring for depth layers. Topological sorting applies only after separate directed front/back constraints are defined.

## Physical validation still required

Before fabrication, check clearances, pin diameters, link thickness, bearing friction, backlash, motor torque, stress, deflection, print orientation, and tolerance accumulation.
