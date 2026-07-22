# Easy Way Input Guide

## How to feed YOUR mechanism's motion into the Linkage Topology Optimizer

---

## What this tool actually does

The optimizer takes periodic motion data for **two rigid bodies** (body A and body B) and finds where to attach a **single rigid connecting link** between them — two pin locations `q_a` on body A and `q_b` on body B such that the distance between them stays nearly constant throughout the motion cycle.

The output is a **3D-printable STL file** of that rigid coupler link.

**Key constraint:** If your mechanism has more than 2 bodies, you must break it down into pairs and run the optimizer once per pair. See Section 6 below.

---

## The data format you need

For each body, you need an array of shape `(n_frames, 3)`:

```
[theta_radians, world_x, world_y]
```

- `theta` — the body's rotation angle in radians
- `world_x, world_y` — the body's **local origin** position in world coordinates (typically its ground pivot)

Example for a crank rotating about the origin at (0,0):

```python
# 121 frames over one full rotation
times = np.linspace(0, 2*np.pi, 121, endpoint=False)
crank_states = np.column_stack([
    times,                          # theta: 0 → 2π
    np.full(121, 0.0),              # x: stays at 0
    np.full(121, 0.0),              # y: stays at 0
])
```

You also need:
- `ground_a = (ax, ay)` — where body A is pinned to the frame
- `ground_b = (bx, by)` — where body B is pinned to the frame
- `motor_positions` — (n_frames, 2) — where the driving force is applied each frame (usually = ground_a repeated)

---

## Method 1: CSV from CAD (Onshape, SolidWorks, Fusion 360) ★ RECOMMENDED

This is the easiest path for most users.

### Export from Onshape

1. Open your assembly in Onshape
2. Go to the **Assembly** tab and run your kinematic animation
3. For each body you need to track:
   - Use **Measure** or create a **Trace** point on the body
   - Export the point's `X, Y` and the body's `Rotation` at each frame
4. Save as CSV with columns: `theta, x, y`
5. Do this separately for body A and body B → two CSV files

### Load in the notebook

```python
motion = rwl.load_motion_from_csv(
    path_a="body_a_motion.csv",
    path_b="body_b_motion.csv",
    ground_a=(0.0, 0.0),        # body A pivot in world coords
    ground_b=(80.0, 0.0),       # body B pivot in world coords
    motor_positions_path=None,   # or path to motor CSV
    delimiter=",",
    skip_header=1,               # if CSV has a header row
    columns=(0, 1, 2),           # which columns are theta, x, y
    label="my_mechanism",
)
```

### CSV tips

- If your CSV has columns in a different order, use `columns=(col_theta, col_x, col_y)` to remap
- If your CSV uses tab separation, set `delimiter="\t"`
- Make sure both CSVs have exactly the same number of rows
- The motion MUST be periodic — the first and last poses should be close to each other

---

## Method 2: Direct numpy arrays

If you already have your motion data in Python (from a simulation, handwritten kinematics, etc.):

```python
n = 121
times = np.linspace(0, 2*np.pi, n, endpoint=False)

# Body A motion (e.g. a crank)
my_states_a = np.column_stack([
    times,                     # theta
    np.full(n, 0.0),           # x
    np.full(n, 0.0),           # y
])

# Body B motion (e.g. a rocker — you compute this)
# This must be REAL data from your mechanism
my_states_b = np.column_stack([
    rocker_angles,             # theta
    np.full(n, 80.0),          # x (ground pivot x)
    np.full(n, 0.0),           # y (ground pivot y)
])

motion = rwl.create_motion_from_arrays(
    states_a=my_states_a,
    states_b=my_states_b,
    ground_a=(0.0, 0.0),
    ground_b=(80.0, 0.0),
    motor_positions=None,      # auto-uses ground_a
    times=times,
    label="my_mechanism",
)
```

---

## Method 3: DXF frames from CAD

If you've exported each frame of your animation as a separate DXF file:

```python
# Requires: pip install ezdxf

frame_files_a = [f"frames/body_a_frame_{i:03d}.dxf" for i in range(121)]
frame_files_b = [f"frames/body_b_frame_{i:03d}.dxf" for i in range(121)]

motion = rwl.load_motion_from_dxf_frames(
    frame_files_a=frame_files_a,
    frame_files_b=frame_files_b,
    ground_a=(0.0, 0.0),
    ground_b=(80.0, 0.0),
    layer_name="BodyGeometry",         # DXF layer with the tracking line
    entity_handles=None,               # or ("1A2", "3F4") for specific LINE handles
    label="dxf_imported_motion",
)
```

### How it works

For each DXF frame, the function finds a LINE entity on the specified layer:
- The LINE's **midpoint** becomes `(x, y)`
- The LINE's **direction** becomes `theta`

### DXF tips

- Put the line representing each body on a clearly named layer (e.g. `Body_A`)
- The tracked LINE should be a segment ON the body, away from the pivot
- For best results, make the LINE span from the pivot to the body's tip
- If you know the DXF handles, pass them as `entity_handles=("handle_a", "handle_b")` for reliability
- Export frames with consistent naming: `frame_000.dxf`, `frame_001.dxf`, ...

---

## Method 4: JSON import/export

Save and reload motion data for reproducibility:

```python
# Save
motion_json = {
    "states_a": motion.states_a.tolist(),
    "states_b": motion.states_b.tolist(),
    "ground_a": motion.ground_a.tolist(),
    "ground_b": motion.ground_b.tolist(),
    "motor_positions": motion.motor_positions.tolist(),
    "times": motion.times.tolist(),
}
import json
Path("my_motion.json").write_text(json.dumps(motion_json, indent=2))

# Load
motion = rwl.load_motion_from_json("my_motion.json", label="reloaded")
```

---

## Understanding `motor_positions`

The `motor_positions` parameter defines the point used to compute the **moment arm triangle**. The area of the triangle formed by:

```
motor_position → pin_A → pin_B
```

is the moment arm that transmits torque. The optimizer penalizes small areas to avoid kinematic singularities where the mechanism would bind.

### For different mechanism types:

| Mechanism type | `motor_positions` should be... |
|---|---|
| Crank-rocker (motor on crank) | Crank ground pivot, repeated every frame |
| Crank-rocker (motor on rocker) | Rocker ground pivot, repeated every frame |
| Two rockers with a driving link | The driving link's ground pivot |
| General case | The point where your actuator applies force |

```python
# Typical: motor drives the crank from its ground pivot
motor_positions = np.repeat([[0.0, 0.0]], n_frames, axis=0)

# Or if motor drives body B:
motor_positions = np.repeat([[80.0, 0.0]], n_frames, axis=0)
```

---

## Breaking down a multi-link chain (e.g. a Satyr leg)

The optimizer designs **one link at a time**. For a serial chain like a leg with Hip → Knee → Ankle, you need to run it **twice**:

### Step 1: Hip-to-Knee coupler

```
Body A = Hip segment (rotates about hip pivot)
Body B = Knee segment (rotates about knee pivot)
```

```python
# Motion data for the hip and knee segments
motion_hip_knee = rwl.create_motion_from_arrays(
    states_a=hip_states,      # hip body motion
    states_b=knee_states,     # knee body motion
    ground_a=(hip_x, hip_y),
    ground_b=(knee_x, knee_y),
    label="hip_to_knee",
)

result_1 = rwl.run_linkage_design(motion_hip_knee, ...)
```

### Step 2: Knee-to-Ankle coupler

```
Body A = Knee segment (rotates about knee pivot)
Body B = Ankle segment (rotates about ankle pivot)
```

```python
motion_knee_ankle = rwl.create_motion_from_arrays(
    states_a=knee_states,      # same knee data as above!
    states_b=ankle_states,     # ankle body motion
    ground_a=(knee_x, knee_y),
    ground_b=(ankle_x, ankle_y),
    label="knee_to_ankle",
)

result_2 = rwl.run_linkage_design(motion_knee_ankle, ...)
```

### Step 3: Assemble with collision layering

```python
designs = {
    "hip_knee_coupler": (motion_hip_knee, result_1),
    "knee_ankle_coupler": (motion_knee_ankle, result_2),
}

# Auto-detect which links intersect and assign Z-layers
paths = rwl.export_all_stls(
    designs,
    output_dir="./satyr_leg_output",
    layer_spacing_mm=3.0,      # 3mm between layers
    thickness_mm=4.0,
    width_mm=6.0,
    hole_radius_mm=2.5,
)
```

This produces STL files with different Z-offsets so the links don't collide in 3D space.

---

## Real-world units and fabrication parameters

The optimizer works in whatever units your motion data uses. The `unit_scale` parameter maps abstract units to millimeters for STL export:

| Your data units | `unit_scale` | Example |
|---|---|---|
| Millimeters | 1.0 | `fabrication_thickness=3.0` → 3mm thick |
| Centimeters | 10.0 | `fabrication_thickness=3.0` → 3mm thick |
| Meters | 1000.0 | `fabrication_thickness=3.0` → 3mm thick |
| Inches | 25.4 | `fabrication_thickness=3.0` → 3mm thick |

```python
params = rwl.DesignParameters(
    unit_scale=1.0,              # my data is in mm
    fabrication_thickness=4.0,   # 4mm thick when printed
    fabrication_width=6.0,       # 6mm wide
    hole_radius=2.5,             # 2.5mm radius pin holes (for M5 pins)
)
```

---

## Caching: don't re-run the same optimization twice

The wrapper automatically caches results. If you re-run the notebook without changing the motion data or optimization parameters, it loads from cache instantly:

```
==========================================================
  ✓ CACHE HIT — loading previous results (instant)
==========================================================
```

### When cache is used

Cache is checked based on a hash of:
- Your motion data (states_a, states_b, motor_positions, grounds)
- Optimization parameters (n_starts, cma_max_iterations, gamma, etc.)
- NOT fabrication parameters (thickness, width, hole_radius)

This means you can change STL export dimensions and re-run instantly without re-optimizing.

### Force re-optimization

If you changed your motion data or want to re-run:

```python
params = rwl.DesignParameters(
    force_rerun=True,    # skip cache
    n_starts=200,        # maybe use more starts this time
)
```

Or clear the cache manually: delete the `linkage_cache/` folder.

---

## Common pitfalls & troubleshooting

### "No valid candidates found"
- **Cause:** The optimizer couldn't find any pair of points that maintain nearly constant distance
- **Fix:** Increase `n_starts` (try 200), relax `minimum_moment_arm_fraction` (try 0.001), or check that your motion is periodic

### High distance CV (> 0.05)
- **Cause:** The link length varies significantly across the cycle — the motion may not be achievable with a single rigid link
- **Fix:** Check your input data. A pure 4-bar mechanism should give CV < 0.001

### Near-zero moment arm
- **Cause:** The mechanism passes through a kinematic singularity (all three points collinear)
- **Fix:** Check your mechanism design. The 5th-percentile moment arm should be > 1-2% of the characteristic length

### Simulation does not converge at some frames
- **Cause:** The constraint system is incompatible — the rigid link can't close at that phase
- **Fix:** Review your motion data. The bodies may be too far apart or at impossible angles

### STL has self-intersections
- **Cause:** The link's `bend_fraction` is too large relative to its length
- **Fix:** Reduce `bend_fraction` or increase `fabrication_width`

---

## Quick reference: all input functions

| Function | Use when... |
|---|---|
| `create_motion_from_arrays()` | You have numpy arrays of body states |
| `load_motion_from_csv()` | You exported CSV from CAD/simulation |
| `load_motion_from_dxf_frames()` | You exported per-frame DXF files |
| `load_motion_from_json()` | You saved motion data as JSON previously |
| `create_crank_rocker_motion()` | You want synthetic test data |

---

## Quick reference: all output functions

| Function | Use when... |
|---|---|
| `run_linkage_design()` | Full pipeline with caching |
| `quick_design()` | One-call convenience wrapper |
| `export_all_stls()` | Multi-link assembly with collision Z-layers |
| `build_multi_link_trajectories()` | Get trajectories for custom collision analysis |
| `assign_multi_link_layers()` | Get Z-layer assignments only |
