Create the python script with extremely detailed visuals/plots where needed. We'll be trying it on jupyterlab environment so you can include sliders, etc and other useful jupyter features)

Here is a detailed, strictly mathematical and algorithmic specification. This serves as an engineering blueprint you can hand off to a Python developer (or follow yourself) to implement the core logic of the paper: **automatically determining exactly where to place new mechanical linkages to replicate an input motion while guaranteeing physical stability.**

---

# Technical Specification: Python-Based Linkage Topology & Kinematic Optimization

## Primary Objective
To mathematically solve for the specific local coordinates \((u, v)\) of new rigid links that, when inserted into a mechanical assembly, force the assembly to replicate a target periodic motion as closely as possible while avoiding mechanical singularities (bifurcations, lock-ups, or zero-moment arms). 

The goal is to generate a kinematic structure that can be physically fabricated and driven by a single phase-driver (motor/crank). 

---

## 1. Core Mathematical Kernel: The Constrained Kinematics Solver
Before optimizing topology, we must simulate the physics of the assembly at time \(t\). We model the character as \(n_c\) rigid components. Each component \(i\) has a state vector \(\mathbf{s}_i = (\alpha, x, y)^T\). We assemble these into a global state vector \(\mathbf{s}\).

**Forward Simulation:**
Given a set of constraints (Pin joints, Motor angles \(m(t)\), Ground constraints), we solve for the new state \(\mathbf{s}\) by minimizing the constraint penalty energy:
\[
E_{c}(\mathbf{s}) = \frac{1}{2}\mathbf{C}(\mathbf{s})^{T}\mathbf{C}(\mathbf{s})
\]
*   **Algorithm Implementation**: Use **Newton-Raphson** iteration. Compute the Jacobian matrix \(\mathbf{J}\) of the constraint vector \(\mathbf{C}\) with respect to the state \(\mathbf{s}\). The Hessian of \(E_c\) is approximated by \(\mathbf{J}^T\mathbf{J}\).
*   **Solver**: Because \(\mathbf{J}^T\mathbf{J}\) is a large, symmetric, positive-semidefinite sparse matrix, implement the step update using a sparse linear algebra solver (e.g., `scipy.sparse.linalg.spsolve`) to solve for \(\Delta\mathbf{s}\).

---

## 2. Topology Design: Mathematical Rule for "Where to Put the Linkages"
The core task is to remove a single motor \(m\) (which had a time-varying angle) and replace it with a rigid link of length \(L\). This link connects a specific point \(\mathbf{q}_a\) on component \(c_a\) to a point \(\mathbf{q}_b\) on component \(c_b\). The mathematical problem is to find the local coordinates \((u_a, v_a)\) and \((u_b, v_b)\) of these points.

### Step 2.1: Distance Variance Minimization
We first approximate the ideal linkage by finding two points on the moving bodies whose world-space distance remains as constant as possible over the entire motion cycle.
For \(n_s\) discrete time samples covering the periodic cycle, we define the mean squared world-space distance:
\[
l_{ab} = \frac{1}{n_s}\sum_i^{n_s} ||\mathbf{x}_a(t_i) - \mathbf{x}_b(t_i)||^2
\]
The variance of this distance is our first objective to minimize:
\[
\delta_{ab} = \frac{1}{n_s}\sum_i^{n_s} \left(||\mathbf{x}_a(t_i) - \mathbf{x}_b(t_i)||^2 - l_{ab}\right)^2
\]
*   **Derivative Calculation**: \(\delta_{ab}\) is a fourth-order polynomial in the unknown local coordinates \((u_a, v_a, u_b, v_b)\). You must analytically derive the Gradient \(\nabla \delta_{ab}\) and the Hessian \(\mathbf{H}_{\delta_{ab}}\).
*   **Solving**: Employ **Newton's method** with a backtracking line search to converge to a local minimum. 

### Step 2.2: Moment Arm Constraint (Penalty Term)
Naively minimizing \(\delta_{ab}\) will often result in a degenerate solution: the new link has zero length, or the moment arm \(l_m\) (the distance from the removed motor \(m\) to the line of action of the new link) becomes zero. A zero moment arm means no torque can be transmitted, rendering the joint locked.
To enforce high torque transmission, we add a penalty based on the area of the triangle formed by the removed motor's position \(\mathbf{x}_m(t)\), the new link's anchor point on the driving component \(\mathbf{x}_a(t)\), and the anchor point on the driven component \(\mathbf{x}_b(t)\):
\[
E_{\mathrm{Area}} = -\log \sum_{i = 1}^{n_{s}}\mathrm{area}\left(\mathbf{x}_{b}(t_{i}),\mathbf{x}_{m}(t_{i}),\mathbf{x}_{a}(t_{i})\right)^{2}
\]
*   **Weighting**: Combine the objectives as a weighted sum: \(J = \delta_{ab} + \gamma E_{\mathrm{Area}}\) (using a weighting factor \(\gamma = 0.1\) or similar). 
*   **Candidate Generation**: Because the variance minimization problem has multiple local minima (creating different visual solutions, as seen in Figure 2), run Step 2.1 and 2.2 repeatedly with **randomized initial guesses for \(\mathbf{q}_a\) and \(\mathbf{q}_b\)**. Generate 5–10 unique valid solutions, calculate their resulting motion via the Forward Kinematic Solver, and present them to the user to select the preferred "shape" or aesthetic.

---

## 3. Global Optimization: Ensuring Smooth, Singularity-Free Motion
Once the topology is fixed (a specific network of bars and pin joints has been chosen), the local coordinates of all the pin joints inserted during the design phase must be fine-tuned to best match the target motion. 
*   **Optimizer Choice**: We must use **Derivative-Free Optimization** (specifically, Covariance Matrix Adaptation Evolution Strategy, or `CMA-ES`). Standard gradient descent cannot be used because the objective function involves the smallest singular value of a matrix (see below), which is not continuously differentiable. Use `cma.fmin` or `cma.CMAEvolutionStrategy` in Python.

### The Cost Function (Objective \(J_{global}\))
We minimize a weighted sum of four energy terms, evaluated over the full motion cycle:
1.  **Trajectory Deviation (\(E_{marker}\))**: Minimize the Euclidean distance between key end-effector markers (e.g., foot, hand) \(\mathbf{m}_i\) and their target positions \(\tilde{\mathbf{m}}_i\) from the original animation.
2.  **State Deviation (\(E_{state}\))**: Minimize the deviation of the current rigid component states \(\mathbf{s}_i(t)\) from the original input animation states \(\tilde{\mathbf{s}}_i(t)\). 
3.  **Joint Alignment (\(E_{joint}\))**: To prevent unnaturally floating linkages, penalize the Euclidean distance between a pin joint \(\mathbf{j}_i\) and the line segment \(\mathbf{l}_i\) corresponding to the original skeleton's component bone to which it attaches.
4.  **Singularity Penalty (\(E_{singular}\))** *(Crucial for 3D printing success)*:
    *   For every time step \(t_i\) in the animation, compute the **Constraint Jacobian Matrix** of the assembly (the same matrix used in the Forward Kinematics solver).
    *   Compute the **Smallest Singular Value** \(\lambda_{\mathrm{min}}\) of this matrix using a sparse SVD decomposition (using `scipy.sparse.linalg.svds` to get only the smallest singular value, rather than the full dense SVD which is computationally infeasible).
    *   **Penalty**: 
        \[
E_{\mathrm{singular}} = \sum_{j}^{n_s}(\lambda_{\mathrm{min}}(t_i) + \epsilon)^{- \alpha}
\]
    *   *Math rationale*: If \(\lambda_{\mathrm{min}} \to 0\), the Jacobian is rank deficient, meaning the mechanism hits a singularity (bifurcation point) where the assembly locks or flips unpredictably. This penalty term aggressively pushes the pin joint parameters away from these degenerate states.

---

## 4. Post-Processing for Aesthetics and Fabrication
### Linkage Shaping (Spline Fitting)
To transition the system from rigid bars to organic-looking character limbs (bones, tails, etc.), take the center-lines of the bars and fit them with Catmull-Rom splines. The tangent directions of the spline end-caps are automatically computed by minimizing the change in tangent direction across the entire periodic motion cycle to prevent tangling.

### 3D Fabrication Layering
Since planar linkages intersect in 2D, you must perform a combinatorial layering operation. Construct an intersection graph of all mechanical components throughout the full motion cycle. Perform a topological sort to assign each link to a specific depth layer (Z-axis offset). This prevents physical collision during motion.

---

## 5. Python Implementation Workflow Summary
1.  **Data Model**: Implement classes for `RigidComponent` (state vector, local point geometry) and `Constraint` (Pin, Motor, Ground).
2.  **Solver Class**: Implement `ForwardKinematicsSolver(components, constraints, time_steps)` returning the time-series state array \(\mathbf{s}(t)\) using Newton-Raphson.
3.  **Topology Heuristic**: Implement `OptimizeLinkagePoints(c_a, c_b, motor_m, time_series_data)`.
    *   Inner loop: Multi-start Newton descent on \(\delta_{ab} + \gamma E_{Area}\).
    *   Return a dictionary of 5-10 candidate local coordinates for pins \(\mathbf{q}_a\) and \(\mathbf{q}_b\).
4.  **Global Optimizer**: Implement `GlobalOptimization(assembly_topology, input_motion_data)`.
    *   Initialize parameters (the local coordinates of all newly added pin joints).
    *   Invoke `cma.fmin` with the objective function:
        `J(params) = w1*E_marker + w2*E_state + w3*E_joint + w4*E_singular`
    *   Ensure `E_singular` is computed via sparse SVD across all time steps.
5.  **Formatting for Fabrication**: Export the finalized coordinate arrays and linkage thicknesses into a mesh-generation script (e.g., using `pycsg` or `trimesh`) for STL file output.

WHAT SUCCESS LOOKS LIKE:
This mathematical pipeline Should give me exact algorithmic machinery to calculate **where to place the extra linkages**, **how long they should be**, and **how to shape them** to produce a graceful, physically viable mechanical automaton from a simple skeletal animation.
