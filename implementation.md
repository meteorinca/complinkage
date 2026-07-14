
### 1. System Architecture Overview
*   **Frontend (WebUI):** Three.js / React. Handles 3D scene rendering, input skeleton import (JSON or BVH), and visual preview of candidate linkages.
*   **Backend (Python):** FastAPI (or Flask) for request handling, Celery for asynchronous task queues (critical, since Global Optimization takes 20+ mins), and a database (SQLite/Postgres) to store design session snapshots.
*   **Core Math Layer:** `NumPy` for arrays, `SciPy` for sparse matrix solvers, and the `cma` Python package for Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

---

### 2. Phase 0: Data Models & Input Representation
*   **Define the Skeleton:** Represent the input virtual character as a hierarchical tree of `RigidComponents`. Each component has a state vector \(\mathbf{s}_i = (\alpha, x, y)^T\). Store local coordinates \(\mathbf{q}_j = (u, v)\) for each joint.
*   **Motion Input:** Store the prescribed time-varying motor angles \(m(t)\) for each joint as a time-series array.
*   **Constraint Definitions:** Implement a data structure that can hold:
    1.  **Pin Constraints:** Target joint point coordinates on two components.
    2.  **Motor Constraints:** Time-varying relative angles.
    3.  **Ground/Static Constraints:** Fixed world-space coordinates for the base.

---

### 3. Phase 1: The Forward Kinematics Solver (The Backbone)
Before you design, you must simulate.
*   **Math:** Minimize the constraint penalty energy \(E_c(\mathbf{s}) = \frac{1}{2} \mathbf{C}(\mathbf{s})^T \mathbf{C}(\mathbf{s})\).
*   **Implementation Strategy:** At each time step \(t\), solve for the state \(\mathbf{s}\) using a **Newton-Raphson** approach.
*   **Solver Details:** Compute the Jacobian \(\mathbf{J}\) of the constraint equations. The Hessian of \(E_c\) is \(\mathbf{J}^T\mathbf{J}\). Because the assembly is large, use **SciPy’s sparse linear algebra libraries** (`scipy.sparse.linalg`) to efficiently solve \(\mathbf{J}^T\mathbf{J} \Delta \mathbf{s} = -\mathbf{J}^T \mathbf{C}\).
*   **Performance bottleneck:** This solver must be called dozens of times per second in the interactive phase. Use vectorized NumPy operations to evaluate all time steps simultaneously if the CPU allows.

---

### 4. Phase 2: The Topology Design Engine (Interactive Mode)
This is the discrete combinatorial phase where the user says "Remove motor X, connect Component A to B."
*   **Distance Variance Minimization (Step A):**
    *   Given two components \(c_a\) and \(c_b\), solve for local points \(\mathbf{q}_a\) and \(\mathbf{q}_b\).
    *   **Math:** Minimize \(\delta_{ab} = \frac{1}{n_s}\sum_i^n (||\mathbf{x}_a(t_i) - \mathbf{x}_b(t_i)||^2 - l_{ab})^2\) (Eq. 3). Note that \(l_{ab}\) is the mean squared distance.
    *   **Implementation:** Compute the gradient and Hessian of this 4th-order polynomial. Use **Newton’s method** (with backtracking line search) to find local minima. Since the objective is multi-modal, run multiple random initializations (e.g., 10-20 attempts per requested candidate).
*   **Moment Arm Penalty (Step B):**
    *   **Problem:** Naive optimization yields zero-length links or zero moment arms (no torque transmission). See Figure 3 in the paper.
    *   **Math:** Add a penalty term \(E_{\mathrm{Area}} = -\log \sum_{i = 1}^{n_{s}}\mathrm{area}\left(\mathbf{x}_{b}(t_{i}),\mathbf{x}_{m}(t_{i}),\mathbf{x}_{a}(t_{i})\right)^{2}\) (Eq. 4).
    *   **Weighting:** The paper uses a weighted sum with a 10:1 ratio (Variance vs. Area). Implement this as `Total = δ_ab + w * E_Area`.
*   **Candidate Preview Generation:**
    *   Run Step A & B. Collect the top 6 unique candidate solutions.
    *   Serialize these parameter sets and send them to the Frontend. The user previews the motion simulation of each candidate instantly.
*   **Commit:** Once the user picks one, you **permanently commit the new rigid link** to the topology, remove the motor constraint, and add the new pin constraints.

---

### 5. Phase 3: Auxiliary Links (Aesthetic Addition)
*   Allow the user to add extra decorative links. Crucially, the system must compute the kinematics of these links as slaves to the parent components they attach to. The frontend UI must allow users to pick specific pin locations on the existing linkages to attach these extras.

---

### 6. Phase 4: The Global Optimization Engine (The Heavy Lifter)
Once the topology is fixed, you must fine-tune the *exact* coordinates of all newly added pin joints. This is a black-box optimization problem.
*   **Optimization Algorithm:** **CMA-ES** (implemented via the `cma` Python package). Do not use gradient descent here, as the SVD of the Jacobian (used for singularity avoidance) has discontinuous derivatives, breaking classical optimization.
*   **Cost Function Definition (The sum of 4 energies):**
    1.  **Trajectory Error \(E_{\mathrm{marker}}\) (Eq. 5):** Euclidean distance between current marker points (e.g., foot, hand) and the target input motion over all time frames.
    2.  **State Error \(E_{\mathrm{state}}\) (Eq. 6):** Difference between current component states and the original motorized skeleton states.
    3.  **Joint Distance Error \(E_{\mathrm{joint}}\) (Eq. 7):** Force the pin joints to remain close to their original lines/axes to prevent "floating" linkages.
    4.  **Singularity Avoidance \(E_{\mathrm{singular}}\) (Eq. 8):**
        *   For every time step in the cycle, compute the **Jacobian** of the constraint system.
        *   Compute the **Singular Value Decomposition (SVD)** of this Jacobian.
        *   Extract the smallest singular value \(\lambda_{\mathrm{min}}\).
        *   Add a penalty: \(E_{\mathrm{singular}} = (\lambda_{\mathrm{min}} + \epsilon)^{-\alpha}\) (with \(\alpha=2\), \(\epsilon=10^{-8}\)).
*   **Weights:** Hard-coded for reproducibility: 100 for Markers, 100 for State, 500 for Joints, and 1 for Singularity. **Note:** Because CMA-ES is evaluation-heavy, run a strict timeout; the paper notes a 20-minute to 2-hour runtime for this phase. Backend must process this asynchronously and notify the user upon completion.

---

### 7. Phase 5: Post-Processing & Collision Layering
*   **Spline Shaping (Aesthetics):**
    *   To transform straight bars into organic-looking bones (e.g., curved legs), implement a Catmull-Rom spline fitting.
    *   Automatically compute the tangent directions for the spline endpoints by minimizing changes in tangent direction over the full motion cycle.
*   **Collision Avoidance (Layering):**
    *   The paper simply offsets components normal to the motion plane. This is a combinatorial assignment problem (Figuring out which link goes above which).
    *   **Implementation:** Compute the 2D bounding boxes of all links over the full animation. Build a collision graph. Run a topological sort to assign depth priority to mitigate intersections.
    *   **Boolean Operations:** Use a CSG (Constructive Solid Geometry) library in Python (like `pycsg`) to union the cylinders (pin joints) and bars (links) into a single mesh for 3D printing.

---

### 8. The WebUI User Experience (UX) and Technical Integration
*   **The Editor View:** A Three.js viewer with a slider for the motion cycle (0% to 100%).
*   **The "Replace Motor" Interaction:**
    *   User clicks a motor (red ball).
    *   User clicks a source component \(c_a\) and a target component \(c_b\).
    *   Backend computes 3-6 candidate links *in real-time* (< 0.1s per candidate, expect < 1s total compute).
    *   **Crucial UI:** The WebUI shows a split-screen or thumbnail gallery of these candidates *already animated* so the user can judge aesthetics and motion quality before committing.
*   **The "Global Optimize" Button:** Once topology is finished, the user clicks this. The backend spawns a Celery task. The user sees a progress bar (CMA-ES iterations).

---

### 9. Critical Implementation Pitfalls to Solve
1.  **Trivial Minimum Traps:** The distance variance minimization (Eq. 3) tends to converge to global minima where the link length is zero (see Fig. 3). Solution: During the initialization step of the Newton method, set bounds that force initial guesses to be sufficiently far from existing joint coordinates, and heavily rely on the \(E_{\mathrm{Area}}\) term to break the degeneracy.
2.  **Sparse Solver Performance:** SVD is very slow for large matrices. The Jacobian size is \(n_{\mathrm{constraints}} \times n_{\mathrm{DOF}}\). For complex characters, this can get massive. **Implementation suggestion:** Use `scipy.sparse.linalg.svds` (which computes only the smallest singular value via an Arnoldi iteration) rather than the dense `numpy.linalg.svd` to gain a massive speedup during the global optimization phase.
3.  **Motor Update Order:** In the topology design phase, ensure you do not break the kinematic chain. If you remove a motor, the solver must immediately convert that motor into a ground-truth length constraint on the new link.

### 10. Data Persistence
*   **Session Serialization:** The design process is iterative. Save the full topology as a JSON tree structure where each node is a `RigidComponent` and each edge is a `Constraint` (Pin, Motor, or Ground).
*   **Landmarks:** Store "Checkpoints" before and after the global optimization so the user can return to a previous state if the CMA-ES produces a weird result (e.g., crazy long links that are impossible to fabricate).


