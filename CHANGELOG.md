# Changelog

Notable changes to BLINDSPOT 2.0 are recorded here.

## Unreleased

### Added
- Full SE(3) pose-graph optimisation for short trajectories, jointly refining camera rotations and translations from odometry and independently verified relative-pose constraints.
- Dashboard full 6DoF correction controls with explicit roll/pitch/yaw convention and corrected PLY/trajectory exports.
- Rigid point-cloud ICP registration with voxel sampling, trimmed correspondences, and fitness/RMSE diagnostics.
- Multi-camera rig pose composition from pairwise stereo extrinsics, with graph connectivity and cycle-residual diagnostics.
- Optional ICP refinement in multi-camera PLY fusion.
- Per-pixel stereo relative confidence maps, dashboard heatmap/CSV export, and confidence-versus-depth-error benchmark diagnostics.
- Ground-truth trajectory evaluation reporting first-pose-aligned ATE and consecutive relative translation/rotation errors.
- Expanded release validation, CLI import checks, Streamlit startup checks, and Docker build checks.

### Improved
- Stereo depth filtering now rejects left-right disparity inconsistencies before reconstructing depth.
- Camera calibration reports per-view reprojection errors and robustly removes gross outlier views when the minimum calibration sample count remains satisfied.
- Deployment and validation documentation now distinguish automated software checks from measured real-camera evidence.

### Limitations
- Loop-closure candidates are not automatically converted into metric relative-pose constraints; users must independently verify supplied measurements.
- ICP is a local optimiser and depends on overlap, initial alignment, geometry, and consistent units.
- Multi-camera rig composition depends on pairwise stereo calibration; it does not synchronise cameras or detect calibration targets itself.
- Stereo confidence is a relative heuristic, not a calibrated probability.
- Synthetic tests and CI do not establish real-world depth, localisation, detection, or safety accuracy. Field claims require independent camera/ground-truth validation.
