# BLINDSPOT 2.0 — Release readiness

This document defines the software release gate for the current prototype. A
passing automated build means the repository is internally consistent; it does
not establish real-world perception accuracy, calibrated uncertainty, safe
navigation, or field readiness.

## Automated release gate

The GitHub Actions workflow must pass all of the following on the exact release
commit:

1. Python source compilation.
2. Complete unit-test suite.
3. Synthetic multi-camera fusion benchmark smoke test.
4. Streamlit health-endpoint startup check.
5. Docker image build and container startup/health-endpoint smoke test.

Keep the workflow URL and commit SHA with the release record.

## Manual evidence required before a field-validation claim

- Camera models, lens, resolution, frame rate, exposure settings, and mounting.
- Intrinsic and stereo calibration files, chessboard dimensions, square size,
  baseline, and units.
- Independently measured depth targets and the depth benchmark JSON.
- Manually labelled detection/tracking footage and benchmark metrics.
- Ground-truth camera trajectory or surveyed scene geometry for 3D reconstruction.
- Raw and corrected trajectory comparisons when using loop-closure constraints.
- Representative lighting, distance, texture, motion, occlusion, and failure cases.

Do not publish a field-accuracy or safety claim without this evidence.

## Known technical boundaries

- Image-quality confidence scores are heuristic unless separately calibrated
  against independent held-out labels.
- Camera calibration reports overall and per-view reprojection errors and can remove gross outlier views with a robust threshold. Reprojection error is an image-space fit metric, not a guarantee of metric accuracy.
- Stereo depth depends on calibration, synchronization, texture, and scene
  conditions. The matcher applies bidirectional left-right disparity consistency
  to reject many occlusion and mismatch pixels; this can reduce valid coverage and
  does not replace measured depth validation.
- YOLO/ByteTrack masking only excludes detected tracked moving regions; missed
  and untracked objects can remain in the reconstruction.
- Stereo visual odometry is local and can accumulate drift.
- The SE(3) pose-graph API and dashboard workflow jointly adjust rotations and translations for short trajectories using supplied relative-pose constraints. They do not discover or metrically verify loop closures; incorrect constraints can worsen results.
- Multi-camera point-cloud fusion still requires poses in a common coordinate frame and distance unit. A separate ICP API can estimate a local rigid alignment between overlapping clouds, but it is sensitive to initialisation, overlap, and scene geometry; it is not intrinsic calibration or a guarantee of correct extrinsics.
- PLY import supports ASCII and binary little-/big-endian PLY 1.0 with scalar
  XYZRGB vertex properties; list-valued vertex properties are not supported.
- Synthetic benchmarks measure software behaviour/throughput on the runner, not real camera accuracy or target-machine performance. The trajectory evaluator and depth evaluator report only against the supplied independent ground truth/known distances.
- Stereo per-pixel confidence combines left-right consistency and local texture as a relative heuristic; it is not a calibrated probability.
- Visual loop-closure candidate retrieval uses ORB matching and RANSAC geometry,
  but it does not generate metric pose constraints or confirm a loop automatically.
  The full SE(3) pose-graph optimiser is implemented and can jointly refine
  rotations and translations from supplied relative-pose constraints. Dashboard
  constraints are manually entered and must be independently verified; automatic
  metric loop-closure detection and constraint generation remain unimplemented.

## Release procedure

1. Merge only reviewed changes with all required checks passing.
2. Confirm README, deployment instructions, and validation checklist match the
   implemented feature set.
3. Create a versioned release only after selecting a commit and documenting
   changes; do not call an untagged development branch a release.
4. Attach CI evidence and the manual validation record.
5. Preserve the raw inputs, calibration files, benchmark outputs, and known
   limitations with each field evaluation.

## Current status

The project is a feature-rich research/prototype application. It is not
field-validated or safety-certified. The real-camera validation checklist
remains intentionally separate from automated CI.


## New software validation surfaces

- Full SE(3) pose-graph optimisation: synthetic unit tests plus the manual trajectory comparison in VALIDATION.md.
- Rigid point-cloud ICP: synthetic transform-recovery tests plus known-object and measured-extrinsic comparisons.
- Stereo confidence map: bounded relative scores and zero for invalid depth; requires held-out depth-error analysis before probabilistic interpretation.
- Ground-truth trajectory evaluator: ATE and consecutive RPE using same-unit trajectories and first-pose alignment without scale fitting.

These additions expand the software prototype. They do not change the requirement for real camera validation before field-use claims.


## Dashboard additions

The dashboard exposes pairwise camera-rig composition, optional ICP refinement
for multi-camera PLY fusion, full SE(3) correction with manually supplied
relative-pose constraints, and stereo relative-confidence visualisation/export.
These interfaces improve workflow access but do not automatically verify
calibration targets, synchronize cameras, detect metric loop closures, or
certify registration correctness. The same independent-measurement release gate
continues to apply.


## Deployment readiness checks

Before publishing a hosted instance:

- Build the Docker image from the exact commit intended for deployment and record its commit SHA.
- Start the container and verify the Streamlit health endpoint from the host.
- Confirm the intended camera stream is reachable from the runtime network; the health endpoint does not test camera access.
- Confirm camera credentials are supplied through the deployment secret manager/environment and are absent from the image, repository, and logs.
- Verify CPU and memory limits with representative clips and point-cloud sizes; no universal production sizing is claimed.
- Restrict dashboard network access and use authenticated/restricted hosting when handling private camera feeds.
- Retain the CI run URL, image tag/digest, calibration inputs, benchmark outputs, and known limitations with the deployment record.
