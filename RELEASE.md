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
5. Docker image build.

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
- Stereo depth depends on calibration, synchronization, texture, and scene
  conditions.
- YOLO/ByteTrack masking only excludes detected tracked moving regions; missed
  and untracked objects can remain in the reconstruction.
- Stereo visual odometry is local and can accumulate drift.
- Pose-graph correction accepts manually supplied loop-closure constraints;
  it does not discover loop closures and currently adjusts translations only.
- Multi-camera point-cloud fusion requires externally verified camera-to-world
  poses in a common coordinate frame and distance unit; the dashboard and CLI do
  not estimate extrinsics or automatically register overlapping clouds.
- PLY import supports ASCII and binary little-/big-endian PLY 1.0 with scalar
  XYZRGB vertex properties; list-valued vertex properties are not supported.
- Synthetic benchmarks measure software throughput on the runner, not real
  camera accuracy or target-machine performance.
- Visual loop-closure candidate retrieval uses ORB matching and RANSAC geometry,
  but it does not generate metric pose constraints or confirm a loop automatically.
  Full 6DoF pose-graph optimisation remains future work; current correction uses
  manually verified translation constraints.

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
