# Offline Execution Policy

This course targets local, offline execution at runtime.

## Non-negotiable runtime rules

1. No HTTP/HTTPS requests, cloud APIs, remote inference endpoints, telemetry, or
   automatic downloads from lesson programs.
2. Camera capture, image processing, geometry, tracking, visualization, and
   tests execute on the local machine.
3. Pretrained weights must be explicitly placed in a documented local directory.
   A lesson must fail with a clear message if a required checkpoint is absent;
   it must never silently download one.
4. Package installation and model acquisition are preparation steps, not runtime
   behavior. After preparing dependencies and weights, disconnect the network
   and run the lesson.
5. Synthetic data generation and tests must not need network access.
6. Any optional dependency must be imported only when its feature is selected.
7. A lesson must document CPU/GPU requirements, local paths, package versions,
   and whether it uses a pretrained checkpoint.

## Suggested local layout

    advanced_3d_scene_reconstruction/
      models/
        vggt/
        sam/
        depth/
        segmentation/
      data/
        recordings/
        calibration/
        test/
      outputs/
      common/
      camera_adapters/
      multi_camera/
      scene_reconstruction/
      temporal_mapping/

Do not commit large model weights or private camera recordings to Git. Store
them locally and document their expected filenames, checksums, source licenses,
and required versions in the relevant lesson README.

## Preparing an offline workstation

On a connected preparation machine, download only the required wheels and
licensed checkpoints. Transfer them to the target computer using approved
offline media. Install from the local wheel directory:

    python -m pip install --no-index --find-links ./wheelhouse -r requirements.txt

The wheelhouse must contain compatible wheels for the target OS, Python version,
CPU architecture, and CUDA runtime. This command intentionally disables package
index access.

## Model-loading requirements

Use explicit local checkpoint paths. Do not use model-loading options that
implicitly fetch missing weights. For example, a lesson should check that a
checkpoint file exists before constructing the model and should report the
expected local path if it does not.

## Verification

A lesson is considered offline-ready only after it passes its synthetic demo
and automated tests with network access disabled. Static source review is useful
but does not prove runtime network independence. Record the tested OS, Python,
package versions, GPU driver/CUDA versions (if applicable), and test output.

## Scope note

The offline policy applies to new and updated lesson code. Existing legacy
lessons that use third-party model wrappers must be reviewed individually:
some wrappers download pretrained weights automatically unless explicitly
configured with local checkpoint paths. Such lessons must not be described as
offline-verified until this review and disconnected execution test are complete.
