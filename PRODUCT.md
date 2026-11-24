# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML/CSS/JS (no build step, no npm), served with the page's JSON API by a Python
standard-library server running in WSL; viewed in the Windows browser at localhost. Simulations
run as subprocesses in the `sofa-env` conda environment (`python -m bowstring.experiment`).

## Users

One researcher (the project's author, robotics PhD level) exploring how a simulated three-segment
bowstring tendon-driven elastica arm behaves under its controller: changing robot, model,
controller, sensing and simulator parameters, running trajectories, watching the arm and reading
the errors. Used on both a laptop screen and a large monitor.

## Product Purpose

An exploratory console for the bowstring arm simulation: pick a sample trajectory, set every
parameter of the true robot, the controller's model of it, the controller, the sensing and the
simulator (the model's own plant, SOFA Cosserat, SOFA beams), run it, replay the arm, compare
runs, and sweep one parameter to see its effect. Success: the user builds intuition about the
robot's behaviour by asking "what if" and getting a trustworthy answer within a minute or two.

## Operating Context

- Runs take roughly 5 s to 2 min each (simulated 6–30 s). The model plant runs 2–3× faster than
  real time (Till et al. 2019 solver); SOFA is slower. Runs are queued and replayed, and the Live
  tab drives the model plant from the keyboard in real time.
- Results are files under `data/console/runs/<id>/` (config, status, result.json, result.npz).
- Units are non-dimensional: lengths in segment lengths, bending stiffness 1, time in seconds.

## Capabilities and Constraints

- Three segments only (the inverse map is square only for n = 3).
- Sample trajectories: circle, figure-eight, rounded rectangle (waiter), wrist rotation, line,
  hold; optional unannounced pushes.
- The robot (plant) and the controller's model can differ (robustness exploration).
- Sensing: tip IMU (navigation / tactical / consumer MEMS / ideal grades) with tendon-length
  encoders, IMU only, or perfect state.
- Controller: feedforward from the inverse map plus PID feedback in wrench space.
- Errors are measured against the commanded reference using the simulator's true state.

## Brand Commitments

- Visual direction chosen by the user (2026-11-23): the standard dark engineering dashboard,
  played straight. Craft bar: Foxglove Studio and Weights & Biases run pages (side panel of
  controls, dark canvas, coloured line charts).

## Evidence on Hand

Real simulation outputs only; no fabricated results. Known limits to state, not hide: SOFA's own
discretisation error (first order in element/section size), tendons inextensible, no hole friction.

## Product Principles

- Trustworthy numbers first: every plotted quantity says what it is measured against.
- Every parameter is reachable, and the defaults reproduce the documented results.
- Fast "what if" loops: change, run, compare, without leaving the page.
