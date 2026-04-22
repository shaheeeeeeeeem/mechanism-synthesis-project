## Mechanism Synthesis Project

This project contains a Python-based analytical synthesis and simulation of an
offset slider-crank mechanism for a partial circular body. The goal is to move
the body through three required precision positions:

- Position A: reference orientation and height
- Position B: 5 mm downward translation with a 45 degree clockwise rotation
- Position C: 15 mm downward translation with a 90 degree clockwise rotation

The mechanism is modeled so that the body center moves only vertically, while
the return stroke happens twice as fast as the forward stroke. The script
computes and visualizes:

- the synthesized slider-crank geometry
- kinematic motion through A, B, and C
- velocity and acceleration quantities for the joints and output body center
- loop-closure vector diagrams for velocity and acceleration
- dynamic input force and torque trends
- an animated simulation of the mechanism motion

## Main File

The primary script for the project is:

- `mainSynthesis.py`

## Generated Outputs

Running the script produces figure and animation files such as:

- `kinematics.png`
- `loop_closure_diagrams.png`
- `precision_positions.png`
- `dynamics.png`
- `mechanism_animation.gif`

## Notes

The current implementation is organized as a single main synthesis and
simulation script for ease of submission and presentation. The repository keeps
`.gitignore` and this README alongside the main program file.
