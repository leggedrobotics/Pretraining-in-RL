# About

The concept of Pedipulation - the manipulation of objects using the robot’s foot - was first introduced by Philip Arm et al in their paper "Pedipulate: Enabling Manipulation Skills using a Quadruped Robot's Leg" and implemented in RSL's internal legged_gym repository on the feature/pedipulation/no_rnd branch.
The project was continued by Jonas Stolle as part of his Master Thesis, where perception was added to enable obstacle avoidance. As part of this work, the implementation was migrated to Isaac Orbit and worked on on various branches following the dev/jonas/pedipulation/* naming. With the move from Isaac Orbit to Isaac Lab, the pedipulation codebase was migrated yet again, with a focus on a clean implementation, readability and the removal of unused code.

# How to Use

Follow the official documentation on how to install Isaac Lab.
Follow the main README in this repo to add isaac.locoma to your existing Isaac Orbit installation.

You will need to import this package (isaac.locoma) in your standalone files (e.g. train.py or play.py) in order for the added environments to by registered by gym.

`import isaac.locoma`


## Commands

The following provides commands to run the training and playing of the different pedipulation environments.

Execute from the Isaac Lab directory.

### Blind, Flat

**train**: `$ ./isaaclab.sh -p scripts/workflows/rsl_rl/train.py --task Pedipulation-Flat-Blind-Anymal-D-v0 --headless`
**play**: `$ ./isaaclab.sh -p scripts/workflows/rsl_rl/play.py --task Pedipulation-Flat-Blind-Anymal-D-Play-v0 --num_envs 4`

### Blind, Rough

**train**: `$ ./isaaclab.sh -p scripts/workflows/rsl_rl/train.py --task Pedipulation-Flat-Rough-Anymal-D-v0 --headless`
**play**: TODO


### Tensorboard

Very useful to monitor training runs.

`tensorboard --logdir /path/to/your/logdir`

Note: The isaaclab python environment needs to be activated for this to work.


# Code structure

The pedipulation task heavily relies heavily on inheritance to avoid duplicate code. A base config class is provided, defining the configurations that are elemental for pedipulation and (likely) don't need to be changed in the child classes. The base *_PLAY environment for example enables debug visualizations, turns of randomizations and increases the command resampling frequency.