# MuJoCo Controllers

Single-file pedagogical implementations of common robotics controllers in MuJoCo.

## Installation

MuJoCo is the *only dependency* required to run the controllers.

```bash
pip install "mujoco>=3.1.0"
```

## Usage

| File | Video | Description |
|------|-------|-------------|
|`diffik.py`|![image info](./images/ur5e.gif)|Differential IK on a 6-DOF UR5e.|
|`diffik_nullspace.py`|![image info](./images/panda.gif)|Differential IK with nullspace control on a 7-DoF Panda.|
|`opspace.py`|![image info](./images/iiwa.gif)|Operational space control on a 7-DOF KUKA iiwa14.|

## Acknowledgements

Robot models are taken from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
