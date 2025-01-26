# 3D-ACT

## Action Chunking Transformer for ALOHA integrated with 3DETR
Using 3D point cloud inputs instead of 2D images

## Installation
For 3D-ACT

```bash
conda create -n aloha python=3.8.10
conda activate aloha
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython
cd act/detr && pip install -e .
```

For 3DETR, you will also need the following packages (either conda install or pip install)

```bash
matplotlib
opencv-python
plyfile
'trimesh>=2.35.39,<2.35.40'
'networkx>=2.2,<2.3'
scipy
```

Also you will need to install pointnet2 layers by running

```bash
cd 3detr/third_party/pointnet2 && python setup.py install
```

**Optionally**, you can install a Cythonized implementation of gIOU for faster training.

```bash
conda install cython
cd 3detr/util && python cython_compile.py build_ext --inplace
```

## Train

```bash
./train.sh
```

You can change major hyperparameters inside the shell script. To change other hyperparameters, read the code and change them.
(*Planning to add config file and clean the code to tune the hyperparameters easily*)

## Evaluation
*Haven't tested yet. Only the train code have been tested. Instructions will be uploaded after testing the evaluation codes*