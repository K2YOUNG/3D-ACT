# 3D-ACT

## Action Chunking Transformer for ALOHA integrated with 3DETR
Using 3D point cloud inputs instead of 2D images

## Installation
You will need to install pointnet2 layers by running

```bash
cd 3detr/third_party/pointnet2 && python setup.py install
```

You will also need Python dependencies (either conda install or pip install)

```bash
matplotlib
opencv-python
plyfile
'trimesh>=2.35.39,<2.35.40'
'networkx>=2.2,<2.3'
scipy
```

**Optionally**, you can install a Cythonized implementation of gIOU for faster training.

```bash
conda install cython
cd 3detr/util && python cython_compile.py build_ext --inplace
```
