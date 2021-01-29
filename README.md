# BoxDemo

## Basic requirements
```bash
python == 3.6
```

## Installation
### 1. numpy, opencv, and open3d
```bash
$pip install -r requirements.txt
```
### 2. pcl
```bash
$pip install cython==0.26.0
```
```bash
$sudo apt-get install libpcl-dev pcl-tools
```

### 3. python-pcl
```bash
$git clone https://github.com/strawlab/python-pcl.git
$cd python-pcl
```

Before build install python-pcl, edit the setup.py to avoid possible conflicts:

1. line 728: change vtk_version = '7.0' to vtk_version = '6.3'

2. line 752: remove some unnecessary libs, including: vtkexpat, vtkfreetype, vtkgl2ps, vtkhdf5, vtkhdf5_hl, vtkjpeg, vtkjsoncpp, vtklibxml2, 
vtkNetCDF, vtkNetCDF_cxx, vtkoggtheora, vtkpng, vtkproj4, vtksqlite, vtktiff, vtkzlib

Then, install python-pcl
```bash
$python setup.py build_ext -i
$python setup.py install
```

## Usage
```bash
$python example.py
```