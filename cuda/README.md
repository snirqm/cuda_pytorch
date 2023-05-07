# Matrix Multiplication Pytorch extention - with CUDA
## Building the project
First make sure you have python3.10 installed
```/usr/bin/bash
sudo apt-get update && sudo apt-get install python3.10 pip
```
Make sure to have CUDA11.8 installed
```/usr/bin/bash 
sudo apt-get update && sudo apt-get install -y wget 
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
``` 
(taken from [here](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local))

Install the project dependencies 
```bash
pip install -r requirements.txt
pip install dist/parallel_mult_cuda-0.1.0.tar.gz
```
(see [requirements.txt](./requirements.txt) for more info)

## Generating the report
```bash
python3.10 main.py
```
This script will create a subfolder `./report` with the needed plots
