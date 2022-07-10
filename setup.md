# Installation

1. install python packages according to requirements.txt
2. install [MinkowskiEngine](https://nvidia.github.io/MinkowskiEngine/)
3. install other packages:
	- openblas
	- ninja
4. create link ```ln -s ./models_ models```
5. setup your dataset path in ```lib/dataset/x.py``` or ```config.py```

# Trouble Shooting

- fix the bug of MinkowskiEngine to support custom-kernel shape
	- find the installation path of minkowskiengine e.g., ```lib/python3.7/site-packages/MinkowskiEngine/MinkowskiKernelGenerator.py```
	- insert this line to line 122:   ```kernel_size=np.array(kernel_size)```

- ```undefined symbol: _ZNK2at6Tensor7is_cudaEv```
	- make sure that the MinkowskiEngine and pytorch uses the same cuda version
