# Run
1. Requirements:
    * python 3.6.8, pytorch 1.5.0, torchvision 0.6.0, cuda 10.1

2. Steps to run:
    * Step1:  put the model weight`rcnn_emd_simple_mge.pth` on `./model/rcnn_emd_simple/outputs`.
              then, put the test images in `./test_imgs/`
	```
	cd tools
	python inference.py
	```
    
	* Note:  the result will be saved in `./tools/outputs`