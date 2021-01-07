# EvoPose2D
Source code for EvoPose2D: Pushing the Boundaries of 2D Human Pose Estimation using Neuroevolution.
Implemented using Python 3.7 and TensorFlow 2.3.

**Proof of results**: The json files containing the results reported in the paper can be found [here](https://drive.google.com/drive/folders/1iO_EVuIYQP1BHG6A0hWaAmD52PA3C7HI?usp=sharing).
These results were generated using the bfloat16 models.

## Getting Started
1. If you haven't already, [install Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create a new conda environment with Python 3.7: ```$ conda create -n evopose2d python==3.7```
3. Clone this repo: ```$ git clone https://github.com/wmcnally/evopose2d.git```
4. Install the dependencies using ```$ pip install -r requirements.txt```
5. Download the 2017 COCO [training](http://images.cocodataset.org/zips/train2017.zip) and [validation](http://images.cocodataset.org/zips/val2017.zip) images.
6. Download the validation person [detections](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing) (from HRNet repo). 
7. Use [write_tfrecords.py](./write_tfrecords.py) and the detection json to generate the training and validation TFRecords. If using Cloud TPU, upload the TFRecords to a Storage Bucket. 


## Demo
1. Download a pretrained float32 model from [here](https://drive.google.com/drive/folders/1neywqc7OC5dRErQJFCnCor5fphkKCzHh?usp=sharing) and place it in a new ```models``` directory.
2. Run ```$ python3 demo.py -c [model_name].yaml -p [/path/to/coco/dataset]``` The test image result will be written to the main directory. Prediction visualization for image id 785 using evopose2d_M_f32 model:

![alt text](./demo_img.jpg)


## Validation
Download a pretrained model from [here](https://drive.google.com/drive/folders/1neywqc7OC5dRErQJFCnCor5fphkKCzHh?usp=sharing) and place it in a new ```models``` directory. The bfloat16 models run best on TPU, and might be slow on GPU. 

Modify the paths to the TFRecords and validation annotation json in the [yaml](./configs) file of the model you downloaded. If using GPU, change the validation batch size to suit your total GPU memory.

**GPU:** ```$ python3 validate.py -c [model_name].yaml```

**Cloud TPU:** ```$ python3 validate.py -c [model_name].yaml --tpu [tpu_name]```

## Training
Modify the paths to the TFRecords and validation annotation json in the [yaml](./configs) file of the model you want to train. If using GPU, change the training and validation batch sizes to suit your total GPU memory and set bfloat16 to 'false'.

**GPU:** ```$ python3 train.py -c [model_name].yaml```

**Cloud TPU:** ```$ python3 train.py -c [model_name].yaml --tpu [tpu_name]```

## Neuroevolution
Modify the paths to the TFRecords and validation annotation json in [E3.yaml](./configs/E3.yaml).

To run on 4 Cloud TPUs, e.g., with names [node-1, node-2, node-3, node-4]: ```$ python3 ga.py -c E3.yaml -a 1 2 3 4```

See [ga.py](./ga.py) arguments for more details. 

## Acknowledgements

Hardware: 
- NVIDIA GPU Grant
- [TensorFlow Research Cloud (TFRC) program](https://www.tensorflow.org/tfrc)

GitHub Repositories:
- [https://github.com/mks0601/TF-SimpleHumanPose](https://github.com/mks0601/TF-SimpleHumanPose)
- [https://github.com/microsoft/human-pose-estimation.pytorch](https://github.com/microsoft/human-pose-estimation.pytorch)
- [https://github.com/HRNet/HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation)
- [https://github.com/megvii-detection/MSPN](https://github.com/megvii-detection/MSPN)
- [https://github.com/mks0601/PoseFix_RELEASE](https://github.com/mks0601/PoseFix_RELEASE)