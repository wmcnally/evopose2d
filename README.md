# EvoPose2D
Source code for EvoPose2D: Pushing the Boundaries of 2D Human Pose Estimation using Neuroevolution.
Implemented using Python 3.7 and TensorFlow 2.3.

**Proof of Results:** The json files containing the results reported in the paper are available [here](https://drive.google.com/drive/folders/1nNrB0o7Uo7gpGE3_F3L2Ukn9R47PIRrF?usp=sharing). 

## Validation / Testing
1. Install the packages listed in [requirements.txt](./requirements.txt).
2. Download a pretrained model: [[bfloat16](https://drive.google.com/drive/folders/1lPXkml5icmKLOGr3o2FQsHEPPe7Du_cH?usp=sharing), float32 (coming soon)]. The bfloat16 models run best on TPU, and might be slow on older GPUs. 
3. Download the COCO 2017 [validation](http://images.cocodataset.org/zips/val2017.zip) / [test](http://images.cocodataset.org/zips/test2017.zip) images.
4. Download the [validation](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) / [test](http://images.cocodataset.org/annotations/image_info_test2017.zip) annotations.
5. Download the [person detections](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing) (from HRNet repo). 
6. Use [write_tfrecords.py](./write_tfrecords.py) and the detection json to generate the validation / test TFRecords.

## Training
Coming soon...




 

##Acknowledgements
We would like to acknowledge the following repositories:
- [https://github.com/mks0601/TF-SimpleHumanPose](https://github.com/mks0601/TF-SimpleHumanPose)
- [https://github.com/microsoft/human-pose-estimation.pytorch](https://github.com/microsoft/human-pose-estimation.pytorch)
- [https://github.com/HRNet/HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation)
- [https://github.com/mks0601/PoseFix_RELEASE](https://github.com/mks0601/PoseFix_RELEASE)