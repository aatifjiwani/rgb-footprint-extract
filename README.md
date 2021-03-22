## A Semantic Segmentation Network for Urban-Scale Building Footprint Extraction Using RGB Satellite Imagery

This repository is the official implementation of **"A Semantic Segmentation Network for Urban-Scale Building Footprint Extraction Using RGB Satellite Imagery"** by Aatif Jiwani, Shubhrakanti Ganguly, Chao Ding, Nan Zhou, and David Chan. 

#### Include Graphic Here

### Requirements

1. To install GDAL/`georaster`, please follow this [doc](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html) for instructions. 
2. Install other dependencies from requirements.txt
```setup
pip install -r requirements.txt
```

### Datasets

#### Downloading the Datasets
1. To download the AICrowd dataset, please go [here](https://www.aicrowd.com/challenges/mapping-challenge-old). You will have to either create an account or sign in to access the training and validation set. Please store the training/validation set inside `<root>/AICrowd/<train | val>` for ease of conversion.
2. To download the Urban3D dataset, please run:
```setup
aws s3 cp --recursive s3://spacenet-dataset/Hosted-Datasets/Urban_3D_Challenge/01-Provisional_Train/ <root>/Urban3D/train
aws s3 cp --recursive s3://spacenet-dataset/Hosted-Datasets/Urban_3D_Challenge/02-Provisional_Test/ <root>/Urban3D/test
``` 
3. To download the SpaceNet Vegas dataset, please run:
```setup
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_2_Vegas.tar.gz <root>/SpaceNet/Vegas/
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_2_Vegas_Test_public.tar.gz <root>/SpaceNet/Vegas/

tar xvf <root>/SpaceNet/Vegas/SN2_buildings_train_AOI_2_Vegas.tar.gz
tar xvf <root>/SpaceNet/Vegas/AOI_2_Vegas_Test_public.tar.gz
```

#### Converting the Datasets

Please use our provided dataset converters to process the datasets. For all converters, please look at the individual files for an example of how to use them. 
1. For AICrowd, use `datasets/converters/cocoAnnotationToMask.py`. 
2. For Urban3D, use `datasets/converters/urban3dDataConverter.py`.
3. For SpaceNet, use `datasets/converters/spaceNetDataConverter.py`

#### Creating the Boundary Weight Maps

In order to train with the exponentially weighted boundary loss, you will need to create the weight maps as a pre-processing step. Please use `datasets/converters/weighted_boundary_processor.py` and follow the example usage. The `inc` parameter is specified for computational reasons. Please decrease this value if you notice very high memory usage. 

**Note:** these maps are not required for evaluation / testing. 
