# 3D Bounding Box Estimation Using Deep Learning and Geometry

A Tensorflow implementation of the paper: Mousavian, Arsalan, et al. [3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496) by Fu-Hsiang Chan.

The aim of this project is to predict the size of the bounding box and orientation of the object in 3D space from a single two dimensional image.
 
## Prerequisites
1. TensorFlow
2. Numpy
3. OpenCV
4. tqdm

## Installation
1. Clone the repository
   ```Shell
   git clone https://github.com/smallcorgi/3D-Deepbox.git
   ```
2. Download the KITTI object detection dataset, calib and label (http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
3. Download the weights file (vgg_16.ckpt).
   ```Shell
   cd $3D-deepbox_ROOT
   wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
   tar zxvf vgg_16_2016_08_28.tar.gz
   ```
4. Compile evaluation code 
   ```Shell
   g++ -O3 -DNDEBUG -o ./kitti_eval/evaluate_object_3d_offline ./kitti_eval/evaluate_object_3d_offline.cpp
   ```
5. [KITTI train/val split used in 3DOP/Mono3D/MV3D](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz)

## Usage

### Train model
   ```Shell
   python main.py --mode train --gpu [gpu_id] --image [train_image_path] --label [train_label_path] --box2d [train_2d_boxes]
   ```

### Test model
   ```Shell
   python main.py --mode test --gpu [gpu_id] --image [test_image_path] --box2d [test_2d_boxes_path] --model [model_path] --output [output_file_path]
   ```

### Evaluation on kitti
   ```Shell
   ./kitti_eval/evaluate_object_3d_offline [ground_truth_path] [predict_path]
   ```

## References
1. https://github.com/shashwat14/Multibin
2. https://github.com/experiencor/didi-starter/tree/master/simple_solution
3. https://github.com/experiencor/image-to-3d-bbox
4. https://github.com/prclibo/kitti_eval
