# PyACF

This is a naive python implementation of Aggregated Channel Feature classifier. For simplicity, it does not follow all the advanced detals in (Piotr's Computer Vision)[https://github.com/pdollar/toolbox] who is the author of ACF. The speed of detection is extremely slow. Therefore, the current version is only for education purpose and not suitable for any comercial use. 

## (Optional) For Developers

Developers could use the development branch. This project still miss the following features: 

1. Fast pyramid feature algorithm is not implemented, the detection is extremely slow. See (this paper)[https://ieeexplore.ieee.org/document/6714453]
2. Multiple-class detection is not implemented, one-class detection only
3. The soft cascaded classifier is not implemented, it is simplified into a single classifier
4. The max/min size of detecting sliding window is 80 x 80/16 x 16 or manually set

This repo need contributor help on issue 1 particullary. We need a fast detection implementation. Some have already implement a (C++ verions of fast detection)[https://github.com/elucideye/acf]. The MATLAB officially suppports (compile the fast detection function)[https://ww2.mathworks.cn/help/vision/ug/example-ObjectDetectionFromImages.html] into C++. However, none of them provide any port of training function code. A possible solution is using the training code of single AdaBoost classifier in our repo and adapt it to the detection function. However, the detection function expected a cascaded classifier instead of a simple classifier. To walkaround that before we solve the issue 3, we may set the thresholds of classifier 0 except the last one.

## Files

The repository have data folder. The data folder should contain two subfolders named "positives" and "negatives", and one annotation file named "label.csv". The positive subfolder includes images with the face in picture, and negative subfolder includes only the images that has no human face, mostly nature landscape. The label csv file have coordinates of bounding boxes in positive images. 

The model folder stores the training results. After training, the model parameters are stored in the path address. 'model/weights.pkl'is default address.

The demo.ipynb file contains a quick tutorial of how to use this package.

## Quick Start

Download the source code directly

```bash
git clone https://github.com/cbw5803/PyACF.git
```

Go into the PyACF directory, install the required packages by run the command
```bash
pip install -r requirements.txt
```

Now open jupyter notebook or any python editor you like to run the code. Import the following library first
```python
import cv2
import pyacf as acf
```

Prepare all training data in one folder, here we use 'data' for our example and run the following command to get the input data. You can replace 'data' folder into any your folder name.

```python
positive_array, negative_array = acf.preprocess(folder='data')
```

Check the shape of positive and negetive data. In our example dataset, we labelled 76 positive patches, and we also prepared 350 negative patches. Therefore, the shape of positive_array should be 76 x 80 x 80 x 3 and the negative_array should be 350 x 80 x 80 x 3

```python
print(positive_array.shape, negative_array.shape)
```

We use the train() funciton to train a casacaded adaboost classifier. After training, the model parameters are stored in the path address. 'model/weights.pkl'is default address. The training process takes 5 minites

```python
acf.train(positive_array, negative_array, path='model/weights.pkl')
```

he detect() function will load the AdaBoost classifier and use a sliding window to find candidate face patches, and eventually use NMS to pick the most possible bounding box. Since this implementation is a naive solution, the detecting speed is EXTREMELY slow. You have to wait 30 minutues to finish. After about 8 asterisks displayed, the detection will be finished.

(This paper)[https://ieeexplore.ieee.org/document/6714453] is the actual ACF implementation that can detect face fast, but I did not implemented the algorithm. If you are interested to make it fast, you can read this paper and do a project to improve the detect speed.

In this test picture, for simplicity we manually use 51 x 51 as the only size for sliding window. To reduce false positive, we also increase the threshold from standard 0.5 to 0.98. To reproduce the result, the random state of AdaBoost had been changed into fixed value 6.

We need import opecv to read data from jpeg file

```python
img = cv2.imread('data/face_test.jpeg')
bboxes = acf.detect(img, min_threshold=0.98, window_size=[51])
```

Finally use the display() to plot the results via matplotlib. The scores are also annotated with the bounding box.

```python
acf.display(bboxes, img)
```

![](result.png)