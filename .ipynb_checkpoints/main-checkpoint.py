import numpy as np
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def preprocess(folder):
    csv_label_file = os.path.join(folder,'label.csv')
    negative_folder = os.path.join(folder, 'negatives/')
    annotations = read_csv_annotation(csv_label_file)
    positive_array = positive_patches(annotations)
    negative_array = negative_patches(negative_folder)
    return positive_array, negative_array

def read_csv_annotation(label_file='data/label.csv'):
    training_data = []
    try:
        with open(label_file, 'r', encoding='utf-8') as file:
            i = 0
            for line in file:
                # skip the first headline
                if i == 0:
                    i += 1
                    continue
                filename, x1, y1, x2, y2, w, h = line.strip().split(',')
                box = [int(x1), int(y1), int(x2), int(y2)]
                training_data.append([filename, box])
                i += 1
    except FileNotFoundError:
        print("File not found.")
    return training_data

def negative_patches(negative_folder='data/negatives/', crop_per_image = 25, window_size=(80, 80)):
    # do not show hidden files
    negative_files = [f for f in os.listdir(negative_folder) if not f.startswith('.')]
    negative_patches = []
    
    for negative_file in negative_files:
        full_path = os.path.join(negative_folder, negative_file)
        img = img = cv2.imread(full_path)
        if img is None:
            print(full_path," is not found\n")
            continue
            
        # Add negative patches
        i = 0
        while i < crop_per_image:
            # Randomly generate bounding box coordinates
            rand_x1 = np.random.randint(0, img.shape[1] - window_size[0])
            rand_y1 = np.random.randint(0, img.shape[0] - window_size[1])
            rand_x2 = rand_x1 + window_size[0]
            rand_y2 = rand_y1 + window_size[1]
            negative_patches.append(img[rand_y1:rand_y2, rand_x1:rand_x2])
            i += 1
    negative_array = np.stack(negative_patches)
    return negative_array

def positive_patches(training_data_list):
    positive_patches = []

    for data in training_data_list:
        # Load image
        img = cv2.imread(data[0])
        if img is None:
            print(data[0]," is not found\n")
            continue

        # Extract bounding box coordinates
        x1, y1, x2, y2 = data[1]
        
        # Crop and resize positive patch
        positive_patch = cv2.resize(img[y1:y2, x1:x2], (80, 80))
        positive_patches.append(positive_patch)

    positive_array = np.stack(positive_patches)
    return positive_array


def aggregated_feature(patches):
    # patches: L x 80 x 80 x 3 input
    L = patches.shape[0]

    # Initialize empty list to store aggregated features
    aggregated_features = np.zeros((L, 20, 20, 10))

    # Loop through each patch. Vectorize into batch procession taking too much effort
    for i in range(L):
        # Convert RGB patch to HSV
        hsv_patch = cv2.cvtColor(patches[i], cv2.COLOR_BGR2HSV)
        
        # Downsample by averaging over 4x4 blocks
        # Store HSV channels as the first 3 channels
        downsampled_patch = cv2.resize(hsv_patch, (20, 20))
        aggregated_features[i,:, :, :3] = downsampled_patch
        
        # Calculate image gradient magnitude and direction
        gradient_x = cv2.Sobel(cv2.cvtColor(patches[i], cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(cv2.cvtColor(patches[i], cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        # down sample to 20 x 20 by stride
        new_shape = (20, 20, 4, 4)
        new_strides = (gradient_magnitude.strides[0] * 4, gradient_magnitude.strides[1] * 4,
                       gradient_magnitude.strides[0], gradient_magnitude.strides[1])
        
        # apply mean for the 4-th channel
        downsampled_magnitude = np.lib.stride_tricks.as_strided(gradient_magnitude, shape=new_shape, strides=new_strides)
        downsampled_magnitude = downsampled_magnitude.reshape(20,20,16)
        aggregated_magnitude = np.mean(downsampled_magnitude, axis=2)
        aggregated_features[i, :, :, 3] = aggregated_magnitude
        
        # apply six bin of direction for 5-th to 10-th
        downsampled_direction = np.lib.stride_tricks.as_strided(gradient_direction, shape=new_shape, strides=new_strides)
        downsampled_direction = downsampled_direction.reshape(20,20,16)
        scale = 16
        aggregated_features[i, :, :, 4] = scale*np.sum((downsampled_direction >= -np.pi) & (downsampled_direction < -np.pi*2/3), axis=2)
        aggregated_features[i, :, :, 5] = scale*np.sum((downsampled_direction >= -np.pi*2/3) & (downsampled_direction < -np.pi*1/3), axis=2)
        aggregated_features[i, :, :, 6] = scale*np.sum((downsampled_direction >= -np.pi*1/3) & (downsampled_direction < 0), axis=2)
        aggregated_features[i, :, :, 7] = scale*np.sum((downsampled_direction >= 0) & (downsampled_direction < np.pi*1/3), axis=2)
        aggregated_features[i, :, :, 8] = scale*np.sum((downsampled_direction >= np.pi*1/3) & (downsampled_direction < np.pi*2/3), axis=2)
        aggregated_features[i, :, :, 9] = scale*np.sum((downsampled_direction >= np.pi*2/3) & (downsampled_direction < np.pi), axis=2)
    
    # return features: L x 20 x 20 x 10 output
    return aggregated_features

def train(positive_examples, negative_examples):
    # the main training function
    # examples: L x 80 x 80 x 3 input
    positive_aggregated = aggregated_feature(positive_examples)
    negative_aggregated = aggregated_feature(negative_examples)
    
    # positive_aggregated: N x 20 x 20 x 10
    # negative_aggregated: 5N x 20 x 20 x 10
    positive_flat = positive_aggregated.reshape(positive_aggregated.shape[0],-1)
    negative_flat = negative_aggregated.reshape(negative_aggregated.shape[0],-1)
    
    # vstack N x 4000, 5N x 4000 -> 3N x 4000
    X_train = np.vstack((positive_flat, negative_flat))
    # hstack 1-D reshaped to (1 x) N, (1 x) 5N -> (1 x) 3N
    y_train = np.hstack(
        (np.ones(positive_examples.shape[0]), np.zeros(negative_examples.shape[0])))
    
    #TODO New in version 1.2: base_estimator was renamed to estimator
    base_classifier_1 = DecisionTreeClassifier(max_depth=2)
    clf_0 = AdaBoostClassifier(
        base_estimator=base_classifier_1,n_estimators=23, random_state=6)
    base_classifier_2 = DecisionTreeClassifier(max_depth=2)
    clf_1 = AdaBoostClassifier(
        base_estimator=base_classifier_1,n_estimators=128, random_state=6)
    base_classifier_3 = DecisionTreeClassifier(max_depth=2)
    clf_2 = AdaBoostClassifier(
        base_estimator=base_classifier_1,n_estimators=512, random_state=6)
    base_classifier_4 = DecisionTreeClassifier(max_depth=2)
    clf_3 = AdaBoostClassifier(
        base_estimator=base_classifier_1,n_estimators=2048, random_state=6)
    
    classifiers = [clf_0, clf_1, clf_2, clf_3]
    i = 0
    for clf in classifiers:
        print("-"*10)
        print("input shape: ", X_train.shape, "label shape:", y_train.shape)
        print("train the ",i," classifier...")
        clf.fit(X_train, y_train)
        print("done.")
        i += 1
        
    print("-"*10)
    print("optimize the thresholds for stages...")
    best_parameters = stage_thresholds(positive_flat, negative_flat, classifiers, 7)
    print("find best thresholds:",best_parameters)
    print("done.")

    print("-"*10)
    print("save the model...")
    save_model(best_parameters, classifiers)
    print('done.')
        
    return None

def stage_thresholds(positive_flat, negative_flat, classifiers, N):
    # positive_examples: N x 4000
    # negative_examples: 5N x 4000
    # classifiers: 4 trained classifiers
    
    parameter_combinations = []
    for i in range(N+1):
        for j in range(i+1):
            for k in range(j+1):
                parameter_combinations.append((k*0.5/N, j*0.5/N, i*0.5/N))
    print("try combinations:",len(parameter_combinations))
    
    # optimize 7 x 7 x 7 to get best accuracy or F1 scores thresholds
    metrics = []
    for each_combination in parameter_combinations:
        metric = cascaded_metric(
            positive_flat, negative_flat, each_combination, classifiers)
        metrics.append(metric)
    best_i = np.argmax(metrics)

    # return stage_thresholds : 3
    return parameter_combinations[best_i]

def cascaded_metric(positive_flat, negative_flat,
                   parameters, classifiers, metrics='accuracy'):
    # positive_flat: N x 4000
    # negative_flat: 5N x 4000
    # parameters: threshold for each stage
    # classifiers: 4 trained classifiers
    y_positive = cascaded_predict(positive_flat, parameters, classifiers)
    TP = y_positive.sum()
    FN = y_positive.shape[0] - TP
    
    y_negative = cascaded_predict(negative_flat, parameters, classifiers)
    FP = y_negative.sum()
    TN = y_negative.shape[0] - FP
    
    if metrics == 'F1':
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        return 2*recall*precision/(recall + precision)
    else:
        return (TP+TN)/(TP+FP+FN+TN)

def cascaded_predict(examples, parameters, classifiers):
    # examples: L x 4000
    # parameters: threshold for each stage
    # classifiers: 4 trained classifiers
    clf_0, clf_1, clf_2, clf_3 = classifiers
    threshold_0, threshold_1, threshold_2 = parameters
    
    y = np.zeros(examples.shape[0])
    selected_0 = clf_0.predict_proba(examples)[:,1] > threshold_0
    output_0 = examples[selected_0]
    if output_0.shape[0] == 0: return y
    y_0 = y[selected_0]
    
    selected_1 = clf_1.predict_proba(output_0)[:,1] > threshold_1
    output_1 = output_0[selected_1]
    if output_1.shape[0] == 0: return y
    y_1 = y_0[selected_1]
    
    selected_2 = clf_2.predict_proba(output_1)[:,1] > threshold_2
    output_2 = output_1[selected_2]
    if output_2.shape[0] == 0: return y
    y_2 = y_1[selected_2]
          
    y_1[selected_2] = y_2 = clf_3.predict(output_2)
    y_0[selected_1] = y_1
    y[selected_0] = y_0
    return y


def save_model(stage_thresholds, classifiers, path='model/weights.pkl'):
    # stage_thresholds: 3
    # classifiers: 4
    
    with open(path, "wb") as file:
        pickle.dump([stage_thresholds, classifiers], file)
    return None

def non_maximum_suppression(boxes_and_scores, iou_threshold):
    # boxes_and_scores: M x 5 input
    # M is number of boxes, 5 are x1, y1, x2, y2 and score
    # https://zhuanlan.zhihu.com/p/78504109
    x1 = boxes_and_scores[:, 0] 
    y1 = boxes_and_scores[:, 1]
    x2 = boxes_and_scores[:, 2] 
    y2 = boxes_and_scores[:, 3] 
    scores = boxes_and_scores[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    # reverse order, argsort provides the index
    order = scores.argsort()[::-1] 
    keep = [] 
    while order.size > 0: 
        keep.append(order[0]) # keep the index of the highest score
        # get all intersection between the highest and rest by broadcast
        xx1 = np.maximum(x1[order[0]], x1[order[1:]]) 
        yy1 = np.maximum(y1[order[0]], y1[order[1:]]) 
        xx2 = np.minimum(x2[order[0]], x2[order[1:]]) 
        yy2 = np.minimum(y2[order[0]], y2[order[1:]]) 
        # area of all intersection, 0 if no intersection 
        w = np.maximum(0.0, xx2 - xx1 + 1) 
        h = np.maximum(0.0, yy2 - yy1 + 1) 
        intersection = w * h 
        iou = intersection / (areas[order[0]] + areas[order[1:]] - intersection) 
        # np.where() return a tuple here, must use [0] to collect the indexes
        iou_index = np.where(iou <= iou_threshold)[0]
        # the iou array length is 1 less than order array
        index = iou_index + 1
        # select by index, generate new order
        order = order[index]
    # return predictions: M x 5 output
    return keep

def sliding_window(image, window_size, step_size=4):
    for y in range(0, image.shape[0] - window_size[0] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[1] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0],:])

            
def get_scales(max_scale=80, min_scale=16, num_octave=8):
    factor = 0.5**(1/num_octave)
    scales = []
    scale = max_scale
    while scale >= min_scale:
        scales.append(scale)
        scale = round(scale * factor)
    return scales


def detect(image, path='./model/weights.pkl', debug=True, fix_window_51=False):
    if debug:
        min_threshold=0.5
        window_level=1
    else:
        min_threshold=0
        window_level=100

    # image: x, y, 3 input, reshape into
    print("image size:", image.shape)
    
    # by default, max window size 80 x 80 and min window size 16 x 16
    sliding_window_sizes = get_scales()
    
    with open(path, "rb") as file:
        stage_thresholds, classifiers = pickle.load(file)
    clf_0, clf_1, clf_2, clf_3 = classifiers
    threshold_0, threshold_1, threshold_2 = stage_thresholds
    
    # number of sample too small
    threshold_0 = max(min_threshold, threshold_0)
    threshold_1 = max(min_threshold, threshold_1)
    threshold_2 = max(min_threshold, threshold_2)
    threshold_3 = max(min_threshold, 0.5)
    
    candidates = []
    if fix_window_51:
        sliding_window_sizes = [51]

    for window_size in sliding_window_sizes[:window_level]:
        print("window size:", window_size)
        i = 0
        for x, y, patch in sliding_window(image, (window_size,window_size)):
            if i%1000 == 0:print('*', end='', flush=True)
            i += 1
            
            patch_resized = cv2.resize(patch, (80, 80)).reshape(1,80,80,3)
            feature_flat = aggregated_feature(patch_resized).reshape(1,-1)

            if clf_0.predict_proba(feature_flat)[0][1] <= threshold_0:
                continue
            if clf_1.predict_proba(feature_flat)[0][1] <= threshold_1:
                continue
            if clf_2.predict_proba(feature_flat)[0][1] <= threshold_2:
                continue
            score = clf_3.predict_proba(feature_flat)[0][1]
            if score <= 0.5:
                continue
            candidates.append([x,y,x+window_size,y+window_size,score])
        print("\nFind candidates:", len(candidates))
    
    # send to non_maximum_suppression
    print("run NMS...")
    selects = non_maximum_suppression(np.array(candidates,dtype=np.float64), 0.25)
    print("final result: ", len(selects))
    
    predictions = []
    for select in selects:
        predictions.append(candidates[select])
    # predictions: M x 5 output
    # M is number of boxes, 5 columns are x1, y1, x2, y2, confidence
    return predictions

def display(predictions, image):
    # load image
    plt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(20, 16))

    # Display the image
    ax.imshow(plt_image)
    
    for x1, y1, x2, y2, score in predictions:

        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='green', facecolor='none')

        # Add the bounding box to the plot
        ax.add_patch(rect)
        ax.text(x1, y1, f'score: {score:.2f}', color='green')
        
    # Display the plot
    plt.show()


    