import matplotlib.pyplot as plt
import cv2

def no_overlap(box_1, box_2):
    rand_x1, rand_y1, rand_x2, rand_y2 = box_1
    rx1, y1, x2, y2 = box_2
    if (rand_x2 < x1): return True
    if (rand_x1 > x2): return True
    if (rand_y2 < y1): return True
    if (rand_y1 > y2): return True

def display_patches(patches, num_columns=5):
    num_patches = patches.shape[0]
    num_rows = (num_patches + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(3*num_columns, 3*num_rows),squeeze=False)

    for i in range(num_rows):
        for j in range(num_columns):
            index = i*num_columns + j
            if index < num_patches:
                plt_image = cv2.cvtColor(patches[index], cv2.COLOR_BGR2RGB)
                axes[i, j].imshow(plt_image)
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')

    plt.show()

def display_features(feature_map, num_columns=5):
    num_features = feature_map.shape[0]
    num_channels = feature_map.shape[-1]
    num_patches = num_features * num_channels
    num_rows = (num_patches + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(3*num_columns, 3*num_rows),squeeze=False)

    for i in range(num_rows):
        for j in range(num_columns):
            index = i*num_columns + j
            index_feature = index // num_channels
            index_channel = index % num_channels
            if index < num_patches:
                plt_image = feature_map[index_feature, :, :, index_channel]
                axes[i, j].imshow(plt_image, cmap='gray')
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')

    plt.show()

def get_training_examples():
    training_annotation = [
        ["data/examples/picture_01.jpg", [273, 144, 364, 235]], 
        ["data/examples/picture_02.jpg", [515, 694, 644, 817]]]
    return training_annotation

def get_detect_image():
    test_file = 'data/examples/picture_01.jpg'
    return cv2.imread(test_file)
