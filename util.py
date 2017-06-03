from skimage.feature import hog
from skimage.io import imread
from PIL import Image
import matplotlib.patches as patches
import numpy as np

import os
import glob
from tqdm import tqdm

def load_negative_samples():
    current_path = os.getcwd()
    pos_train_path_1 = os.path.join(current_path, 'Datasets', 'Tray', 'Ara2013-RPi')
    pos_train_path_2 = os.path.join(current_path, 'Datasets', 'Tray', 'Ara2013-Canon')
    pos_train_path_3 = os.path.join(current_path, 'Datasets', 'Tray', 'Ara2012')

    file_path = []
    for train_path in [pos_train_path_1, pos_train_path_2, pos_train_path_3]:
        for im_path in glob.glob(os.path.join(train_path, '*rgb.png')):
            bbox_path = im_path[:-7]+'bbox.csv'
            file_path.append((im_path, bbox_path))
    negative_samples = []
    for file_path_pair in tqdm(file_path):
        samples_from_img = generate_negative_samples(file_path_pair)
        for sample in samples_from_img:
            negative_samples.append(sample)
    return negative_samples

def load_positive_samples():
    current_path = os.getcwd()
    pos_train_path_1 = os.path.join(current_path, 'Datasets', 'Tray', 'Ara2013-RPi')
    pos_train_path_2 = os.path.join(current_path, 'Datasets', 'Tray', 'Ara2013-Canon')
    pos_train_path_3 = os.path.join(current_path, 'Datasets', 'Tray', 'Ara2012')

    file_path = []
    for train_path in [pos_train_path_1, pos_train_path_2, pos_train_path_3]:
        for im_path in glob.glob(os.path.join(train_path, '*rgb.png')):
            bbox_path = im_path[:-7]+'bbox.csv'
            file_path.append((im_path, bbox_path))
    positive_samples = []
    for file_path_pair in tqdm(file_path):
        samples_from_img = generate_positive_samples(file_path_pair)
        for sample in samples_from_img:
            positive_samples.append(sample)
    return positive_samples

def generate_negative_samples(file_path_pair, window_size=(224, 224), step_size=(50, 50)):
    im_path, csv_file_path = file_path_pair
    bboxes = np.loadtxt(csv_file_path, delimiter=',')

    im = imread(im_path)
    positive_patch = list()
    for row in bboxes:
        width = row[4] - row[0]
        height = row[3] - row[1]
        positive_patch.append([row[0], row[1], width, height])
    negative_samples = []
    for (x, y, im_window) in sliding_window(im, window_size, step_size):
        if (im_window.shape[0] != window_size[0] or im_window.shape[1] != window_size[1]):
            continue
        neg_candidate = (x, y, window_size[1], window_size[0])
        is_candidate = True
        for pos in positive_patch:
            if overlapping_plant(neg_candidate, pos) > 0.2:
                is_candidate = False
        if is_candidate:
            negative_samples.append(im_window)
    return negative_samples

def generate_positive_samples(file_path_pair, window_size=(224, 224), step_size=(20, 20)):
    im_path, csv_file_path = file_path_pair
    bboxes = np.loadtxt(csv_file_path, delimiter=',')

    im = imread(im_path)
    positive_patch = list()
    for row in bboxes:
        width = row[4] - row[0]
        height = row[3] - row[1]
        positive_patch.append([row[0], row[1], width, height])
    positive_samples = []
    for (x, y, im_window) in sliding_window(im, window_size, step_size):
        if (im_window.shape[0] != window_size[0] or im_window.shape[1] != window_size[1]):
            continue
        pos_candidate = (x, y, window_size[1], window_size[0])
        is_candidate = False
        for pos in positive_patch:
            if overlapping_center(pos_candidate, pos):
                is_candidate = True
        if is_candidate:
            for i in range(4):
                im_window = np.rot90(im_window)
                positive_samples.append(im_window)
    return positive_samples

def overlapping_center(detection_1, detection_2, radius=20):
    center_1 = np.array([detection_1[0] + detection_1[2]/2, detection_1[1] + detection_1[3]/2])
    center_2 = np.array([detection_2[0] + detection_2[2]/2, detection_2[1] + detection_2[3]/2])

    if np.sqrt(np.sum((center_1 - center_2)**2)) < radius:
        return True
    else:
        return False

def overlapping_area(detection_1, detection_2):
    '''
    Function to calculate overlapping area'si
    `detection_1` and `detection_2` are 2 detections whose area
    of overlap needs to be found out.
    Each detection is list in the format ->
    [x-top-left, y-top-left, width-of-detection, height-of-detection]
    The function returns a value between 0 and 1,
    which represents the area of overlap.
    0 is no overlap and 1 is complete overlap.
    Area calculated from ->
    http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    '''
    # Calculate the x-y co-ordinates of the 
    # rectangles
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[2]
    x2_br = detection_2[0] + detection_2[2]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[3]
    y2_br = detection_2[1] + detection_2[3]

    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[2] * detection_2[3]
    area_2 = detection_2[2] * detection_2[3]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

def overlapping_plant(detection_1, detection_2):
    '''
    Function to calculate overlapping area'si
    `detection_1` and `detection_2` are 2 detections whose area
    of overlap needs to be found out.
    Each detection is list in the format ->
    [x-top-left, y-top-left, width-of-detection, height-of-detection]
    The function returns a value between 0 and 1,
    which represents the area of overlap.
    0 is no overlap and 1 is complete overlap.
    Area calculated from ->
    http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    '''
    # Calculate the x-y co-ordinates of the 
    # rectangles
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[2]
    x2_br = detection_2[0] + detection_2[2]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[3]
    y2_br = detection_2[1] + detection_2[3]

    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    # area_1 = detection_1[2] * detection_2[3]
    area_2 = detection_2[2] * detection_2[3]
    # total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(area_2)

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window
    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in range(0, image.shape[0], step_size[0]):
        for x in range(0, image.shape[1], step_size[1]):
            yield (x, y, image[y:y + window_size[0], x:x + window_size[1]])


def nms(detections, threshold=.5):
    '''
    This function performs Non-Maxima Suppression.
    `detections` consists of a list of detections.
    Each detection is in the format ->
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
    If the area of overlap is greater than the `threshold`,
    the area with the lower confidence score is removed.
    The output is a list of detections.
    '''
    if len(detections) == 0:
        return []
    # Sort the detections based on confidence score
    detections = sorted(detections, key=lambda detections: detections[2],
            reverse=True)
    # Unique detections will be appended to this list
    new_detections=[]
    # Append the first detection
    new_detections.append(detections[0])
    # Remove the detection from the original list
    del detections[0]
    # For each detection, calculate the overlapping area
    # and if area of overlap is less than the threshold set
    # for the detections in `new_detections`, append the 
    # detection to `new_detections`.
    # In either case, remove the detection from `detections` list.
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            candidate_box = [detection[i] for i in [0, 1, 3, 4]]
            selected_box = [new_detection[i] for i in [0, 1, 3, 4]]
            if overlapping_area(candidate_box, selected_box) > threshold:
                del detections[index]
                break
        else:
            new_detections.append(detection)
            del detections[index]
    return new_detections