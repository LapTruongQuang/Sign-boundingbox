import time 
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET
import pickle
import zipfile

from skimage.transform import resize
from skimage import feature
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def readData():
  img_lst = []
  label_lst = []
  annotations_dir = './traditionalMLApp/data/traffic_sign_detection/annotations'
  img_dir = './traditionalMLApp/data/traffic_sign_detection/images'
  for xml_file in os.listdir(annotations_dir):
    xml_filepath = os.path.join(annotations_dir, xml_file)
    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    folder = root.find('folder').text
    img_filename = root.find('filename').text
    img_filepath = os.path.join(img_dir, img_filename)
    img = cv2.imread(img_filepath)

    for obj in root.findall('object'):
      classname = obj.find('name').text
      if classname == 'trafficlight': #we ignore trafficlight' class since we do traffic sign classification
        continue
      
      xmin = int(obj.find('bndbox/xmin').text)
      ymin = int(obj.find('bndbox/ymin').text)
      xmax = int(obj.find('bndbox/xmax').text)
      ymax = int(obj.find('bndbox/ymax').text)

      object_img = img[ymin:ymax, xmin:xmax]
      img_lst.append(object_img)
      label_lst.append(classname)
  return img_lst, label_lst

def preprocess_img(img):
  if len(img.shape) > 2:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = img.astype(np.float32)

  resized_img = resize(
      img,
      output_shape=(32, 32),
      anti_aliasing=True
  )

  hog_feature = feature.hog(
      resized_img,
      orientations=9,
      pixels_per_cell=(8,8),
      cells_per_block=(2,2),
      transform_sqrt=True,
      block_norm='L2',
      feature_vector=True
  )
  return hog_feature

def sliding_window(img, window_sizes, stride, scale_factor):
    img_height, img_width = img.shape[:2]
    windows = []
    for window_size in window_sizes:
        window_width, window_height = window_size
        for ymin in range(0, img_height - window_height + 1, stride):
            for xmin in range(0, img_width - window_width + 1, stride):
                xmax = xmin + window_width
                ymax = ymin + window_height
                windows.append([xmin, ymin, xmax, ymax])
    return windows

def pyramid(img, scale=0.8, min_size=(30, 30)):
  acc_scale = 1.0
  pyramid_imgs = [(img, acc_scale)]

  i = 0
  while True:
      acc_scale = acc_scale * scale
      h = int(img.shape[0] * acc_scale)
      w = int(img.shape[1] * acc_scale)
      if h < min_size[1] or w < min_size[0]:
        break
      img = cv2.resize(img, (w, h))
      pyramid_imgs.append((img, acc_scale * (scale ** i)))
      i += 1

  return pyramid_imgs

def visualize_bbox(img, bboxes, label_encoder):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  for box in bboxes:
    xmin, ymin, xmax, ymax, predict_id, conf_score = box
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    classname = label_encoder.inverse_transform([predict_id])[0]
    label = f"{classname} {conf_score:.2f}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)

    cv2.rectangle(img, (xmin, ymin - 20), (xmin + w, ymin), (0, 255, 0), -1)

    cv2.putText(img, label, (xmin, ymin -5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 1)
  img = cv2.imencode('.png', img)[1].tobytes()
  return img, label
def compute_iou(bbox, bboxes, bbox_area, bboxes_area):
  xxmin = np.maximum(bbox[0], bboxes[:, 0])
  yymin = np.maximum(bbox[1], bboxes[:, 1])
  xxmax = np.minimum(bbox[2], bboxes[:, 2])
  yymax = np.minimum(bbox[3], bboxes[:, 3])

  w = np.maximum(0, xxmax - xxmin + 1)
  h = np.maximum(0, yymax - yymin + 1)

  intersection = w * h 
  iou = intersection/(bbox_area+bboxes_area-intersection)
  return iou

def nms(bboxes, iou_threshold):
  if not bboxes:
    return []
  scores = np.array([bbox[5] for bbox in bboxes])
  sorted_indices = np.argsort(scores)[::-1]

  xmin = np.array([bbox[0] for bbox in bboxes])
  ymin = np.array([bbox[1] for bbox in bboxes])
  xmax = np.array([bbox[2] for bbox in bboxes])
  ymax = np.array([bbox[3] for bbox in bboxes])

  areas = (xmax - xmin + 1)*(ymax-ymin+1)
  keep = []
  while sorted_indices.size > 0:
    i = sorted_indices[0]
    keep.append(i)

    iou = compute_iou(
        [xmin[i], ymin[i], xmax[i], ymax[i]],
        np.array(
            [
                xmin[sorted_indices[1:]],
                ymin[sorted_indices[1:]],
                xmax[sorted_indices[1:]],
                ymax[sorted_indices[1:]]]
        ).T,
        areas[i],
        areas[sorted_indices[1:]]
    ) 
    idx_to_keep = np.where(iou <= iou_threshold)[0]
    sorted_indices = sorted_indices[idx_to_keep+1]
  
  return [bboxes[i] for i in keep]

def run(img_ori):
    stride = 15
    window_sizes = [
        (32, 32),
        (64, 64),
        (128, 128)
    ]
    conf_threshold = 0.95
    iou_threshold = 0.1

    img_lst, label_lst = readData()
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(label_lst)

    img_features_lst = []
    for img in img_lst:
        hog_feature = preprocess_img(img)
        img_features_lst.append(hog_feature)

    img_features = np.array(img_features_lst)
    random_state = 20
    test_size = 0.3
    is_shuffle = True

    X_train, X_val, y_train, y_val = train_test_split(
        img_features, encoded_labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=is_shuffle
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    clf = SVC(kernel = 'rbf',
          random_state=random_state,
          probability=True,
          C=0.5
          )
    clf.fit(X_train, y_train)
    

    bboxes = []
    pyramid_imgs = pyramid(img_ori)
    for pyramid_img_info in pyramid_imgs:
        pyramid_img, scale_factor = pyramid_img_info
        window_lst = sliding_window(
            pyramid_img,
            window_sizes=window_sizes,
            stride=stride,
            scale_factor=scale_factor 
        )
        for window in window_lst:
            xmin, ymin, xmax, ymax = window
            object_img = pyramid_img[ymin:ymax, xmin:xmax]
            preprocessed_img = preprocess_img(object_img)
            normalized_img = scaler.transform([preprocessed_img])[0]
            decision = clf.predict_proba([normalized_img])[0]
            if np.all(decision < conf_threshold): #threshold the confidence score.
                continue
            else:
                predict_id = np.argmax(decision)
                conf_score = decision[predict_id]
                xmin = int(xmin / scale_factor)
                ymin = int(ymin / scale_factor)
                xmax = int(xmax / scale_factor)
                ymax = int(ymax / scale_factor)
                bboxes.append([xmin, ymin, xmax, ymax, predict_id, conf_score])
    bboxes = nms(bboxes, iou_threshold)
    processed_img, a = visualize_bbox(img_ori, bboxes, label_encoder)  # Replace 'bboxes' and 'label_encoder' with your data
    return processed_img, a