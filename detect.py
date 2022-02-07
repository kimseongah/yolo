import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import csv
from flask import Flask, request, make_response
from werkzeug.utils import secure_filename

WEIGHTS_PATH = './checkpoints/yolov4-416'
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within images')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

app = Flask("yolo-server")


def to_images_data(images):
    input_size = 416
    #original_image=cv2.imread(images)
    original_image = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    
    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)
    return images_data, original_image

def print_result(wr, counted_classes):
    for key, value in counted_classes.items():
            print("Number of {}s: {}".format(key, value))
            wr.writerow([key, value])

def show_result(images_data, original_image):
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.5
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
    
    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
    
    # custom allowed classes (uncomment line below to allow detections for only people)
    #allowed_classes = ['person']

    # if count flag is enabled, perform counting of objects
    # count objects found
    counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
    # loop through dict and print
    image = utils.draw_bbox(original_image, pred_bbox, False, counted_classes, allowed_classes=allowed_classes, read_plate = False)

    image = Image.fromarray(image.astype(np.uint8))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite('./detections/' + 'detection' + '.png', image)
    return counted_classes

# def main(_argv):
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = InteractiveSession(config=config)
#     #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    
#     images = FLAGS.images
    
#     f = open('result.csv','w',newline='')
#     wr = csv.writer(f)
#     # load model
#     interpreter = tf.lite.Interpreter(model_path='./checkpoints/yolov4-416.tflite')
#     # loop through images in list and run Yolov4 model on each
    
#     images_data, original_image = to_images_data(images)
#     count_classes = show_result(images_data, original_image, interpreter)
#     print_result(wr, count_classes)


@app.route('/')
def health_check():
    return make_response('healthy', 200)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
saved_model_loaded = tf.saved_model.load(WEIGHTS_PATH, tags=[tag_constants.SERVING])

@app.route('/process', methods=['GET'])
def process_image():
    image = request.files.get('image')
    if image is None:
        return make_response("there is no given image file", 400)
    #images =Image.open(image)
    image.save('./files'+secure_filename(image.filename))
    images = cv2.imread('./files'+secure_filename(image.filename))
    #print(images)
    result = {}
    if request.args.get('yolo') is not None:
        images_data, original_image = to_images_data(images)
        result = show_result(images_data, original_image)
    return make_response(result, 200)

if __name__=='__main__':
    app.run(debug=True)       

