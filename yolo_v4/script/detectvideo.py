#!/usr/bin/env python3
import time
import tensorflow as tf
import sys
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
sys.path.insert(1, '/usr/local/lib/python3.6/dist-packages/cv2')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.config import cfg

from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
print("cv2 version: ", cv2.__version__)
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from cv_bridge import CvBridge, CvBridgeError
from yolo_v4_new.msg import ROI
from yolo_v4_new.msg import ROI_array
import rospy
from sensor_msgs.msg import Image as rosimage

Generic_Location = "/home/lucaubuntu18/Solomon_TransparentObject_ws/src/yolo_v4/script"

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', Generic_Location + '/checkpoints/yolo_0912',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', Generic_Location + '/data/road.mp4', 'path to input video')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.75, 'score threshold')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dis_cv2_window', False, 'disable cv2 window during the process') # this is good for the .ipynb

cv_image = np.zeros((0,0,3), np.uint8)
bridge = CvBridge()

class_name = ["soda", "coffee", "juice"]
def realsense_callback(rosimage):
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(rosimage, "bgr8")

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_id = 0
    while not rospy.is_shutdown():
        frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        # # return_value, frame = vid.read()
        # if return_value:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     image = Image.fromarray(frame)
        # else:
        #     if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
        #         print("Video processing complete")
        #         break
        #     raise ValueError("No image! Try with another video format")
        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
                
        image = utils.draw_bbox(frame, pred_bbox)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        info = "fps: %.2f" %(1./exec_time)

        print(info)

        image_h, image_w, _ = frame.shape

        out_boxes, out_scores, out_classes, num_boxes = pred_bbox

        ROI_array_msg = ROI_array()
        
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > NUM_CLASS: continue
            ROI_msg = ROI()
            coor = out_boxes[0][i]
            ROI_msg.min_x = int(coor[1]) #x min
            ROI_msg.min_y = int(coor[0]) #y min
            ROI_msg.Max_x = int(coor[3]) #x max 
            ROI_msg.Max_y = int(coor[2]) #y max
            ROI_msg.score = float(out_scores[0][i])
            ROI_msg.object_name = str(class_name[int(out_classes[0][i])])
            
            ROI_array_msg.ROI_list.append(ROI_msg)

        roi_array_pub.publish(ROI_array_msg)

        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not FLAGS.dis_cv2_window:
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        if FLAGS.output:
            out.write(result)

        frame_id += 1

if __name__ == '__main__':
    try:
        rospy.init_node('yolo_v4_new_detect_video', anonymous=True)
        rospy.Subscriber("/camera/color/image_raw", rosimage, realsense_callback)
        roi_array_pub = rospy.Publisher("/object/ROI_array", ROI_array, queue_size=100)
        rate = rospy.Rate(10) # 10hz

        app.run(main)
    except SystemExit:
        pass
