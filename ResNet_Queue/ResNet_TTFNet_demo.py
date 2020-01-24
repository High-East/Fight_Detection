import os
import pickle
import argparse
import time
import cv2
import numpy as np
from collections import deque
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv


def get_session(gpu_fraction=0.8):
    '''
    GPU memory handling
    '''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.compat.v1.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Fight Detection demo')
    parser.add_argument("--cnn_model", required=False, default='./model/violence_resnet.model',
        help="path to cnn model")
    parser.add_argument("--label-bin", required=False, default='./model/lb.pickle',
        help="path to label")
    parser.add_argument("-i", "--input", required=False, default='./video/demo4.mp4',
        help="path to input video")
    parser.add_argument("-s", "--size", type=int, default=60,
        help="size of queue")
    args = parser.parse_args()
    return args


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    KTF.set_session(get_session())
    args = parse_args()
    # CNN load
    print("[Load] CNN model")
    model = load_model(args.cnn_model)
    lb = pickle.loads(open(args.label_bin, "rb").read())
    # TTFNet load
    print("[Load] TTFNet")
    config_file = '/home/forbboaz/boaz-adv-project/ttfnet/configs/ttfnet/ttfnet_d53_2x.py'
    checkpoint_file = '/home/forbboaz/boaz-adv-project/ttfnet/checkpoints/epoch_24.pth'
    model_detect = init_detector(config_file, checkpoint_file, device='cuda:0')
    # Input setting
    vpath = args.input
    if args.input == 'camera':
        vpath = 0
    video = cv2.VideoCapture(vpath)
    writer = None
    (W, H) = (None, None)
    # Image preprocessing
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")  # mean value of ImageNet
    Q = deque(maxlen=args.size)
    # Read frames from video
    while True:
        fps_time = time.time()
        (success, frame) = video.read()
        # release the file pointers
        if not success:
            print("[Finished] Release file pointers")
            writer.release()
            video.release()
            break
        # setting image shape
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype("float32")
        frame -= mean
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        i = 1  # (non-fight, fight)
        label = lb.classes_[i]
        prob = results[i]*100
        text_color = (0, 255, 0)  # default: green
        if prob > 70:  # fight
            text_color = (0, 0, 255)  # red
            # Run TTFNet
            result_detect = inference_detector(model_detect, output)
            (person_bboxes, object_bboxes, image) = show_result(output, result_detect, model_detect.CLASSES, score_thr=0.5, wait_time=2)
            output = image
        else:
            label = 'Normal'
        text = "State : {:8} ({:3.2f}%)".format(label, prob)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, text, (35, 30), font, 0.75, text_color, 3)
        output = cv2.rectangle(output, (35, 50), (300, 60), text_color, -1)
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(os.path.splitext(vpath)[0] + "_output.mp4", fourcc, 30, (W, H), True)
        cv2.putText(output, "FPS: %f" % (1.0 / (time.time() - fps_time)), (35, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        writer.write(output)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


if __name__ == '__main__':
    main()



