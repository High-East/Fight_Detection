import os
import re
from collections import deque
import numpy as np
import argparse
import cv2
from tqdm import tqdm

#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
                       

def exactly_sorting(array):
    '''
    sort exactly for list or array which element is string
    example: [1, 10, 2, 4] -> [1, 2, 4, 10]
    '''
    str2int = lambda string: int(string) if string.isdigit() else string
    key = lambda key: [str2int(x) for x in re.split("([0-9]+)", key)]
    return sorted(array, key=key)


def sorted_list(path):
    return exactly_sorting(os.listdir(path))

def full_path_sorted_list(path):
    full_path = sorted_list(path)
    for i in range(len(full_path)):
        full_path[i] = path + full_path[i]
    return full_path




def parse_args():
    parse = argparse.ArgumentParser(description="test")
    parse.add_argument("-l", "--label", required=False, default="fight", help="labeling fight or non")
    parse.add_argument("-m", "--model", required=False, default="/Users/KDH/Boaz/Project/Fight_detection/Github/classification/model/violence_resnet.model",
        help="path to trained serialized model")
    parse.add_argument("-i", "--input", required=False, default='/Users/KDH/Boaz/Project/Fight_detection/DataSet/NTU_CCTV/NTU_Fight/',
        help="path to our input videos directory")
    parse.add_argument("-o", "--output", required=False, default="/Users/KDH/Boaz/Project/Fight_detection/Github/classification/", help="path for saving accuracy.npy")
    parse.add_argument("-s", "--size", type=int, default=60,
        help="size of queue for averaging")
    args = parse.parse_args()
    return args



def test(video_path, model, label, size):
    Q = deque(maxlen=size)
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    vid = cv2.VideoCapture(video_path)
    (W, H) = (None, None)
    results = []
    count = 0
    print("[{}] start".format(os.path.basename(video_path)))
    while True:
        # read the next frame from the file
        (grabbed, frame) = vid.read()
        count += 1
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        # clone the output frame, then convert it from BGR to RGB
        # ordering, resize the frame to a fixed 224x224, and then
        # perform mean subtraction
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype("float32")
        frame -= mean
        # make predictions on the frame and then update the predictions
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        # perform prediction averaging over the current history of
        # previous predictions
        if label == "fight":
            result = np.array(Q).mean(axis=0)[1]
            if result >= 0.7:
                result = 1
            else:
                result = 0
        elif label == "non":
            #result = np.array(Q).mean(axis=0)[0]
            print("non-fight not yet make...")
        else:
            print("Unrecognizable label")
        results.append(result)  # 0 = non-fight, 1 = fight
    accuracy = round(np.array(results).mean(), 2)
    print("accuracy: ", accuracy)
    return accuracy


        # if count % 10 == 0:
        #     print("[accuracy = {}] {} frame is completed".format(np.array(results).mean(), count))
   # if label == "fight":
    #    accuracy = round(np.array(results).mean(), 2)
     #   print("accuracy: ", accuracy)
      #  return accuracy
   # elif label == "non":
    #    accuracy = round(1 - np.array(results).mean(), 2)
     #   print("accuracy: ", accuracy)
      #  return accuracy
    #else:
     #   print("Unrecognizable label")



def main():
    args = parse_args()
    model = args.model
    label = args.label
    size = args.size
    vid_dir = args.input
    output = args.output
    videos_path = full_path_sorted_list(vid_dir)[1:]
    print("[Info] Model loading...")
    model = load_model(args.model)
    result_list = []
    outlier = []
    for i in tqdm(range(len(videos_path))):
        acc = test(videos_path[i], model, label, size)
        result_list.append(acc)
        total_accuracy = round(np.array(result_list).mean(), 2)
        print("Total accuracy:", total_accuracy)
        if acc <= 0.75:
            outlier.append((os.path.basename(videos_path[i]), acc))
    np.save(output + "acc.npy", result_list)
    np.save(output + "outlier.npy", outlier)

if __name__ == '__main__':
    main()














