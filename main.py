import numpy as np
import copy
import time
import cv2
import json
import os
import multiprocessing
import matplotlib.pyplot as plt
from input_reader import InputReader, VideoReader, DShowCaptureReader, try_int
from tracker import Tracker, get_model_base_path
import pymongo
from sklearn.neighbors import KDTree

print("Works!")

bias = [0.0 for _ in range(10000)]
bias[0] = -0.07
bias[1] = -0.05

weights = [1 for _ in range(10000)]
weights[2] = 0
weights[3] = 0
weights[4] = 0

# face oval
# 0 - 16

# left brow
# 17 - 21

# right brow
# 22 - 26

# sinus
# 27 - 30

# nostrils
# 31 - 35

# right eye
# 36 - 41

# left eye
# 42 - 47

# top lip
# 50 is center
# 48 - 52

# bottom lip
# 55 is center
# 53 - 57

# right mouth edge
# 58

# top teeth
# 59 - 61

# left mouth edge
# 62

# bottom teeth
# 63 - 65

min_vals = [100000000.0 for _ in range(10000)]
max_vals = [-100000000.0 for _ in range(10000)]

def dist(p1: list, p2: list) -> float:
    if len(p1) != len(p2):
        raise Exception("Dimensions of point 1 does not match dimensions of point 2!")
    result = 0
    for i in range(len(p1)):
        result += (p2[i] - p1[i])**2
    return result

def MSE(X: list, Y: list) -> float:
    mse = 0
    if len(X) != len(Y):
        raise Exception("Non matching array length")
    for i in range(len(X)):
        mse += (X[i] - Y[i])**2
    return mse / len(X)

def normalize_array(arr: list):
    print(arr)
    new_arr = []
    for obj in arr:
        new_arr.append((arr[i] - min_vals[i]) / (max_vals[i] - min_vals[i]))
    return new_arr

vec_normalize_array = np.vectorize(normalize_array)

images_filenames = []
images_parameters = []

allowed_extensions = ['.json']
image_folder = './Images/Frames/'
if not os.path.exists(image_folder):
    print("Input data folder doesn't exist!")
    exit(1)
folder_contents = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and os.path.splitext(os.path.join(image_folder, f))[1] in allowed_extensions]

print("Loading data...")
dbclient = pymongo.MongoClient(f"mongodb://localhost:27017/", serverSelectionTimeoutMS = 2000)
dbflag = True
dbfileflag = True
try:
    dbclient.server_info()
except:
    dbflag = False
    print("Can't access MongoDB")

if not os.path.exists("./Images/image_dataset.data.json"):
    dbfileflag = False
    print("No exported MongoDB json file found")

if dbflag:
    dbfileparsed = os.path.basename("./Images/image_dataset.data.json").split('.')
    database = dbclient[dbfileparsed[0]]
    dbcollection = database[dbfileparsed[1]]

    dbdata = dbcollection.find({}, {"crop": 1, "parameters":1})

    for obj in dbdata:
        images_filenames.append(obj['crop'])
        images_parameters.append(obj['parameters'])
elif dbfileflag:
    with open("./Images/image_dataset.data.json", 'r') as fin:
        data = json.loads(fin.read())
        for obj in data:
            images_filenames.append(obj['crop'])
            images_parameters.append(obj['parameters'])
else:
    for i, f in enumerate(folder_contents):
        if i % 100 == 0:
            print(f"\r{i/len(folder_contents) * 100:.2f}%", end="")
        with open(image_folder + f, 'r') as cin:
            img_data = json.loads(cin.read())
        # print(img_data)
        images_filenames.append(img_data['crop'])
        images_parameters.append(img_data['parameters'])
print(f"\rLoaded {len(images_parameters)} records")

if len(images_parameters) < 1:
    print("No data to work with. Exiting")
    exit(0)

print("Initializing video capture...")
input_reader = InputReader(
    capture=0, width=640, height=460, fps=60, raw_rgb=False, dcap=9, use_dshowcapture=True
)

ret, frame = input_reader.read()
print("Capture initialized")

height = frame.shape[0]
width = frame.shape[1]

print("Creating face tracker...")
tracker = Tracker(
    width, height, max_threads=multiprocessing.cpu_count(), model_type=3, max_faces=1, silent=True
)

faces = tracker.predict(frame)
print("Face tracker created")

indexes_set = set()

for par in images_parameters:
    for i, val in enumerate(par):
        min_vals[i] = min(min_vals[i], val)
        max_vals[i] = max(max_vals[i], val)

if __debug__:
    fig, ax = plt.subplots()
    _fig, _ax = plt.subplots()

data_sampling_start_time = time.perf_counter()
while time.perf_counter() - data_sampling_start_time <= 2:
    ret, frame = input_reader.read()

    faces = tracker.predict(frame)

    for face_num, f in enumerate(faces):
        f = copy.copy(f)

        frame = cv2.putText(
            frame,
            str(f.id),
            (int(f.bbox[0]), int(f.bbox[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 255),
        )
        frame = cv2.putText(
            frame,
            f"{f.conf:.4f}",
            (int(f.bbox[0] + 18), int(f.bbox[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
        )

        box_y = int(f.bbox[0])
        box_x = int(f.bbox[1])
        box_y_size = int(f.bbox[2])
        box_x_size = int(f.bbox[3])
        frame = cv2.rectangle(frame, (box_y, box_x), (box_y + box_y_size, box_x + box_x_size), (255, 0, 0), 1)

        for pt_num, (x, y, c) in enumerate(f.lms):
            x = int(x + 0.5)
            y = int(y + 0.5)
            color = (0, 255, 0)
            if pt_num >= 66:
                color = (255, 255, 0)
            if pt_num == 33:
                color = (255, 0, 255)
                pass
            if not (x < 0 or y < 0 or x >= height or y >= width):
                cv2.circle(frame, (y, x), 1, color, -1)

        params = [float(dist(f.pts_3d[50], f.pts_3d[55])) * 1, float(dist(f.pts_3d[62], f.pts_3d[58])) * 1]

        for i in range(2):
            if params[i] < min_vals[i]:
                min_vals[i] = params[i]
                data_sampling_start_time = time.perf_counter()
            if params[i] > max_vals[i]:
                max_vals[i] = params[i]
                data_sampling_start_time = time.perf_counter()
            if min_vals[i] == max_vals[i]:
                min_vals[i] -= 0.00000001

    cv2.imshow("Cap", frame)
    print(f"{time.perf_counter() - data_sampling_start_time:.2f}s")
    key = cv2.waitKey(int(1000 / 200))
    if key == 113 or key == 27:
        exit(0)

normalized_img_params = []
for img_params in images_parameters:
    _img_params = []
    for i, p in enumerate(img_params):
        if i >= 2:
            _img_params.append(p)
        else:
            _img_params.append((p - min_vals[i]) / (max_vals[i] - min_vals[i]))
        _img_params[i] *= weights[i]
    normalized_img_params.append(_img_params)

if __debug__:
    X = np.array(normalized_img_params)[:, 0]
    Y = np.array(normalized_img_params)[:, 1]

    _ax.clear()
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)
    _ax.scatter(X, Y)
    _fig.canvas.draw()
    point_plot = np.array(_fig.canvas.renderer.buffer_rgba())
    cv2.imshow("Points", point_plot)

tree = KDTree(normalized_img_params)

while True:
    total_time = time.perf_counter()
    ret, frame = input_reader.read()

    faces = tracker.predict(frame)

    for face_num, f in enumerate(faces):
        f = copy.copy(f)
        # print(f.pts_3d)
        # print(f.normalize_pts3d(f.pts_3d))
        frame = cv2.putText(
            frame,
            str(f.id),
            (int(f.bbox[0]), int(f.bbox[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 255),
        )
        frame = cv2.putText(
            frame,
            f"{f.conf:.4f}",
            (int(f.bbox[0] + 18), int(f.bbox[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
        )

        box_y = int(f.bbox[0])
        box_x = int(f.bbox[1])
        box_y_size = int(f.bbox[2])
        box_x_size = int(f.bbox[3])
        frame = cv2.rectangle(frame, (box_y, box_x), (box_y + box_y_size, box_x + box_x_size), (255, 0, 0), 1)

        for pt_num, (x, y, c) in enumerate(f.lms):
            # if pt_num == 66 and (f.eye_blink[0] < 0.30 or c < 0.20):
                # continue
            # if pt_num == 67 and (f.eye_blink[1] < 0.30 or c < 0.20):
                # continue
            x = int(x + 0.5)
            y = int(y + 0.5)
            color = (0, 255, 0)
            if pt_num >= 66:
                color = (255, 255, 0)
            if pt_num == 33:
                color = (255, 0, 255)
                pass
            if not (x < 0 or y < 0 or x >= height or y >= width):
                cv2.circle(frame, (y, x), 1, color, -1)

        # params = [dist(f.pts_3d[50], f.pts_3d[55]), dist(f.pts_3d[62], f.pts_3d[58]), *f.quaternion]
        # params = [float(dist(f.pts_3d[50], f.pts_3d[55])) * 1, float(dist(f.pts_3d[62], f.pts_3d[58])) * 1, *f.euler]
        # params = [float(dist(f.pts_3d[50], f.pts_3d[55])) * 1, float(dist(f.pts_3d[62], f.pts_3d[58])) * 1]
        params = [float(dist(f.pts_3d[50], f.pts_3d[55])), float(dist(f.pts_3d[62], f.pts_3d[58])), f.euler[0], f.euler[1], f.euler[2]]
        for i in range(len(params)):
            params[i] *= weights[i]
        # params = [dist(f.pts_3d[50], f.pts_3d[55]), dist(f.pts_3d[62], f.pts_3d[58])]
        # rot = f.quaternion

        # params = []
        # for p in f.pts_3d[48:].tolist():
        #     for j in p:
        #         params.append(j)

        if __debug__:
            blank = np.empty((500, 500, 3))

        # print(params)

        # for i in range(len(params)):
        for i in range(2):
            # min_vals[i] = min(min_vals[i], params[i])
            # max_vals[i] = max(max_vals[i], params[i])
            # if min_vals[i] == max_vals[i]:
            #     min_vals[i] -= 0.00000001
            params[i] = (params[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
            params[i] += bias[i]

        if __debug__:
            blank = cv2.circle(blank, (int(params[0] * 500), 500 - int(params[1] * 500)), 15, (0, 255, 0), -1)

        start_time = time.perf_counter()
        # errors = np.array([])
        # errors = []
        # normalized_img_params = []
        # for img_params in normalized_img_params:
            # errors = np.append(errors, dist(params, img_params))
            # _img_params = []
            # for i, p in enumerate(img_params):
                # _img_params.append((p - min_vals[i]) / (max_vals[i] - min_vals[i]))
            # errors.append(MSE(params, img_params))
            # normalized_img_params.append(_img_params)
        # min_index = min(range(len(errors)), key=errors.__getitem__)
        # errors = np.array(errors)
        # min_index = np.argmin(errors)

        min_dist, min_ind = tree.query([params])
        min_index = int(min_ind[0][0])

        # print(min_dist, min_ind, min_index)

        # min_index = 0

        # if __debug__:
            # ax.clear()
            # X left - 0.125 * width
            # X right -  0.9 * width
            # ax.plot(errors)
            # fig.canvas.draw()
            # error_graph = np.array(fig.canvas.renderer.buffer_rgba())
            # min_index_pos_x = (min_index / len(errors) * 0.775 * error_graph.shape[1]) + error_graph.shape[1] * 0.125
            # error_graph = cv2.circle(error_graph, (int(min_index_pos_x), int(error_graph.shape[0] * 0.1)), 5, (0, 0, 255), -1)
            # cv2.imshow("Error", error_graph)

            # X = np.array(normalized_img_params)[:, 0]
            # Y = np.array(normalized_img_params)[:, 1]

            # _ax.clear()
            # # _ax.set_xticks([0, 1])
            # # _ax.set_yticks([0, 1])
            # _ax.set_xlim(0, 1)
            # _ax.set_ylim(0, 1)
            # _ax.scatter(X, Y)
            # _fig.canvas.draw()
            # point_plot = np.array(_fig.canvas.renderer.buffer_rgba())
            # cv2.imshow("Points", point_plot)

        if __debug__:
            indexes_set.add(min_index)
            print(f"Total images used: {len(indexes_set)}")

        if __debug__:
            blank = cv2.circle(blank, (int(normalized_img_params[min_index][0] * 500), 500 - int(normalized_img_params[min_index][1] * 500)), 15, (255, 0, 0), -1)
            # blank = cv2.circle(blank, (int(min_index / len(errors) * 500), 50), 5, (0, 0, 255), -1)
        
        # print(images_parameters[min_index])
        # print(dist(params, images_parameters[min_index]))
        print(f"{1000 * (time.perf_counter() - start_time):.2f}ms")
        img = cv2.imread(image_folder + images_filenames[min_index])
        if __debug__:
            cv2.imshow("Pos", blank)
        cv2.imshow("Teto", img)

    cv2.imshow("Cap", frame)
    print(f"FPS: {1 / (time.perf_counter() - total_time):.2f}")
    # key = cv2.waitKey(int(1000 / 200))
    key = cv2.waitKey(1)
    if key == 113 or key == 27:
        break