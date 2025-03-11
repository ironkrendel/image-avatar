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

print("Works!")

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

images_filenames = []
images_parameters = []

allowed_extensions = ['.json']
image_folder = './Images/Frames/'
if not os.path.exists(image_folder):
    print("Input data folder doesn't exist!")
    exit(1)
folder_contents = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and os.path.splitext(os.path.join(image_folder, f))[1] in allowed_extensions]

print("Loading data...")
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

X = np.array(images_parameters)[:, 0]
Y = np.array(images_parameters)[:, 1]

_fig, _ax = plt.subplots()
_ax.scatter(X, Y)
_fig.canvas.draw()
point_plot = np.array(_fig.canvas.renderer.buffer_rgba())
cv2.imshow("Points", point_plot)

print("Initializing video capture...")
input_reader = InputReader(
    capture=0, width=640, height=460, fps=30, raw_rgb=False, dcap=9, use_dshowcapture=True
)

ret, frame = input_reader.read()
print("Capture initialized")

height = frame.shape[0]
width = frame.shape[1]

print("Creating face tracker...")
tracker = Tracker(
    width, height, max_threads=multiprocessing.cpu_count(), model_type=3, max_faces=1
)

faces = tracker.predict(frame)
print("Face tracker created")

indexes_set = set()

min_vals = [100000000 for _ in range(10000)]
max_vals = [-100000000 for _ in range(10000)]

fig, ax = plt.subplots()

while True:
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
        params = [float(dist(f.pts_3d[50], f.pts_3d[55])) * 1, float(dist(f.pts_3d[62], f.pts_3d[58])) * 1]
        # params = [float(dist(f.pts_3d[50], f.pts_3d[55])), float(dist(f.pts_3d[62], f.pts_3d[58])), f.euler[0] * 10, f.euler[1] * 10, f.euler[2] * 10]
        # params = [dist(f.pts_3d[50], f.pts_3d[55]), dist(f.pts_3d[62], f.pts_3d[58])]
        # rot = f.quaternion

        # params = []
        # for p in f.pts_3d[48:].tolist():
        #     for j in p:
        #         params.append(j)

        blank = np.empty((500, 500, 3))

        # print(params)

        for i in range(len(params)):
            min_vals[i] = min(min_vals[i], params[i])
            max_vals[i] = max(max_vals[i], params[i])
            if min_vals[i] == max_vals[i]:
                min_vals[i] -= 0.00000001
            params[i] = (params[i] - min_vals[i]) / (max_vals[i] - min_vals[i])

        blank = cv2.circle(blank, (int(params[0] * 500), int(params[1] * 500)), 15, (0, 255, 0), -1)

        start_time = time.perf_counter()
        # errors = np.array([])
        errors = []
        for img_params in images_parameters:
            # errors = np.append(errors, dist(params, img_params))
            errors.append(dist(params, img_params))
        min_index = min(range(len(errors)), key=errors.__getitem__)
        # errors = np.array(errors)

        ax.clear()
        ax.plot(errors)
        fig.canvas.draw()
        error_graph = np.array(fig.canvas.renderer.buffer_rgba())
        cv2.imshow("Error", error_graph)

        # min_index = np.argmin(errors)
        indexes_set.add(min_index)
        print(len(indexes_set))
        blank = cv2.circle(blank, (int(images_parameters[min_index][0] * 500), int(images_parameters[min_index][1] * 500)), 15, (255, 0, 0), -1)
        # print(images_parameters[min_index])
        # print(dist(params, images_parameters[min_index]))
        print(f"{1000 * (time.perf_counter() - start_time):.2f}ms")
        img = cv2.imread(image_folder + images_filenames[min_index])
        cv2.imshow("Pos", blank)
        cv2.imshow("Teto", img)

    cv2.imshow("Cap", frame)
    key = cv2.waitKey(int(1000 / 200))
    if key == 113 or key == 27:
        break