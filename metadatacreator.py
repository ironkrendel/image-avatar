import cv2
import os
import argparse
import multiprocessing
import json
import time
from tracker import Tracker, get_model_base_path

def dist(p1: list[3], p2: list[3]) -> float:
    sqr_x = (p2[0] - p1[0]) ** 2
    sqr_y = (p2[1] - p1[1]) ** 2
    sqr_z = (p2[2] - p1[2]) ** 2
    return sqr_x + sqr_y + sqr_z

def MetadataCreatorThread(path, filenames, model):
    tracker = Tracker(
        1920, 1080, max_threads=1, model_type=model, max_faces=1, try_hard=True
    )
    for f in filenames:
        start_time = time.perf_counter()
        img = cv2.imread(path + f)
        height, width = img.shape[:2]
        # tracker = Tracker(
        #     width, height, max_threads=1, model_type=model, max_faces=1, try_hard=True, silent=True
        # )
        tracker.width = width
        tracker.height = height
        tracker.faces = []
        tracker.additional_faces = []
        tracker.detected = 0
        faces = tracker.predict(img)

        if len(faces) < 1:
            print('-' * 100)
            print(f)
            print('-' * 100)
            continue

        face = faces[0]

        # center_x = face.bbox[1] + face.bbox[3] / 2
        # center_y = face.bbox[0] + face.bbox[2] / 2
        # side = face.bbox[2]
        # top_left = (int(center_x - side / 2), int(center_y - side / 2))
        # top_right = (int(center_x + side / 2), int(center_y - side / 2))
        # bottom_left = (int(center_x - side / 2), int(center_y + side / 2))
        # bottom_right = (int(center_x + side / 2), int(center_y + side / 2))
        # crop_img = img[int(face.bbox[0]):int(face.bbox[0] + face.bbox[2]), int(center_x - side / 2):int(center_x + side / 2)]
        # cv2.imwrite(path + os.path.splitext(f)[0] + '_crop' + os.path.splitext(f)[1], crop_img)

        center = (face.lms[33][1], face.lms[33][0])
        side = face.bbox[2]
        crop_start_x = int(center[0] - side / 2)
        crop_start_y = int(center[1] - side / 2)
        # crop_img = cv2.rectangle(img, (crop_start_x, crop_start_y), (crop_start_x + int(side), crop_start_y + int(side)), (255, 0, 0), 1)

        crop_img = img[crop_start_y:crop_start_y + int(side), crop_start_x:crop_start_x + int(side)]
        crop_img = cv2.resize(crop_img, (500, 500), interpolation=cv2.INTER_LINEAR)


        # box_y = int(face.bbox[0])
        # box_x = int(face.bbox[1])
        # box_y_size = int(face.bbox[2])
        # box_x_size = int(face.bbox[3])
        # crop_img = cv2.rectangle(img, (box_y, box_x), (box_y + box_y_size, box_x + box_x_size), (255, 0, 0), 1)
        # crop_img = cv2.circle(crop_img, center, 5, (0, 255, 0), 2)
        # if os.path.exists(path + os.path.splitext(f)[0] + '_crop' + os.path.splitext(f)[1]):
        #     os.remove(path + os.path.splitext(f)[0] + '_crop' + os.path.splitext(f)[1])

        result = {}
        result['filename'] = f
        # result['lms'] = face.lms.tolist()
        # result['pts_3d'] = face.pts_3d.tolist()
        result['crop'] = os.path.splitext(f)[0] + '_crop' + os.path.splitext(f)[1]
        # result['parameters'] = [float(dist(face.pts_3d[50], face.pts_3d[55])), float(dist(face.pts_3d[62], face.pts_3d[58])), *(face.quaternion.tolist())]
        result['parameters'] = [float(dist(face.pts_3d[50], face.pts_3d[55])), float(dist(face.pts_3d[62], face.pts_3d[58])), *face.euler]
        # result['parameters'] = [*face.euler]
        # result['parameters'] = [float(dist(face.pts_3d[50], face.pts_3d[55])), float(dist(face.pts_3d[62], face.pts_3d[58])), face.euler[0] * 10, face.euler[1] * 10, face.euler[2] * 10]
        result_json = json.dumps(result)

        with open(path + os.path.splitext(f)[0] + '.json', 'w') as fout:
            fout.write(result_json)
        cv2.imwrite(path + os.path.splitext(f)[0] + '_crop' + os.path.splitext(f)[1], crop_img)
        print(f"{1000 * (time.perf_counter() - start_time):.2f}ms {f}")

def deleteOldMetadata(path):
    allowed_extensions = ['.json']
    folder_contents = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and os.path.splitext(os.path.join(path, f))[1] in allowed_extensions]
    for f in folder_contents:
        with open(path + f, 'r') as cin:
            img_data = json.loads(cin.read())
        if os.path.exists(path + img_data['crop']):
            os.remove(path + img_data['crop'])
        os.remove(path + f)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", help="Folder with input data to create metadata for", default="./Images/Frames/")
    parser.add_argument("-T", "--max-threads", help="Set the maximum number of threads", default=4, type=int)
    parser.add_argument("-m", "--model", help="Select the model used for face processing", default=3, type=int)
    parser.add_argument("-D", "--delete", help="Delete all metadata and associated files", action="store_true")

    args=parser.parse_args()

    if args.delete:
        deleteOldMetadata(args.input)
        exit(0)

    allowed_extensions = ['.json']
    folder_contents = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f)) and os.path.splitext(os.path.join(args.input, f))[1] in allowed_extensions]
    ignore_files = []

    for f in folder_contents:
        with open(args.input + f, 'r') as cin:
            img_data = json.loads(cin.read())
        if os.path.exists(args.input + img_data['crop']):
            ignore_files.append(img_data['crop'])

    allowed_extensions = ['.png', '.jpg', '.jpeg']
    folder_contents = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f)) and os.path.splitext(os.path.join(args.input, f))[1] in allowed_extensions and f not in ignore_files]

    if len(folder_contents) <= 0:
        print("Input folder is empty!")
        exit(0)

    if len(folder_contents) < args.max_threads:
        files = [[_] for _ in folder_contents]
    else:
        start_index = 0
        slice_len = int(len(folder_contents) / args.max_threads)
        files = []
        for i in range(args.max_threads):
            files.append(folder_contents[start_index:start_index + slice_len])
            start_index += slice_len
        if start_index < len(folder_contents):
            for elem in folder_contents[start_index:]:
                files[0].append(elem)

    # files = [[_] for _ in folder_contents]

    input_data = [(args.input, files[_], args.model) for _ in range(len(files))]

    start_time = time.perf_counter()
    process_pool = multiprocessing.Pool(min(args.max_threads, len(input_data)))
    process_pool.starmap(MetadataCreatorThread, input_data)
    print(f"Total duration: {time.perf_counter() - start_time:.2f}s")

if __name__ == "__main__":
    multiprocessing.freeze_support()

    main()