import os
import argparse
import multiprocessing
import json
import time
import pymongo

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def dist(p1: list[3], p2: list[3]) -> float:
    sqr_x = (p2[0] - p1[0]) ** 2
    sqr_y = (p2[1] - p1[1]) ** 2
    sqr_z = (p2[2] - p1[2]) ** 2
    return sqr_x + sqr_y + sqr_z

def MetadataCreatorThread(path, filenames, model, quick, dbaddr, dbport, dbfile):
    if len(filenames) < 1:
        return

    import cv2
    from tracker import Tracker, get_model_base_path

    dbclient = pymongo.MongoClient(f"mongodb://{dbaddr}:{dbport}/")
    dbflag = True
    try:
        dbclient.server_info()
    except:
        dbflag = False
        print("Can't access MongoDB")

    if dbflag:
        dbfileparsed = os.path.basename(dbfile).split('.')
        if len(dbfileparsed) == 3 and dbfileparsed[-1] == 'json':
            database = dbclient[dbfileparsed[0]]
            dbcollection = database[dbfileparsed[1]]
        else:
            dbflag = False

    sample_img = cv2.imread(path + filenames[0])

    tracker = Tracker(
        # sample_img.shape[1], sample_img.shape[0], max_threads=1, model_type=model, max_faces=1, try_hard=(not quick), silent=True, threshold=0.85, detection_threshold=0.85
        sample_img.shape[1], sample_img.shape[0], max_threads=1, model_type=model, max_faces=1, try_hard=(not quick), silent=True, threshold=0.8, detection_threshold=0.7
    )
    total_len = len(filenames)
    for i, f in enumerate(filenames):
        try:
            if not os.path.exists(path + f):
                continue
            start_time = time.perf_counter()
            img = cv2.imread(path + f)
            if img.shape[0] == 0 or img.shape[1] == 0:
                continue
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
                # print('-' * 100)
                # print(f)
                # print('-' * 100)
                print('\033[91m' + f"{1000 * (time.perf_counter() - start_time):.2f}ms {i / total_len * 100:.2f}% {f}" + '\033[0m')
                continue

            face = faces[0]

            if face.euler[0] == 0 or face.euler[1] == 0 or face.euler[2] == 0 or float(dist(face.pts_3d[50], face.pts_3d[55])) >= 0.35 or float(dist(face.pts_3d[62], face.pts_3d[58])) >= 0.35:
                print('\033[91m' + f"{1000 * (time.perf_counter() - start_time):.2f}ms {i / total_len * 100:.2f}% {f}" + '\033[0m')
                continue

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

            # for pt_num, (x, y, c) in enumerate(face.lms):
            #     x = int(x + 0.5)
            #     y = int(y + 0.5)
            #     color = (0, 255, 0)
            #     if pt_num >= 66:
            #         color = (255, 255, 0)
            #     if pt_num == 33:
            #         color = (255, 0, 255)
            #         pass
            #     if not (x < 0 or y < 0 or x >= height or y >= width):
            #         cv2.circle(img, (y, x), 1, color, -1)

            # img = cv2.putText(
            #     img,
            #     f"{face.conf:.4f}",
            #     (int(face.bbox[0] + 18), int(face.bbox[1] - 6)),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 0, 255),
            # )

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
            result['parameters'] = [float(dist(face.pts_3d[50], face.pts_3d[55])) * 1, float(dist(face.pts_3d[62], face.pts_3d[58])) * 1, face.euler[0] * 1, face.euler[1] * 1, face.euler[2] * 1]
            # result['parameters'] = [float(dist(face.pts_3d[50], face.pts_3d[55])) * 1, float(dist(face.pts_3d[62], face.pts_3d[58])) * 1]
            # result['parameters'] = []
            # for p in face.pts_3d[48:].tolist():
            #     for j in p:
            #         result['parameters'].append(j)
            # result['parameters'] = [*face.euler]
            # result['parameters'] = [float(dist(face.pts_3d[50], face.pts_3d[55])), float(dist(face.pts_3d[62], face.pts_3d[58])), face.euler[0] * 10, face.euler[1] * 10, face.euler[2] * 10]
            result_json = json.dumps(result)
            with open(path + os.path.splitext(f)[0] + '.json', 'w') as fout:
                fout.write(result_json)
            cv2.imwrite(path + os.path.splitext(f)[0] + '_crop' + os.path.splitext(f)[1], crop_img)
            if dbflag:
                dbcollection.insert_one(result)
            print('\033[92m' + f"{1000 * (time.perf_counter() - start_time):.2f}ms {i / total_len * 100:.2f}% {f}" + '\033[0m')
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            print(f)
            break

def ExportDBToJson(dbaddr, dbport, file):
    from bson.json_util import dumps

    dbclient = pymongo.MongoClient(f"mongodb://{dbaddr}:{dbport}/")
    try:
        dbclient.server_info()
    except:
        print("Can't access MongoDB")
        return

    dbfileparsed = os.path.basename(file).split('.')
    if len(dbfileparsed) == 3 and dbfileparsed[-1] == 'json':
        database = dbclient[dbfileparsed[0]]
        dbcollection = database[dbfileparsed[1]]

    data = dbcollection.find().to_list()
    for obj in data:
        del obj['_id']
    with open(file, 'w') as fout:
        json.dump(json.loads(dumps(data)), fout)

def MetadataNormalizer(path):
    allowed_extensions = ['.json']
    folder_contents = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and os.path.splitext(os.path.join(path, f))[1] in allowed_extensions]
    if len(folder_contents) < 1:
        return
    with open(path + folder_contents[0], 'r') as cin:
        tmp_data = json.loads(cin.read())
    min_vals = [100000000000 for _ in range(len(tmp_data['parameters']))]
    max_vals = [-100000000000 for _ in range(len(tmp_data['parameters']))]
    print("Analyzing data range")
    for i, f in enumerate(folder_contents):
        if i % 100 == 0:
            print(f"\r{i / len(folder_contents) * 100:.2f}%", end="")
        try:
            with open(path + f, 'r') as cin:
                img_data = json.loads(cin.read())
            for i in range(len(img_data['parameters'])):
                min_vals[i] = min(min_vals[i], img_data['parameters'][i])
                max_vals[i] = max(max_vals[i], img_data['parameters'][i])
        except Exception as e:
            pass
    print()
    print("Remaping values")
    for i, f in enumerate(folder_contents):
        if i % 100 == 0:
            print(f"\r{i / len(folder_contents) * 100:.2f}%", end="")
        try:
            with open(path + f, 'r') as cin:
                img_data = json.loads(cin.read())
            new_params = []
            for i in range(len(img_data['parameters'])):
                new_params.append(((img_data['parameters'][i] - min_vals[i])) / (max_vals[i] - min_vals[i]))
            img_data['parameters'] = new_params
            result_json = json.dumps(img_data)
            with open(path + f, 'w') as fout:
                fout.write(result_json)
        except Exception as e:
            pass
    print()

def deleteOldMetadata(path):
    print("Searching metadata files")
    allowed_extensions = ['.json']
    folder_contents = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and os.path.splitext(os.path.join(path, f))[1] in allowed_extensions]
    print(f"Found {len(folder_contents)} files")
    for i, f in enumerate(folder_contents):
        if i % 100 == 0:
            print(f"\r{i / len(folder_contents) * 100:.2f}%", end="")
        try:
            with open(path + f, 'r') as cin:
                img_data = json.loads(cin.read())
            if os.path.exists(path + img_data['crop']):
                os.remove(path + img_data['crop'])
        except:
            pass
        os.remove(path + f)
    print()

def main():
    parser.add_argument("-i", "--input", help="Folder with input data to create metadata for", default="./Images/Frames/")
    parser.add_argument("-db", "--database", help="JSON file with exported MongoDB", default="./Images/image_dataset.data.json")
    parser.add_argument("--database-port", help="MongoDB port", default=27017)
    parser.add_argument("--database-address", help="MongoDB address", default="localhost")
    parser.add_argument("-E", "--export", help="Export MongoDB to JSON", action="store_true")
    parser.add_argument("-T", "--max-threads", help="Set the maximum number of threads", default=4, type=int)
    parser.add_argument("-m", "--model", help="Select the model used for face processing", default=3, type=int)
    parser.add_argument("-D", "--delete", help="Delete all metadata and associated files", action="store_true")
    parser.add_argument("-Q", "--quick", help="Disables extra face detection scanning at the cost of less images being tagged", action="store_true")
    parser.add_argument("-N", "--normalize", help="Normalize already created metadata", action="store_true")
    parser.add_argument("-S", "--start", help="Index of file in a folder from which to start with", type=int, default=0)
    parser.add_argument("-C", "--count", help="Total number of files to scan", type=int, default=-1)

    args=parser.parse_args()

    print("Started")
    if args.delete:
        deleteOldMetadata(args.input)
        exit(0)
    elif args.normalize:
        MetadataNormalizer(args.input)
        exit(0)
    elif args.export:
        ExportDBToJson(args.database_address, args.database_port, args.database)
        exit(0)

    print("Scanning for existing crops")
    allowed_extensions = ['.json']
    folder_contents = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f)) and os.path.splitext(os.path.join(args.input, f))[1] in allowed_extensions]
    ignore_files = []

    for i, f in enumerate(folder_contents):
        if i % 100 == 0:
            print(f"\r{i / len(folder_contents) * 100:.2f}%", end="")
        with open(args.input + f, 'r') as cin:
            img_data = json.loads(cin.read())
        if os.path.exists(args.input + img_data['crop']):
            ignore_files.append(img_data['crop'])
    print(f"\rFound {len(ignore_files)} cropped images")

    print("Scanning existing images")
    allowed_extensions = ['.png', '.jpg', '.jpeg']
    folder_contents = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f)) and os.path.splitext(os.path.join(args.input, f))[1] in allowed_extensions and f not in ignore_files]
    folder_contents = folder_contents[args.start:]

    if args.count != -1:
        folder_contents = folder_contents[:args.count]

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
    print(f"Found {len(folder_contents)} images")

    # database preparations
    print("Dropping MongoDB collection")
    dbclient = pymongo.MongoClient(f"mongodb://{args.database_address}:{args.database_port}/")
    dbflag = True
    try:
        dbclient.server_info()
        dbfileparsed = os.path.basename(args.database).split('.')
        if len(dbfileparsed) == 3 and dbfileparsed[-1] == 'json':
            database = dbclient[dbfileparsed[0]]
            database.drop_collection(dbfileparsed[1])
        print("Dropped MongoDB collection")
    except:
        dbflag = False
        print("Can't access MongoDB")

    # files = [[_] for _ in folder_contents]

    input_data = [(args.input, files[_], args.model, args.quick, args.database_address, args.database_port, args.database) for _ in range(len(files))]
    print("Created input data for workers")
    print("Launching threads")
    start_time = time.perf_counter()
    process_pool = multiprocessing.Pool(min(args.max_threads, len(input_data)))
    process_pool.starmap(MetadataCreatorThread, input_data)

    ExportDBToJson(args.database_address, args.database_port, args.database)
    print(f"Total duration: {time.perf_counter() - start_time:.2f}s")

if __name__ == "__main__":
    multiprocessing.freeze_support()

    main()