data_dirs = []
import json
import os.path
import random

import cv2
import numpy as np

data_folder = "data/synthetic/"
backgrounds_videos = "data/background_videos/"

blur = 3
frame_number = 5
size = 288, 384
fps = 30
train_percentage = 0.8

path, dirs, files = next(os.walk(data_folder))
dirs = [k for k in dirs if 'Dataset' in k]

for folder in dirs:
    file = open("{}/{}/captures_000.json".format(data_folder, folder))
    data = json.load(file)

    data_dirs.append((data["captures"][0]["filename"].split("/", 1)[0],
                      data["captures"][0]["annotations"][1]["filename"].split("/", 1)[0]))

    file.close()

if not os.path.exists('dataset/train/Videos'):
    os.makedirs('dataset/train/Videos')

if not os.path.exists('dataset/validation/Videos'):
    os.makedirs('dataset/validation/Videos')

if not os.path.exists('dataset/train/InstanceSegmentation'):
    os.makedirs('dataset/train/InstanceSegmentation')

if not os.path.exists('dataset/validation/InstanceSegmentation'):
    os.makedirs('dataset/validation/InstanceSegmentation')

video_count = 0

for rgb_folder, segmentation_folder in data_dirs:
    RGB_image_path = data_folder + rgb_folder + "/"
    instance_segmentation_path = data_folder + segmentation_folder + "/"

    path, dirs, files = next(os.walk(RGB_image_path))
    file_count = len(files)

    path, dirs, files = next(os.walk(backgrounds_videos))

    files.sort()
    train_backgrounds = files[:int(len(files) * train_percentage)]
    validation_backgrounds = files[int(len(files) * train_percentage):]

    final_images = []
    final_segmentations = []

    i = 1
    background_video = cv2.VideoCapture(backgrounds_videos + random.choice(train_backgrounds))
    for count in range(1, file_count + 1):
        # if True:
        try:
            rgb_image = cv2.imread(RGB_image_path + "rgb_{}.png".format(count))
            segmentation_image = cv2.imread(instance_segmentation_path + "Instance_{}.png".format(count))

            number_of_frames = background_video.get(cv2.CAP_PROP_FRAME_COUNT)
            background_video.set(cv2.CAP_PROP_POS_FRAMES, random.randrange(number_of_frames - frame_number))

            ret, background_image = background_video.read()
            background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)

            subtract = int((rgb_image.shape[1] - (
                    (background_image.shape[1] / background_image.shape[0]) * rgb_image.shape[0])) / 2)

            u = np.array([0, 0, 0])
            l = np.array([0, 0, 0])

            mask = cv2.inRange(segmentation_image, l, u)
            mask = cv2.blur(mask, (blur, blur))  # Maybe change to use guarsian filter
            if subtract != 0:
                mask = mask[:, subtract:-subtract]
            mask = cv2.resize(mask, (384, 288))

            rgb_image = cv2.blur(rgb_image, (blur, blur))  # Maybe change to use guarsian filter
            if subtract != 0:
                rgb_image = rgb_image[:, subtract:-subtract]
            rgb_image = cv2.resize(rgb_image, (384, 288))

            final = np.array(background_image * (mask / 255) + cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY) * (
                    np.ones((288, 384)) - (mask / 255)), dtype='uint8')

            if subtract != 0:
                final_segmentation_image = segmentation_image[:, subtract:-subtract]
            else:
                final_segmentation_image = segmentation_image
            final_segmentation_image = cv2.resize(final_segmentation_image, (384, 288))

            final_images.append(final)
            final_segmentations.append(final_segmentation_image)

            i += 1

            if (count) % frame_number == 0:
                if (count / file_count) < train_percentage:
                    background_video = cv2.VideoCapture(backgrounds_videos + random.choice(train_backgrounds))
                    out = cv2.VideoWriter('dataset/train/Videos/video_{}.mp4'.format(video_count),
                                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
                    for frame in final_images:
                        out.write(frame)
                    out.release()

                    out = cv2.VideoWriter('dataset/train/InstanceSegmentation/segmentation_{}.mp4'.format(video_count),
                                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
                    for frame in final_segmentations:
                        out.write(frame)
                    out.release()
                else:
                    background_video = cv2.VideoCapture(backgrounds_videos + random.choice(validation_backgrounds))
                    out = cv2.VideoWriter('dataset/validation/Videos/video_{}.mp4'.format(video_count),
                                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
                    for frame in final_images:
                        out.write(frame)
                    out.release()

                    out = cv2.VideoWriter(
                        'dataset/validation/InstanceSegmentation/segmentation_{}.mp4'.format(video_count),
                        cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
                    for frame in final_segmentations:
                        out.write(frame)
                    out.release()

                final_images = []
                final_segmentations = []

                video_count += 1
        except:
            background_video = cv2.VideoCapture(backgrounds_videos + random.choice(files))
            print("error")
