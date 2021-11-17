import os
import random
import cv2
import matplotlib
import numpy as np

videos_path_train = "dataset/train/Videos"
segmentations_path_train = "dataset/train/InstanceSegmentation"

videos_path_validation = "dataset/validation/Videos"
segmentations_path_validation = "dataset/validation/InstanceSegmentation"

backgrounds_videos = "data/background_videos/"

test_videos_path = "data/human_videos/"

frame_number = 5
train_percentage = 0.8

if not os.path.exists('mini_project_dataset/train/person_in_water'):
    os.makedirs('mini_project_dataset/train/person_in_water')

if not os.path.exists('mini_project_dataset/train/no_person_in_water'):
    os.makedirs('mini_project_dataset/train/no_person_in_water')

if not os.path.exists('mini_project_dataset/validation/person_in_water'):
    os.makedirs('mini_project_dataset/validation/person_in_water')

if not os.path.exists('mini_project_dataset/validation/no_person_in_water'):
    os.makedirs('mini_project_dataset/validation/no_person_in_water')

path, dirs, background_files = next(os.walk(backgrounds_videos))
background_files.sort()
train_backgrounds = background_files[:int(len(background_files) * train_percentage)]

path, dirs, video_files = next(os.walk(videos_path_train))
file_count = len(video_files)
video_files.sort()

path, dirs, segmentation_files = next(os.walk(segmentations_path_train))
segmentation_files.sort()

# define polygon points
points = np.array([[[115, 288], [50, 100], [0, 100], [0, 288]]], dtype=np.int32)

for video_number in range(file_count - 1):
    try:
        background_video = cv2.VideoCapture(backgrounds_videos + random.choice(train_backgrounds))
        number_of_frames = background_video.get(cv2.CAP_PROP_FRAME_COUNT)
        background_video.set(cv2.CAP_PROP_POS_FRAMES, random.randrange(number_of_frames - frame_number))

        video_path = videos_path_train + "/{}".format(video_files[video_number])
        segmentation_path = segmentations_path_train + "/{}".format(segmentation_files[video_number])

        video_cap = cv2.VideoCapture(video_path)
        segmentation_cap = cv2.VideoCapture(segmentation_path)

        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_number % 100 == 0:
            print("video number {}".format(video_number))

        frames = []
        background_frames = []
        for frame in range(0, frame_count, int(frame_count / 2)):
            video_ret, video_frame = video_cap.read()
            segmentation_ret, segmentation_frame = segmentation_cap.read()

            video_gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
            segmentation_gray = cv2.cvtColor(segmentation_frame, cv2.COLOR_BGR2GRAY)

            video_poly = video_gray.copy()
            cv2.polylines(video_poly, [points], True, (0, 0, 255), 1)

            mask = np.zeros_like(video_gray)
            cv2.fillPoly(mask, [points], (1))

            image = video_gray * mask
            image = image[100:, :115]
            frames.append(image)

            background_ret, background_frame = background_video.read()

            background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)

            background_poly = background_gray.copy()
            cv2.polylines(background_poly, [points], True, (0, 0, 255), 1)

            mask = np.zeros_like(background_gray)
            cv2.fillPoly(mask, [points], (1))

            background = background_gray * mask
            background = background[100:, :115]
            background_frames.append(background)

        segmentation_values = segmentation_gray[np.where(mask == 1)]

        person_in_water = sum(segmentation_values / 255) > 10

        rgb = np.dstack((frames[0], frames[1], frames[2]))
        rgb = rgb.astype(np.uint8)

        background_rgb = np.dstack((background_frames[0], background_frames[1], background_frames[2]))
        background_rgb = background_rgb.astype(np.uint8)

        if person_in_water:
            cv2.imwrite("mini_project_dataset/train/person_in_water/image_{}.png".format(video_number), rgb)
            cv2.imwrite("mini_project_dataset/train/no_person_in_water/image_{}.png".format(video_number), background_rgb)

            #matplotlib.image.imsave('mini_project_dataset/train/person_in_water/image_{}.png'.format(video_number), rgb)
            #matplotlib.image.imsave('mini_project_dataset/train/no_person_in_water/image_{}.png'.format(video_number), background_rgb)
        # else:
        # matplotlib.image.imsave('mini_project_dataset/no_person_in_water/image_{}.png'.format(video_number), rgb)

    except NameError as e:
        print(e)
        print("error")
    except IndexError as e:
        print(e)
    except ValueError as e:
        print(e)

path, dirs, background_files = next(os.walk(backgrounds_videos))
background_files.sort()
validation_backgrounds = background_files[int(len(background_files) * train_percentage):]

path, dirs, video_files = next(os.walk(videos_path_validation))
file_count = len(video_files)
video_files.sort()

path, dirs, segmentation_files = next(os.walk(segmentations_path_validation))
segmentation_files.sort()

# define polygon points
points = np.array([[[115, 288], [50, 100], [0, 100], [0, 288]]], dtype=np.int32)

for video_number in range(file_count - 1):
    try:
        background_video = cv2.VideoCapture(backgrounds_videos + random.choice(train_backgrounds))
        number_of_frames = background_video.get(cv2.CAP_PROP_FRAME_COUNT)
        background_video.set(cv2.CAP_PROP_POS_FRAMES, random.randrange(number_of_frames - frame_number))

        video_path = videos_path_validation + "/{}".format(video_files[video_number])
        segmentation_path = segmentations_path_validation + "/{}".format(segmentation_files[video_number])

        video_cap = cv2.VideoCapture(video_path)
        segmentation_cap = cv2.VideoCapture(segmentation_path)

        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_number % 100 == 0:
            print("video number {}".format(video_number))

        frames = []
        background_frames = []
        for frame in range(0, frame_count, int(frame_count / 2)):
            video_ret, video_frame = video_cap.read()
            segmentation_ret, segmentation_frame = segmentation_cap.read()

            video_gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
            segmentation_gray = cv2.cvtColor(segmentation_frame, cv2.COLOR_BGR2GRAY)

            video_poly = video_gray.copy()
            cv2.polylines(video_poly, [points], True, (0, 0, 255), 1)

            mask = np.zeros_like(video_gray)
            cv2.fillPoly(mask, [points], (1))

            image = video_gray * mask
            image = image[100:, :115]
            frames.append(image)

            background_ret, background_frame = background_video.read()

            background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)

            background_poly = background_gray.copy()
            cv2.polylines(background_poly, [points], True, (0, 0, 255), 1)

            mask = np.zeros_like(background_gray)
            cv2.fillPoly(mask, [points], (1))

            background = background_gray * mask
            background = background[100:, :115]
            background_frames.append(background)

        segmentation_values = segmentation_gray[np.where(mask == 1)]

        person_in_water = sum(segmentation_values / 255) > 10

        rgb = np.dstack((frames[0], frames[1], frames[2]))
        rgb = rgb.astype(np.uint8)

        background_rgb = np.dstack((background_frames[0], background_frames[1], background_frames[2]))
        background_rgb = background_rgb.astype(np.uint8)

        if person_in_water:
            cv2.imwrite("mini_project_dataset/validation/person_in_water/image_{}.png".format(video_number), rgb)
            cv2.imwrite("mini_project_dataset/validation/no_person_in_water/image_{}.png".format(video_number), background_rgb)
        # else:
        # matplotlib.image.imsave('mini_project_dataset/no_person_in_water/image_{}.png'.format(video_number), rgb)
    except NameError as e:
        print(e)
        print("error")
    except IndexError as e:
        print(e)
    except ValueError as e:
        print(e)

path, dirs, files = next(os.walk(test_videos_path))

frames = []
chunks = []

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

for file in files:
    if file != '.DS_Store':
        video_cap = cv2.VideoCapture(test_videos_path + "/" + file)
        video_ret = True
        while video_ret:
            video_ret, video_frame = video_cap.read()
            if not video_ret:
                break
            gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)
        video_chunks = list(divide_chunks(frames, 5))
        for video_chunk in video_chunks:
            chunks.append(video_chunk)
        frames = []

points = np.array( [[[115, 288], [50,100], [0,100], [0,288]]], dtype=np.int32 )

number = 1
for chunk in chunks:
    if len(chunk) == 5:
        frames = []
        for frame_index in range(0, 5, 2):
            frame = chunk[frame_index]

            video_poly = frame.copy()
            cv2.polylines(video_poly, [points], True, (0,0,255), 1)

            mask = np.zeros_like(frame)
            cv2.fillPoly(mask, [points], (1))

            image = frame * mask
            image = image[100:,:115]

            frames.append(image)

        rgb = np.dstack((frames[0], frames[1], frames[2]))
        rgb = rgb.astype(np.uint8)
        cv2.imwrite('mini_project_dataset/test/image_{}.png'.format(number), rgb)
        number += 1
