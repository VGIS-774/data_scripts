import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from PIL import Image, ImageDraw

videos_path = "data/archive/"
N = 30
polygon = [(135, 288), (40, 65), (120, 70), (384, 90), (384, 0), (0, 0), (0, 288)]
size = 288, 384
fps = 30

if not os.path.exists('data/background_videos'):
    os.makedirs('data/background_videos')

for folder in os.walk(videos_path):
    for name in folder[1]:
        print(name)
        path = videos_path + name + "/"
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for file in onlyfiles:
            full_path = path + file
            video = cv2.VideoCapture(full_path)
            FOI = video.get(cv2.CAP_PROP_FRAME_COUNT) * np.linspace(0, 0.99, N)

            frames = []
            for frame_index in FOI:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = video.read()
                frames.append(frame)

            try:
                backgroundFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

                img = Image.new('L', (backgroundFrame.shape[1], backgroundFrame.shape[0]), 0)

                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                mask = np.array(img)

                new_video = []

                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
                    ret, frame = video.read()
                    frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * mask + cv2.cvtColor(backgroundFrame,
                                                                                                   cv2.COLOR_BGR2GRAY) * (
                                             np.ones((288, 384)) - mask), dtype='uint8')
                    new_video.append(frame)

                out = cv2.VideoWriter('data/background_videos/{}_{}.mp4'.format(name, file), cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps, (size[1], size[0]), False)
                for frame in new_video:
                    data = np.random.randint(0, 256, size, dtype='uint8')
                    out.write(frame)
                out.release()

            except:
                print("error")
