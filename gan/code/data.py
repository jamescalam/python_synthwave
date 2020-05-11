from pytube import YouTube
import os
import cv2
import numpy as np
from PIL import Image
import skimage.measure
import tensorflow as tf
from mnist import MNIST


class GetVideo:
    def __init__(self, http):
        # check if video is part of playlist
        if "&list" in http:
            # if it is, remove playlist part of http
            http = http.split("&list")[0]

        video = YouTube(http)  # initialise video object
        # download to videos/tmp
        video.streams.get_by_itag(18).download('../videos/tmp')

        self.title = video.title  # get video title

        self.path = os.path.join('../videos/tmp', f"{self.title}.mp4")  # get video path

    def capture(self, frames, step, all=False):
        try:
            cap = cv2.VideoCapture(self.path)
        except FileNotFoundError:
            self.path = os.listdir('../videos/tmp')[0]
            cap = cv2.VideoCapture(self.path)

        # begin capture 1/10th through video to avoid intros
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # get number of frames in video

        # calculate frame at which we are 1/10th of way through
        tenth = round(n_frames / 10)

        count = 0
        img = 0

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                print(f"Reached no success, {img} images downloaded.")
                break

            count += 1

            if count < tenth and not all:
                # if we haven't reached the 1/10th point yet, don't begin saving frames
                # and skip this iteration
                continue

            if count % step == 0:
                path = os.path.join('../videos/img', f"{self.title}_{img}.jpg")
                cv2.imwrite(path, frame)
                img += 1

            if img >= frames and not all:
                print(f"Reached max number of frames, {img} images downloaded.")
                break

            # if we have decided to pull full video with 'all'
            if count >= n_frames - 3 * step:
                break

        cap.release()
        cv2.destroyAllWindows()

    def clear(self, path='../videos/tmp'):
        vids = (x for x in path if '.mp4' in x)
        for vid in vids:
            print(f"Deleting '{vid}'")
            os.remove(os.path.join(path, vid))


class Mnist:
    def __init__(self, path='../data/mnist'):
        data = MNIST(path)
        self.data, self.labels = data.load_training()
        print(f"MNIST data loaded, single array:\n{self.data[0]}")

    def format_data(self, buffer, batchsize):
        # reshape and normalise to -1 -> 1
        self.data = [np.reshape((np.array(x)/127.5)-1., (28, 28, 1)) for x in self.data]
        print(f"data min/max = {self.data[0].min()}/{self.data[0].max()}")
        self.data = tf.data.Dataset.from_tensor_slices(self.data)
        self.data = self.data.shuffle(buffer).batch(batchsize, drop_remainder=True)

class Data:
    def __init__(self, color=False):
        self.data = []

        for i, img in enumerate(os.listdir('../videos/img')):
            # load the image
            image = Image.open(os.path.join("../videos/img", img))
            if color:
                pass
            else:
                # convert to grayscale
                image = image.convert('L')
            # convert to numpy array
            array = np.asarray(image)

            if i % 10 == 0:
                print(f"Loaded image {i}")

            if array.shape[0] == 360 and array.shape[1] == 640 and not color:
                array = skimage.measure.block_reduce(array, (5, 5), np.mean)
                self.data.append(array/255.)  # divide by 255 acts as normalisation
            elif array.shape[0] == 360 and array.shape[1] == 640 and color:
                array = skimage.measure.block_reduce(array, (5, 5), np.mean)
                self.data.append(array/255.)  # divide by 255 acts as normalisation
            else:
                print(f"shape[0] = {array.shape[0]}\nshape[1] = {array.shape[1]}")

        self.width, self.height = self.data[0].shape

    def format_data(self, buffer, batchsize):
        # reshape
        self.data = [np.reshape(x, (x.shape[0], x.shape[1], 1)) for x in self.data]
        self.data = tf.data.Dataset.from_tensor_slices(self.data)
        self.data = self.data.shuffle(buffer).batch(batchsize, drop_remainder=True)

    def shapes(self):
        # print array shapes
        for d in self.data:
            print(d.shape)
