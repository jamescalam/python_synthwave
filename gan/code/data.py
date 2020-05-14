from pytube import YouTube
import os
import cv2
import numpy as np
from PIL import Image
import skimage.measure
import tensorflow as tf
from mnist import MNIST


def downsample_rgb(array, downsample):
    """
    Function to downsample RGB array dimensions without affecting colour
    dimensions.
    """
    r = skimage.measure.block_reduce(array[:, :, 0], (downsample, downsample), np.mean)
    g = skimage.measure.block_reduce(array[:, :, 1], (downsample, downsample), np.mean)
    b = skimage.measure.block_reduce(array[:, :, 2], (downsample, downsample), np.mean)
    return np.stack((r, g, b), axis=-1)

def import_image(path, colour, downsample):
    # load the image
    image = Image.open(path)
    if colour:
        pass
    else:
        # convert to grayscale
        image = image.convert('L')
        
    # resize image if needed
    if image.size != (640, 360):
        # alert user
        print(f"Resizing image from {image.size} to (640, 360).")
        image = image.resize((640, 360))

    # convert to numpy array
    array = np.asarray(image)

    if not colour:
        # grayscale downsampling is easy
        array = skimage.measure.block_reduce(array, (downsample, downsample), np.mean)
    else:
        # rgb downsampling is still easy but required different method
        array = downsample_rgb(array, downsample)

    return array/127.5-1  # divide by 127.5 acts as normalisation
    


class GetVideo:
    def __init__(self, http):
        # check if video is part of playlist
        if "&list" in http:
            # if it is, remove playlist part of http
            http = http.split("&list")[0]
            
        print(f"Downloading '{http}'.")

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
    def __init__(self, colour=False, downsample=1, limit=1000):
        self.data = []

        for i, img in enumerate(os.listdir('../videos/img')):
            
            path = os.path.join("../videos/img", img)
            self.data.append(import_image(path, colour, downsample))
            
            # update user every 10 images
            if i % 10 == 0:
                print(f"Loaded image {i}")

            if i >= limit:
                print("Image limit reached.")
                break

        if colour:
            self.height, self.width, self.depth = self.data[0].shape
        else:
            self.height, self.width = self.data[0].shape
            self.depth = 1

    def format_data(self, buffer, batchsize):
        # if using grayscale, must make array 3-D with depth/z = 1
        if self.depth == 1:
            self.data = [np.reshape(x, (x.shape[0], x.shape[1], 1)) for x in self.data]
        # convert to TF dataset object
        self.data = tf.data.Dataset.from_tensor_slices(self.data)
        # shuffle the data and place into batches
        self.data = self.data.shuffle(buffer).batch(batchsize, drop_remainder=True)

    def shapes(self):
        # print array shapes
        for d in self.data:
            print(d.shape)
