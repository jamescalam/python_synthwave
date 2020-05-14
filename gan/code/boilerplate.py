import sys
import tensorflow as tf
sys.path.insert(0, 'code')

import data as d
import model as m

# settings
NEW_DATA = True
NEW_DATA2 = False
NEW_DATA3 = False


BATCHSIZE = 64
BUFFER = 10000
EPOCHS = 5000
LATENT_UNITS = 100
VIS = True
DOWNSAMPLE = 2
COLOUR = True

G_OPT = 1e-3
D_OPT = 1e-4

VIDEOS = [
    "https://www.youtube.com/watch?v=85bkCmaOh4o",
    "https://www.youtube.com/watch?v=KPa1_7AF1lM&list=RD85bkCmaOh4o&index=2",
    "https://www.youtube.com/watch?v=ICcFMBzOnYs&list=RD85bkCmaOh4o&index=3",
    "https://www.youtube.com/watch?v=wOMwO5T3yT4&list=RD85bkCmaOh4o&index=4",
    "https://www.youtube.com/watch?v=WI4-HUn8dFc&list=RD85bkCmaOh4o&index=5",
    "https://www.youtube.com/watch?v=GBUCmMxmup0&list=RD85bkCmaOh4o&index=6",
    "https://www.youtube.com/watch?v=XccPsuqAz4E&list=RD85bkCmaOh4o&index=16",
    "https://www.youtube.com/watch?v=yhCuCqJbOVE&list=RD85bkCmaOh4o&index=20",
    "https://www.youtube.com/watch?v=Sk-9vORoAZo&list=RD85bkCmaOh4o&index=21",
    "https://www.youtube.com/watch?v=-UaZDpF3bS0&list=RD85bkCmaOh4o&index=25",
    "https://www.youtube.com/watch?v=r4J5nKy6dDw&list=RD85bkCmaOh4o&index=30",
    "https://www.youtube.com/watch?v=lWKO3_ti3Gs&list=RD85bkCmaOh4o&index=28",
    "https://www.youtube.com/watch?v=emNgfuw8vlA&list=RD85bkCmaOh4o&index=30",
    "https://www.youtube.com/watch?v=HfdhaCo-c3M",
    "https://www.youtube.com/watch?v=PWMTDRWJqu4",
    "https://www.youtube.com/watch?v=mZvQ9ipTK_8",
    "https://www.youtube.com/watch?v=GfKs8oNP9m8",
    "https://www.youtube.com/watch?v=pvomDfQmhWQ&list=PL9R15-E_ygVdRXnL_TtLRunDid5eqBU7u",
    "https://www.youtube.com/watch?v=Qqc5vNajuIg",
    "https://www.youtube.com/watch?v=zn-o1kS8qgk",
    "https://www.youtube.com/watch?v=2Ns_E0qLTZM",
    "https://www.youtube.com/watch?v=ZULGQzE3MUA"
]

if NEW_DATA:
    for http in VIDEOS:
        vids = d.GetVideo(http)
        vids.capture(frames=100, step=70, all=True)
        vids.clear()

if NEW_DATA2:
    vids = d.GetVideo("https://www.youtube.com/watch?v=pvomDfQmhWQ&list=PL9R15-E_ygVdRXnL_TtLRunDid5eqBU7u")
    vids.capture(frames=1, step=10, all=True)
    vids.clear()
    
if NEW_DATA3:
    # this contains limited selection of synthwave neon grid loops
    VIDEOS = [
        "https://www.youtube.com/watch?v=Qqc5vNajuIg",
        "https://www.youtube.com/watch?v=zn-o1kS8qgk",
        "https://www.youtube.com/watch?v=2Ns_E0qLTZM",
        "https://www.youtube.com/watch?v=ZULGQzE3MUA"
    ]
    for vid in VIDEOS:
        vids = d.GetVideo(vid)
        vids.capture(frames=200, step=10)

DATA = d.Data(colour=COLOUR, downsample=DOWNSAMPLE, limit=1500)  # load data

DATA.format_data(BUFFER, BATCHSIZE)  # format data
DATA.shapes()

# create image and noise to input as training progress examples
noise = tf.random.normal(
    [int(640/DOWNSAMPLE), int(360/DOWNSAMPLE)]
    )
img = d.import_image(
    'C:/Users/James/Google Drive/Articles/synthwave_gans/diego-jimenez-A-NVHPka9Rk-unsplash.jpg',
    False,
    DOWNSAMPLE)
img = tf.convert_to_tensor(img)
vis_img = [noise, img]

#LATENT_UNITS = [DATA.height, DATA.width]
# setup generator
G = m.Generator(LATENT_UNITS, DATA.width, DATA.height, DATA.depth)
G.dcgan_v2()  # deep convolution GAN
G.model.summary()
G.optimiser(G_OPT)  # setup optimiser

# setup discriminator
D = m.Discriminator(DATA.width, DATA.height, DATA.depth)
D.dcgan()  # deep convolution GAN
D.model.summary()
D.optimiser(D_OPT)  # setup optimiser

model = m.Train(G, D, BATCHSIZE, LATENT_UNITS)
model.fit(DATA.data, EPOCHS, VIS, plot_grid=1, colour=True,
          figsize=(32, 18), vis_arrays=None)
