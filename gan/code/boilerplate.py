import sys
sys.path.insert(0, 'code')

import data as d
import model as m

# settings
NEW_DATA = False
NEW_DATA2 = True
BATCHSIZE = 64
BUFFER = 10000
EPOCHS = 5000
LATENT_UNITS = 100
VIS = True

G_OPT = 1e-3
D_OPT = 2e-6

"""
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
    "https://www.youtube.com/watch?v=HfdhaCo-c3M"
]
"""

if NEW_DATA:
    # lots of visuals in one video
    vids = d.GetVideo("https://www.youtube.com/watch?v=PWMTDRWJqu4")
    vids.capture(frames=100, step=240, all=True)
    vids.clear()
    # lots of visuals in one video
    vids = d.GetVideo("https://www.youtube.com/watch?v=mZvQ9ipTK_8&t=4s")
    vids.capture(frames=100, step=240, all=True)
    vids.clear()
    # lots of visuals in one video
    vids = d.GetVideo("https://www.youtube.com/watch?v=GfKs8oNP9m8")
    vids.capture(frames=100, step=240, all=True)
    vids.clear()

if NEW_DATA2:
    vids = d.GetVideo("https://www.youtube.com/watch?v=pvomDfQmhWQ&list=PL9R15-E_ygVdRXnL_TtLRunDid5eqBU7u")
    vids.capture(frames=1, step=10, all=True)
    vids.clear()

DATA = d.Data()  # load data

DATA.format_data(BUFFER, BATCHSIZE)  # format data
DATA.shapes()

# setup generator
G = m.Generator(LATENT_UNITS, DATA.width, DATA.height, BATCHSIZE)
G.conv()  # switch to transpose convolution net
G.optimiser(G_OPT)  # setup optimiser

# setup discriminator
D = m.Discriminator(DATA.width, DATA.height, BATCHSIZE)
D.optimiser(D_OPT)  # setup optimiser

model = m.Train(G, D, DATA.width, DATA.height, BATCHSIZE, LATENT_UNITS)
model.fit(DATA.data, EPOCHS, VIS)

