import sys
sys.path.insert(0, '')

import data as d
import model as m

BATCHSIZE = 64
BUFFER = 10000
EPOCHS = 500
LATENT_UNITS = 100
VIS = True

G_OPT = 1e-3
D_OPT = 5e-5

data = d.Mnist()
data.format_data(BUFFER, BATCHSIZE)

G = m.Generator(LATENT_UNITS, 28, 28)
G.mnist_dcgan()  # switch to mnist dataset model
G.optimiser(G_OPT)  # setup optimiser
G.model.summary()  # print model summary

D = m.Discriminator()
D.mnist_dcgan()  # switch to mnist dataset model
D.optimiser(D_OPT)  # setup optimiser
D.model.summary()  # print model summary

model = m.Train(G, D, 28, 28, BATCHSIZE, LATENT_UNITS)

model.fit(data.data, EPOCHS, VIS)