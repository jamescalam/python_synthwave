# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:20:02 2020

Ad-hoc code for building Synthwave visuals in Python's Matplotlib.
*** Really really not optimised or pretty.

@author: James
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation, cm
from matplotlib.colors import ListedColormap
sys.path.insert(0, r'./')
import synthwave_parts as synth

background = '#000000'
horizon = {'grad': cm.gnuplot(range(28)),  # only take black to purple part of gradient
           'dist': 18}
ground = {'grad': cm.gnuplot(range(12)),  # only black -> purple gradient
          'dist': 50}

matplotlib.rcParams['axes.facecolor'] = background

# key parameters
perspective = (0, 5)
motion_lines = 10
frames = 200
xlim = (-100, 100) 
ylim = (-50, 70)

# figure and axes setup
fig = plt.figure(figsize=(20, 12))
ax = plt.axes(xlim=xlim, ylim=ylim)
exp_debug = False

# create gradient ground foreground
plt.imshow([[0, 0], [1, 1]],
           cmap=ListedColormap(ground['grad']),
           interpolation='bicubic',
           aspect='auto',
           extent=(xlim[0], xlim[1], 0, -ground['dist']))

synth.make_sun(ax, 0, 40, 20)
stars = synth.make_stars(ax, xlim[0], xlim[1], 10, ylim[1])

# create gradient background
plt.imshow([[0, 0], [1, 1]],
           # only black-to-purple part of gnuplot
           cmap=ListedColormap(horizon['grad']),
           interpolation='bicubic',
           aspect='auto',
           extent=(xlim[0], xlim[1], 0, horizon['dist']),
           zorder=-1)

synth.make_skyline(ax, xlim[0], xlim[1], 0, 15, towers=70, colour='#000000')

# define glow building line function
def grid_line(x, y, c='#bc13fe'):
    plt.plot(x, y, color=c, linewidth=5, alpha=.2)
    plt.plot(x, y, color=c, linewidth=4, alpha=.2)
    plt.plot(x, y, color=c, linewidth=3, alpha=.2)
    plt.plot(x, y, color=c, linewidth=2.5, alpha=.2)
    plt.plot(x, y, 'w', linewidth=2, alpha=.6)


# exponential function for horizontal line movement
def exp(x, start, end, scale=0.02):
    if x >= start:
        x = x - start
        return np.exp(x * scale)
    else:
        x = end + x - start
        return np.exp(x * scale)


# creating vertical lines, originate at 0, 10 then mask all above horizon (y=0)
for i in range(int(xlim[0]*10), int(xlim[1]*10), int(xlim[1]/2)):
    x = np.linspace(perspective[0], i, 60)
    y = np.linspace(perspective[1], ylim[0], 60)
    # masking anything above horizon line
    y = np.ma.masked_where(y > 0, y)
    # plotting our perspective line
    grid_line(x, y)

# initialise motion dictionary
motion = {}
# create motion lines
for i in range(motion_lines):
    motion[str(i)] = {}
    # initialise motion line
    motion[str(i)]['glow1'], = ax.plot([], [], linewidth=5,
                                       color='#bc13fe', alpha=.2)
    motion[str(i)]['glow2'], = ax.plot([], [], linewidth=4,
                                       color='#bc13fe', alpha=.2)
    motion[str(i)]['glow3'], = ax.plot([], [], linewidth=3,
                                       color='#bc13fe', alpha=.2)
    motion[str(i)]['glow4'], = ax.plot([], [], linewidth=2.5,
                                       color='#bc13fe', alpha=.2)
    motion[str(i)]['line'], = ax.plot([], [], linewidth=2,
                                      color='w', alpha=.6)

# create horizon line (we'll make a few to create illusion of distance density
grid_line([xlim[0], xlim[1]], [0, 0])
grid_line([xlim[0], xlim[1]], [-.25, -.25])
grid_line([xlim[0], xlim[1]], [-.75, -.75])


if exp_debug:
    x = np.linspace(0, 50, 50)
    for j in range(motion_lines):
        plt.figure(figsize=(8, 8))
        # get start position for exponential func
        start = (50 / motion_lines) * j
        # y is assigned using the exponential function
        y = [exp(x_, start, 50) for x_ in x]
        grid_line(x, y)
        plt.tight_layout()
        plt.savefig(f'exp_{j}.png')

    plt.figure(figsize=(16,8))
    c = ['#bc13fe', '#1b03a3', '#0A9C9C', '#39ff14', '#ccff00',
         '#FAED27', '#FD5F00', '#ff073a', '#ff2965', '#ff69b4']
    for j in range(motion_lines):
        # get start position for exponential func
        start = (50 / motion_lines) * j
        # y is assigned using the exponential function
        y = [exp(x_, start, 50) for x_ in x]
        grid_line(x, y, c=c[j])
    plt.tight_layout()
    plt.savefig('exp_all.png')


# select random sample of the stars to make sparkle
stars = stars[stars[:, 1] > 30]  # only stars in the upper part of the sky
select = np.random.randint(0, len(stars), int(len(stars)*0.6))
stars = stars[list(select)]


# animation function, creates the plot frame by frame (i)
def animate(i):
    # movement lines will decrease y position exponentially with i
    x = [xlim[0], xlim[1]]
    # assign new line data for motion lines
    for j in range(motion_lines):
        # get start position for exponential func
        start = (frames / motion_lines) * j
        # y is assigned using the exponential function
        y = [-exp(i, start, frames) + 1] * 2
        # set data for each line
        for line in motion[str(j)]:
            motion[str(j)][line].set_data(x, y)

    # make the stars sparkle
    for star in stars:
        alpha = np.random.uniform(0, .3)
        ax.scatter(star[0], star[1], s=star[2]+.4, alpha=alpha,
                   c='#ffffff', zorder=0)
        dark = np.random.uniform(0, .3)
        ax.scatter(star[0], star[1], s=star[2]+.4, alpha=dark,
                   c='#000000', zorder=0)


anim = animation.FuncAnimation(fig, animate, frames=int(frames/10),
                               interval=20, blit=False)
plt.tight_layout()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
anim.save('../visuals/synthwave_outrun.gif', fps=60, writer='imagemagick')
