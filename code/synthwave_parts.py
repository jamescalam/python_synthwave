# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 08:58:11 2020

@author: James
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def make_sun(ax, cx, cy, r):
    def make_grad(cx, cy, r, alpha):
        circ = mpatches.Circle((cx, cy), r, facecolor='none')
        ax.add_patch(circ)

        plt.imshow([[1, 1], [0, 0]],
                   cmap=cm.plasma,
                   interpolation='bicubic',
                   aspect='auto',
                   extent=(cx-r, cx+r, cy-r, cy+r),
                   alpha=alpha,
                   clip_path=circ,
                   clip_on=True)

    make_grad(cx, cy, r+.2, alpha=.4)
    make_grad(cx, cy, r+.4, alpha=.4)
    make_grad(cx, cy, r+.8, alpha=.4)
    make_grad(cx, cy, r+.16, alpha=.4)
    make_grad(cx, cy, r, alpha=1)


    ax.plot([cx-r, cx+r], [cy-(r*0.8), cy-(r*0.8)],
             linewidth=r*.4, color='k')

    ax.plot([cx-r, cx+r], [cy-(r*0.6), cy-(r*0.6)],
             linewidth=r*.3, color='k')

    ax.plot([cx-r, cx+r], [cy-(r*0.4), cy-(r*0.4)],
             linewidth=r*.2, color='k')

    ax.plot([cx-r*1.1, cx+r*1.1], [cy-(r*0.2), cy-(r*0.2)],
             linewidth=r*.1, color='k')


def make_skyline(ax, min_x, max_x, min_y, max_y, towers=30,
                 colour='k', heights=None):
    if heights is None:
        heights = np.random.randint(0, max_y-min_y, towers)

    x = np.linspace(min_x, max_x, towers)

    ax.bar(x, heights, width=(max_x/towers)*2, align='edge',
           bottom=min_y, color=colour)


def make_stars(ax, min_x, max_x, min_y, max_y, stars=100):
    #!!! TODO: to add alpha so that at min_y alpha = 0 and at max_y/2 alpha = 1
    x = np.random.uniform(min_x, max_x, stars)
    y = np.random.uniform(min_y, max_y, stars)
    alpha_multiplier = np.random.uniform(.5, 1, len(x))
    size = np.random.uniform(1, 6, len(x))

    for i in range(len(y)):
        # calculate relative height
        h = (y[i] - min_y) / (max_y - min_y)

        ax.scatter(x[i], y[i], alpha=h*alpha_multiplier[i]*.1,
                   s=size+.4, c='#ffffff', zorder=0)
        ax.scatter(x[i], y[i], alpha=h*alpha_multiplier[i]*.1,
                   s=size+.3, c='#ffffff', zorder=0)
        ax.scatter(x[i], y[i], alpha=h*alpha_multiplier[i]*.1,
                   s=size+.2, c='#ffffff', zorder=0)
        ax.scatter(x[i], y[i], alpha=h*alpha_multiplier[i]*.1,
                   s=size+.1, c='#ffffff', zorder=0)
        ax.scatter(x[i], y[i], alpha=h*alpha_multiplier[i]*.6,
                   s=size, c='#ffffff', zorder=0)

    return np.array([x, y, alpha_multiplier, size]).T
