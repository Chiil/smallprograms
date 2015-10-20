import numpy as np
import pylab as pl

pl.ioff()

rand_field = np.random.rand(10,10)

fig = pl.figure()
cm = pl.pcolormesh(rand_field, vmin=0, vmax=1)
pl.colorbar()

press_coord = None

def onclick(event):
    indexx = int(event.xdata)
    indexy = int(event.ydata)
    print("Index ({0},{1}) will be set to zero".format(indexx, indexy))
    rand_field[indexy, indexx] = 0.
    cm.set_array(rand_field.ravel())
    event.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

pl.show()
