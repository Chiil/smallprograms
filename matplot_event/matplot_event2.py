import numpy as np
import pylab as pl

pl.ioff()

rand_field = np.random.rand(10,10)

fig = pl.figure()
cm = pl.pcolormesh(rand_field, vmin=0, vmax=1)
pl.colorbar()

x_press = None
y_press = None

def onpress(event):
    global x_press, y_press
    x_press = int(event.xdata) if (event.xdata != None) else None
    y_press = int(event.ydata) if (event.ydata != None) else None

def onrelease(event):
    global x_press, y_press
    x_release = int(event.xdata) if (event.xdata != None) else None
    y_release = int(event.ydata) if (event.ydata != None) else None

    if (x_press != None and y_press != None and x_release != None and y_release != None):
        (xs, xe) = (x_press, x_release+1) if (x_press <= x_release) else (x_release, x_press+1)
        (ys, ye) = (y_press, y_release+1) if (y_press <= y_release) else (y_release, y_press+1)
        print("Slice [{0}:{1},{2}:{3}] will be set to zero".format(xs, xe, ys, ye))
        rand_field[ys:ye, xs:xe] = 0.
        cm.set_array(rand_field.ravel())
        event.canvas.draw()

    x_press = None
    y_press = None

cid_press   = fig.canvas.mpl_connect('button_press_event'  , onpress  )
cid_release = fig.canvas.mpl_connect('button_release_event', onrelease)

pl.show()
