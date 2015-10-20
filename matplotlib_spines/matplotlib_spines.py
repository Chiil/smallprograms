import pylab as pl

n = 10.

pl.rcParams['axes.linewidth']=n
pl.rcParams['xtick.major.width']=n
pl.rcParams['ytick.major.width']=n
pl.rcParams['xtick.major.size']=2.*n
pl.rcParams['ytick.major.size']=2.*n

pl.figure()
ax = pl.subplot(111)
ax.xaxis.set_tick_params(width=n)
ax.yaxis.set_tick_params(width=n)

ax.spines['right'].set_visible(False)
ax.get_yaxis().tick_left()
ax.spines['top'].set_visible(False)
ax.get_xaxis().tick_bottom()

for loc, s in ax.spines.items():
    s.set_capstyle('projecting')

