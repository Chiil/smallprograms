# set the values
n = 64
xsize = 0.5
stretch = 1.02

# non-equidistant grid
dx = xsize*(1-stretch)/(1-stretch**n)
x[0] = 0.5*dx
for k in range(1,n):
  x[k] = x[k-1] + 0.5*dx
  dx   *= stretch
  x[k] += 0.5*dx

y = np.sin(x)

# equidistant grid
nnew = 100
dx_new = xsize / nnew
x_new = np.linspace(0.5*dx_new, xsize-0.5*dx_new, nnew)


