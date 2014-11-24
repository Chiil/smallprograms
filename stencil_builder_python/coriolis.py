#!/usr/bin/python

from StencilBuilder import *

u = Field("u" , uloc)
v = Field("v" , vloc)
w = Field("w" , wloc)

fc = Scalar("fc")

ut = fc * interpx( interpy( v ) )
vt = fc * interpx( interpy( u ) )

print("ut[ijk] = {0};\n".format(ut.getString(0,0,0,10)))
print("vt[ijk] = {0};\n".format(vt.getString(0,0,0,10)))
