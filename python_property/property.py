class Grid:
    def __init__(self):
        self._dx = 10.

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, value):
        if (value < 0):
            raise ValueError("dx cannot be negative")
        else:
            self._dx = value

grid = Grid()

print("dx = {0}".format(grid.dx))

grid.dx = 5.

print("dx = {0}".format(grid.dx))

grid.dx = -5.
