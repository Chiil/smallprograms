class Grid:
    def __init__(self):
        self._dx = 10.

    @property
    def dx(self):
        return self._dx

grid = Grid()

print("dx = {0}".format(grid.dx))

grid.dx = 5.
