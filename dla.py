import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


PERIMETER = 0
PARTICLE = 3


class DLA:
    def __init__(self, n, padding=5):
        assert n > 10, "Grid size must be greater than 10"
        self.n = n
        self.grid = np.zeros((n, n), dtype=np.uint8)
        # Set of perimeter points
        self.perimeter = set()
        # Set of particle points
        self.occupied = set()
        self.n_2 = n // 2
        self.seed = (self.n_2, self.n_2)
        # Padding used for starting ring
        self.padding = padding
        # Radius of ring that walkers are released from. This is the "starting
        # ring"
        self.ringr = padding
        self.ringr2 = self.ringr * self.ringr
        # Radius of circle marking the out-of-bounds area. Walkers that reach
        # this bound are discarded and another is started.
        self.maxr = self.n_2
        self.maxr2 = self.maxr * self.maxr
        self.nx = [1, -1, 0, 0]
        self.ny = [0, 0, 1, -1]
        self.dirs = list(zip(self.nx, self.ny))

        self.grid[self.seed] = PARTICLE
        self.occupied.add(self.seed)
        self.add_perimeter(self.seed)
        # Cache a bunch of random values to avoid repeatedly calling randint
        # for a single value. get_randint() below refreshes the cache once it
        # is exhausted.
        # This was done because randint calls were accounting for nearly 30% of
        # the runtime
        self.irand = 0
        self.nrand = 1024 * 10
        self.rand_vals = np.random.randint(4, size=self.nrand)

        self.done = False

    def step(self):
        if not self.done:
            # Walk a particle toward seed
            pt = self.walk_particle()
            self.grid[pt] = PARTICLE
            self.perimeter.discard(pt)
            self.occupied.add(pt)
            self.add_perimeter(pt)
            r2 = ((pt[0] - self.n_2) * (pt[0] - self.n_2)) + (
                (pt[1] - self.n_2) * (pt[1] - self.n_2)
            )
            r = np.sqrt(r2)
            self.ringr = min(r + self.padding, self.maxr)
            self.ringr2 = self.ringr * self.ringr
            self.done = r >= self.maxr - 2

    def add_perimeter(self, pt):
        nn = [(pt[0] + i, pt[1] + j) for i, j in self.dirs]
        for p in nn:
            if p not in self.occupied and p not in self.perimeter:
                self.perimeter.add(p)
                self.grid[p] = PERIMETER

    def walk_particle(self):
        contact = False
        while not contact:
            # start
            ix, iy = self.get_random_start()
            r2 = 0
            restart = False
            while not restart and not contact:
                dx, dy = self.dirs[self.get_randint()]
                # Increase step size of walker if it is outside of the starting
                # ring.
                speedup = 1 + ((r2 > self.ringr2) * 4)
                ix += dx * speedup
                iy += dy * speedup
                pt = (ix, iy)
                r2 = ((ix - self.n_2) * (ix - self.n_2)) + (
                    (iy - self.n_2) * (iy - self.n_2)
                )
                # Check if walker has reached cluster
                contact = pt in self.perimeter
                # Check if walker has walked out of allowed area
                restart = r2 >= self.maxr2
        return pt

    def get_random_start(self):
        # Produce a walker starting point that sits on the starting ring
        theta = np.random.rand() * 2 * np.pi
        ix = int(self.ringr * np.cos(theta)) + self.n_2
        iy = int(self.ringr * np.sin(theta)) + self.n_2
        return (ix, iy)

    def get_randint(self):
        if self.irand >= self.nrand:
            self.irand = 0
            self.rand_vals = np.random.randint(4, size=self.nrand)
        v = self.rand_vals[self.irand]
        self.irand += 1
        return v


class SimRunner:
    def __init__(self, sim, max_iters):
        self.sim = sim
        self.max_iters = max_iters

    def run(self):
        i = 1
        while not sim.done and i < self.max_iters:
            sim.step()


class SimAnimation:
    def __init__(self, sim, interval):
        self.sim = sim
        self.fig = plt.figure()
        self.im = None
        self.ani = None
        self.interval = interval
        self.paused = False

    def init(self):
        self.im = plt.imshow(
            self.sim.grid, interpolation="none", animated=True, cmap="gray"
        )
        plt.axis("off")
        return (self.im,)

    def update(self, *args):
        self.sim.step()
        self.im.set_data(self.sim.grid)
        return (self.im,)

    def on_click(self, event):
        if event.key != " ":
            return
        if self.paused:
            self.ani.event_source.start()
            self.paused = False
        else:
            self.ani.event_source.stop()
            self.paused = True

    def run(self):
        self.fig.canvas.mpl_connect("key_press_event", self.on_click)
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.init,
            interval=self.interval,
            blit=True,
        )
        plt.show()


if __name__ == "__main__":
    sim = DLA(300)
    # SimAnimation(sim, 1).run()
    SimRunner(sim, 10000).run()