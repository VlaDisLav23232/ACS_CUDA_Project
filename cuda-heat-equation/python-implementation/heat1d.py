import numpy
from matplotlib import pyplot

length = 2
k = .466
temp_left = 200
temp_right = 200

total_time = 4

dx = .1
x_vec = numpy.linspace(0, length, int(length/dx))

dt = .0001
t_vec = numpy.linspace(0, total_time, int(total_time/dt))

u = numpy.zeros([len(t_vec), len(x_vec)])

u[:, 0] = temp_left
u[:, -1] = temp_right

for t in range(1, len(t_vec)-1):
    for x in range(1, len(x_vec)-1):
        u[t+1, x] = k * (dt / dx**2) * (u[t, x+1] - 2*u[t, x] +
                                        u[t, x-1]) + u[t, x]

    pyplot.plot(x_vec, u[t], 'black')
    pyplot.pause(.001)
    pyplot.cla()


pyplot.plot(x_vec, u[0])
pyplot.ylabel("Temperature (C˚)")
pyplot.xlabel("Distance Along Rod (m)")
pyplot.show()