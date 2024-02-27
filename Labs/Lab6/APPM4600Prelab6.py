import numpy

def forward_difference(f, s, h):
    return (f(s+h) - f(s))/h

def centered_difference(f, s, h):
    return (f(s+h) - f(s-h))/(2*h)

h = 0.01*2.**(-numpy.arange(0, 10))

f = lambda x: numpy.cos(x)

s = numpy.pi/2

print(forward_difference(f, s, h))
print(centered_difference(f, s, h))