# if we have to find the points
def evaluate_line1(f, x0, x1, alpha):
    fx0 = f(x0)
    fx1 = f(x1)

    m = (fx1 - fx0)/(x1-x0)

    y = lambda x: m * (x - x0) + fx0

    return y(alpha)

# if the points are given
def evaluate_line2(point1, point2, alpha):
    (x0, fx0) = point1
    (x1, fx1) = point2

    m = (fx1 - fx0)/(x1-x0)

    y = lambda x: m * (x - x0) + fx0

    return y(alpha)