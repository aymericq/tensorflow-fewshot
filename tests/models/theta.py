def compute_dLF():
    # m(theta, x) = (theta*x)^2
    x1 = 1.0
    x2 = 2.0
    x3 = 3.0
    y1 = 1.0
    y2 = 2.0
    y3 = 3.0
    theta = 1.0

    # mx1 = m(theta, x1)
    mx1 = theta**2 * x1**2
    mx2 = theta**2 * x2**2
    mx3 = theta**2 * x3**2

    # g1 = d/d_theta[(mx1 - y1)^2]
    g1 = 2 * (mx1 - y1) * 2 * theta * x1**2
    g2 = 2 * (mx2 - y2) * 2 * theta * x2**2
    g3 = 2 * (mx3 - y3) * 2 * theta * x3**2

    # dg1 = d/d_theta[g1, x=x1]
    dg1 = 2 * ((2 * theta * x1**2)**2 + (mx1 - y1) * 2 * x1**2)
    dg2 = 2 * ((2 * theta * x2**2)**2 + (mx2 - y2) * 2 * x2**2)
    dg3 = 2 * ((2 * theta * x3**2)**2 + (mx3 - y3) * 2 * x3**2)

    print("dg1", dg1)
    print("dg2", dg2)
    print("dg3", dg3)

    # m1x1 = m(theta - g1, x1)
    m1x1 = (theta - g1)**2 * x1**2
    m2x2 = (theta - g2)**2 * x2**2
    m3x3 = (theta - g3)**2 * x3**2

    print("m1x1", m1x1)
    print("m2x2", m2x2)
    print("m3x3", m3x3)

    # dm1x1 = d/d_theta[mx1, x=x1]
    dm1x1 = x1**2 * (2 * (theta - g1) * (1 - dg1))
    dm2x2 = x2**2 * (2 * (theta - g2) * (1 - dg2))
    dm3x3 = x3**2 * (2 * (theta - g3) * (1 - dg3))

    print("dm1x1", dm1x1)
    print("dm2x2", dm2x2)
    print("dm3x3", dm3x3)

    # dLF = d/d_theta[(m1x1 - y1)^2 + .. + (m3x3 - y3)^2]
    dLF = 2 * (m1x1 - y1) * dm1x1 \
        + 2 * (m2x2 - y2) * dm2x2 \
        + 2 * (m3x3 - y3) * dm3x3

    return dLF