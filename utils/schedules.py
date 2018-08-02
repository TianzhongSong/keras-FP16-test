from .callbacks import Step


def onetenth_4_8_12(lr):
    steps = [4, 8, 12]
    lrs = [lr, lr / 10, lr / 100, lr / 1000]
    return Step(steps, lrs)


def onetenth_10_15_20(lr):
    steps = [10, 15, 15]
    lrs = [lr, lr / 10, lr / 100, lr / 1000]
    return Step(steps, lrs)


def onetenth_50_75(lr):
    steps = [50, 75]
    lrs = [lr, lr / 10, lr / 100]
    return Step(steps, lrs)


def wideresnet_step(lr):
    steps = [60, 120, 160]
    lrs = [lr, lr / 5, lr / 25, lr / 125]
    return Step(steps, lrs)
