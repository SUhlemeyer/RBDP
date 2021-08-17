def energy_loss(beta, x):
    return (1 - beta) * x


def inv_energy_loss(beta, x):
    return x / (1 - beta)


def round_down(x,h):
    return int(x - (x % h))


def round_up(x,h):
    if x % h == 0:
        return int(x)
    else:
        return int(x - (x % h) + h)
