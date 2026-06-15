def get_weights(mode):
    if mode == "green":
        return [0.6, 0.2, 0.2]
    elif mode == "performance":
        return [0.2, 0.6, 0.2]
    else:
        return [0.33, 0.33, 0.33]
