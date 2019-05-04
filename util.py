import numpy as np


def rgb2hsi(rgb: np.ndarray) -> np.ndarray:
    """Conversion RGB to HSI

        h (hue) in [0; 2 pi], s (saturation) in [0; 1], i (intensity) in [0; 1].
        If 'rgb' color is black, white or grayscale then h component is undefined (np.nan).

        Description:
            http://omar.alkadi.net/wp-content/uploads/DIPch6.pdf (22 p.)

    :param rgb: RGB color. Color components must be in [0; 1].
    :return: Color in HSI color space.
    """
    assert rgb.ndim == 3 and rgb.shape[-1] == 3, "'rgb' has incomplete shape." \
                                                         "Expected (..., ..., 3) got {}.".format(rgb.shape)
    assert ((rgb >= 0) & (rgb <= 1)).all(), "Color components must be in [0; 1]."

    i = rgb.mean(axis=2)
    s = 1 - 1 / i * rgb.min(axis=2)
    rg_diff = rgb[..., 0] - rgb[..., 1]
    rb_diff = rgb[..., 0] - rgb[..., 2]
    gb_diff = rgb[..., 1] - rgb[..., 2]

    h = np.arccos(0.5 * (rg_diff + rb_diff) / np.sqrt(rg_diff ** 2 + rb_diff * gb_diff))
    cond = rgb[..., 2] > rgb[..., 1]
    h[cond] = 2 * np.pi - h[cond]

    h = np.clip(h, 0, 2 * np.pi)
    s = np.clip(s, 0, 1)
    i = np.clip(i, 0, 1)

    return np.dstack((h, s, i))


def hsi2rgb(hsi: np.ndarray) -> np.ndarray:
    """Conversion HSI to RGB

        h (hue) in [0; 2 pi], s (saturation) in [0; 1], i (intensity) in [0; 1].
        If h is np.nan then r (red) = g (green) = b (blue) = i (intensity).

        Description:
            http://omar.alkadi.net/wp-content/uploads/DIPch6.pdf (22 p.)

    :param hsi: HSI color.
    :return: Color in RGB color space.
    """
    assert hsi.ndim == 3 and hsi.shape[-1] == 3, "'hsi' has incomplete shape." \
                                                 "Expected (..., ..., 3) got {}".format(hsi.shape)

    h = hsi[..., 0]
    s = hsi[..., 1]
    i = hsi[..., 2]
    x = i * (1 - hsi[..., 1])

    cond1 = h < np.pi * 2 / 3
    y = np.zeros_like(h)
    z = np.zeros_like(h)

    y[cond1] = i[cond1] * (1 + s[cond1] * np.cos(h[cond1]) / np.cos(np.pi / 3 - h[cond1]))
    z[cond1] = 3 * i[cond1] - (x[cond1] + y[cond1])

    rest = h >= 4 * np.pi / 3

    h[rest] -= 4 * np.pi / 3
    y[rest] = i[rest] * (1 + s[rest] * np.cos(h[rest]) / np.cos(np.pi / 3 - h[rest]))
    z[rest] = 3 * i[rest] - (x[rest] + y[rest])

    cond2 = ~(cond1 | rest)
    h[cond2] -= 2 * np.pi / 3
    y[cond2] = i[cond2] * (1 + s[cond2] * np.cos(h[cond2]) / np.cos(np.pi / 3 - h[cond2]))
    z[cond2] = 3 * i[cond2] - (x[cond2] + y[cond2])

    res = np.dstack((x, y, z))

    # y z x
    res[cond1] = np.roll(res[cond1], shift=-1, axis=-1)

    # z x y
    res[rest] = np.roll(res[rest], shift=1, axis=-1)

    # Fill NaN value of intensity value
    where_nan = np.isnan(h)

    res[where_nan] = i[where_nan].reshape(-1, 1)

    return np.clip(res, 0, 1)




if __name__ == "__main__":
    colors = np.linspace((0, 0, 0), (1, 1, 1), num=250).reshape(-1, 1, 3)
    #colors = np.random.sample((10000, 100, 3))

    # print(colors)

    hsi = rgb2hsi(colors)

    print(hsi)

    rec_rgb = hsi2rgb(hsi)

    # print(rec_rgb)

    if not np.allclose(colors, rec_rgb):
        print("Error")
        print("!" * 10)



