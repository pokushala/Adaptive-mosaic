import os
import random

import numpy as np
import skimage
from PIL import Image
from skimage import color
from skimage import io


def _load_func(path):
    im = Image.open(path)
    rgb_im = im if im.mode == "RGB" else im.convert("RGB")

    return np.array(rgb_im)


class Tiles:

    def __init__(self, tiles_dir: str):
        self.tiles_directory = tiles_dir
        self.__exts = (".jpg", ".png")
        self.tiles = self.__load_tiles()
        self.mean_colors = self.__compute_mean_lab_colors()

    def __load_tiles(self):
        dirs = [self.tiles_directory]
        paths_to_load = []

        while dirs:
            cur_dir = dirs.pop()
            print(cur_dir)
            with os.scandir(cur_dir) as dir_it:
                for item in dir_it:
                    if item.is_dir():
                        dirs.append(item.path)
                    elif item.is_file():
                        try:
                            _, ext = os.path.splitext(item.name)
                            if ext in self.__exts:
                                paths_to_load.append(item.path)
                        except ValueError:
                            print(f"Skip: {item.path}")
        paths_to_load = random.sample(paths_to_load, 200)

        return io.ImageCollection(paths_to_load, conserve_memory=False, load_func=_load_func)

    def __compute_mean_lab_colors(self):
        mean_colors = []
        print("Compute mean lab colors for tiles.")
        for i in range(len(self)):
            print("Progress: {:.2%}".format(i / len(self)), end="\r")
            mean = self[i].mean(axis=(0, 1), keepdims=True)
            mean_colors.append(color.rgb2lab(mean))

        print("")
        print("Done")

        return np.array(mean_colors)

    def __getitem__(self, item):
        # assert isinstance(item, int)
        return skimage.img_as_float(self.tiles[item])

    def __len__(self):
        return len(self.tiles)
