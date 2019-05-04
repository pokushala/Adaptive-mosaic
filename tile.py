import os
import warnings
import timeit
import random
import tempfile

import numpy as np
import skimage
from PIL import Image
from skimage import io, transform

from scipy import spatial

__all__ = ["Tiles", "load_image"]


def load_image(path: str) -> np.ndarray:
    im = Image.open(path)
    rgb_im = im if im.mode == "RGB" else im.convert("RGB")

    return np.array(rgb_im)


class Tiles:

    def __init__(self, tiles_dir: str, max_tile_size):
        self.tiles_directory = tiles_dir
        self.__max_size = max_tile_size
        self.__exts = (".jpg", ".png")
        self.__dump_dir = tempfile.TemporaryDirectory(dir=".")

        self.tiles = self.__load_tiles()

        print("Total tiles: {}".format(len(self.tiles)))
        self.mean_colors = self.__compute_mean_rgb_colors()

        print("Building KD-tree.")
        self.__kd_tree = spatial.KDTree(self.mean_colors)

    def __load_tiles(self):
        dirs = [self.tiles_directory]
        paths_to_load = []

        while dirs:
            cur_dir = dirs.pop()
            print("Scan: ", cur_dir)
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
                            warnings.warn("Skip: {}. It has not extension.".format(item.path))

        paths_to_load = random.choices(paths_to_load, k=400)

        print("Resizing tiles.")
        collections = io.ImageCollection(paths_to_load, conserve_memory=False, load_func=load_image)

        paths = []

        width, height = self.__max_size

        for i in range(len(collections)):
            fname = os.path.join(self.__dump_dir.name, f"{i + 1}.jpg")
            paths.append(fname)

            image = skimage.img_as_float32(collections[i])
            resized = transform.resize(image, (height, width), order=3)
            io.imsave(fname, resized, plugin="pil", quality=100)
            print("{:.2%}".format(i / len(collections)), end="\r")

        return io.ImageCollection(paths, conserve_memory=False, load_func=load_image)

    def __compute_mean_rgb_colors(self):
        mean_colors = []
        print("Computing mean rgb colors for tiles.")

        for i in range(len(self)):
            print("Progress: {:.2%}".format(i / len(self)), end="\r")
            mean_colors.append(self[i].mean(axis=(0, 1)))

        print("")
        print("Done")

        return np.array(mean_colors)

    def k_indices_nearest_tile(self, rgb_color: np.ndarray, k: int) -> np.ndarray:
        assert len(rgb_color) == 3
        # assert isinstance(rgb_color.dtype, np.float)
        assert k > 0

        _, indices = self.__kd_tree.query(rgb_color, k)
        return indices


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__dump_dir.cleanup()

    def __getitem__(self, item):
        return skimage.img_as_float32(self.tiles[item])

    def __len__(self):
        return len(self.tiles)
