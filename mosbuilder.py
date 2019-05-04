import random
import os
from collections import defaultdict
from typing import Tuple, List, Dict

import numpy as np
from skimage import color
from scipy import linalg

import tile

__all__ = ["Box", "MosaicBuilder"]

Box = Tuple[int]


class MosaicBuilder:
    def __init__(self, threshold: float, k_near_title: int, tiles_data: tile.Tiles
                 , min_box_size: Tuple[int] = (5, 5), max_box_size: Tuple[int] = (75, 75)):
        self.tiles_data = tiles_data
        self._threshold = threshold
        self._k_near = min(k_near_title, len(tiles_data))
        self.__min_size = min_box_size
        self.__max_size = max_box_size
        self.__image = None

    # find k nearest tiles for a part and return one random
    def nearest_tile_index(self, box: Box):
        # go through each tile in turn looking for the best match for the part of the image represented by 'img_data'
        part_mean_lab_clr = self.mean_rgb_clr(box)
        best_fit_tile_indices = self.tiles_data.k_indices_nearest_tile(part_mean_lab_clr, self._k_near)
        tile_index = random.choice(best_fit_tile_indices)
        return tile_index

    # return mean lab color for img
    def mean_rgb_clr(self, box: Box, keepdims=False) -> np.ndarray:
        x, y, width, height = box
        mean_rgb = self.__image[y: y + height, x: x + width].mean(axis=(0, 1), keepdims=keepdims)
        return mean_rgb

    def color_distance_rgb(self, clr1_rgb: np.ndarray, clr2_rgb: np.ndarray) -> float:
        return linalg.norm(clr1_rgb - clr2_rgb)

    def color_distance_cie2000(self, rgb1: np.ndarray, rgb2: np.ndarray) -> float:
        lab1 = color.rgb2lab(rgb1)
        lab2 = color.rgb2lab(rgb2)

        return color.deltaE_ciede2000(lab1, lab2)

    # division function
    def adaptive_image_partition(self, image: np.ndarray, init_box: Box = None) -> Dict[int, List[Box]]:
        if init_box is not None:
            assert len(init_box) == 4, "The length of init_box must be equal to 4."
            x0, y0, in_width, in_height = init_box
            im_width, im_height = image.shape[-2::-1]
            assert in_width > 0, "The init width must be > 0."
            assert in_height > 0, "The init width must be > 0."
            assert 0 <= x0 < im_width, "The init box is outside of image boundary."
            assert 0 <= y0 < im_height, "The init box is outside of image boundary."
            assert 0 <= x0 + in_width <= im_width, "The init box is outside of image boundary."
            assert 0 <= y0 + in_height <= im_height, "The init box is outside of image boundary."

        pocket_with_quarters = [(0, 0) + image.shape[-2::-1]] if init_box is None else [init_box]
        result_slices = defaultdict(list)

        self.__image = image

        progress = 0
        total = pocket_with_quarters[0][-2] * pocket_with_quarters[0][-1]

        print_every = 10
        progress_threshold = total // print_every
        progress_threshold = 1 if progress_threshold == 0 else progress_threshold

        while pocket_with_quarters:
            current_box = pocket_with_quarters.pop()

            boxes = self.quarter_image(current_box)

            if self.continue_division(boxes):
                pocket_with_quarters += boxes  # его части добавили в карман для дальнейшего деления
            else:
                for box in boxes:
                    *_, width, height = box
                    progress += width * height
                    result_slices[self.nearest_tile_index(box)].append(box)

                if progress % progress_threshold == 0:
                    print(f"PID: {os.getpid()} Progress: {progress / total:.2%}")

        print("Boxes computed")

        return result_slices

    # split one image into 4 parts
    def quarter_image(self, box: Box):
        x0, y0, width, height = box
        new_width = width // 2
        new_height = height // 2

        if new_width < self.__min_size[0] or new_height < self.__min_size[1]:
            return (box, )
        else:
            return (
                    (x0, y0, new_width, new_height),
                    (x0 + new_width, y0, new_width + width % 2, new_height),
                    (x0, y0 + new_height, new_width, new_height + height % 2),
                    (x0 + new_width, y0 + new_height, new_width + width % 2, new_height + height % 2)
                    )

    # check stop division condition
    def continue_division(self, quarters: Tuple[Box]):
        if len(quarters) == 1:
            return False
        else:
            fisrt_box = quarters[0]
            *_, width, height = fisrt_box

            if width > self.__max_size[0] and height > self.__max_size[1]:
                return True
            else:
                rgb_colors = tuple(self.mean_rgb_clr(box, keepdims=True) for box in quarters)

                mean_color = 0
                total = len(rgb_colors) * (len(rgb_colors) + 1) // 2

                for i in range(len(quarters)):
                    for j in range(i + 1, len(quarters)):
                        mean_color += self.color_distance_cie2000(rgb_colors[i], rgb_colors[j])
                return mean_color / total > self._threshold
