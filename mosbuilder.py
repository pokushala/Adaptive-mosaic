import random
import os
from typing import Tuple, List

import numpy as np
from skimage import color

import tile

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
        part_mean_lab_clr = self.mean_lab_clr(box)
        clr_differences_between_part_and_tiles = np.array(tuple(self.color_distance_lab(self.tiles_data.mean_colors[i], part_mean_lab_clr)
                                                           for i in range(len(self.tiles_data))))
        best_fit_tile_indexes = np.argpartition(clr_differences_between_part_and_tiles, self._k_near, axis=None)[:self._k_near]
        tile_index = random.choice(best_fit_tile_indexes)
        return tile_index

    # return mean lab color for img
    def mean_lab_clr(self, box: Box) -> np.ndarray:
        x, y, width, height = box
        mean_rgb = self.__image[y: y + height, x: x + width].mean(axis=(0, 1), keepdims=True)
        return color.rgb2lab(mean_rgb)

    # calculate color difference using CIE2000 standart
    def color_distance_rgb(self, clr1_rgb: np.ndarray, clr2_rgb: np.ndarray) -> float:
        # Convert from RGB to Lab Color Space
        clr1_lab = color.rgb2lab(clr1_rgb)
        clr2_lab = color.rgb2lab(clr2_rgb)
        # Find the color difference
        delta_e = color.deltaE_ciede2000(clr1_lab, clr2_lab)
        return delta_e

    def color_distance_lab(self, clr1_lab: np.ndarray, clr2_lab: np.ndarray) -> float:
        # Find the color difference
        delta_e = color.deltaE_ciede2000(clr1_lab, clr2_lab)
        return delta_e

    # division function
    def adaptive_image_partition(self, image: np.ndarray, init_box: Box = None) -> List[Box]:
        if init_box is not None:
            assert len(init_box) == 4, "The length of init_box must be equal to 4."
            x0, y0, in_width, in_height = init_box
            im_width, im_height = image.shape[-2::-1]
            assert in_width <= 0, "The init width must be > 0."
            assert in_height <= 0, "The init width must be > 0."
            assert 0 <= x0 < im_width, "The init box is outside of image boundary."
            assert 0 <= y0 < im_height, "The init box is outside of image boundary."
            assert 0 <= x0 + in_width < im_width, "The init box is outside of image boundary."
            assert 0 <= y0 + in_height < im_height, "The init box is outside of image boundary."

        pocket_with_quarters = [(0, 0) + image.shape[-2::-1]] if init_box is None else [init_box]
        result_slices = []

        self.__image = image

        progres = 0
        total = pocket_with_quarters[0][-2] * pocket_with_quarters[0][-1]

        while pocket_with_quarters:
            current_box = pocket_with_quarters.pop()

            boxes = self.quarter_image(current_box)

            if self.continue_division(boxes):
                pocket_with_quarters += boxes  # его части добавили в карман для дальнейшего деления
            else:
                for box in boxes:
                    *_, width, height = box
                    progres += width * height
                print(f"PID: {os.getpid()} Progress: {progres / total:.2%}")
                result_slices.extend({"tile_index": self.nearest_tile_index(box), "box": box} for box in boxes)

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
                lab_colors = tuple(self.mean_lab_clr(box) for box in quarters)

                mean_color = 0
                total = len(lab_colors) * (len(lab_colors) + 1) // 2

                for i in range(len(quarters)):
                    for j in range(i + 1, len(quarters)):
                        mean_color += self.color_distance_lab(lab_colors[i], lab_colors[j])
                return mean_color / total > self._threshold
