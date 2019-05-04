import argparse
import os
import sys
from multiprocessing import Pool
from typing import Dict, Tuple
import itertools
import warnings
from collections import defaultdict

import numpy as np
import skimage
from skimage import io
from skimage import transform
from PIL import Image, ImageDraw, ImageFont

import util

from mosbuilder import MosaicBuilder, Box
from tile import Tiles, load_image


def process_part(image: np.ndarray, box: Box, threshold: float, k_near: int, tiles: Tiles
                 , min_sizes: Tuple[int], max_sizes: Tuple[int]) -> Dict[int, Box]:
    builder = MosaicBuilder(threshold, k_near, tiles, min_box_size=min_sizes, max_box_size=max_sizes)
    return builder.adaptive_image_partition(image, box)


def mosaic(args):

    def split_image_parts(box: Box):
        x0, y0, width, height = box
        new_width = width // 2
        new_height = height // 2

        return (
            (x0, y0, new_width, new_height),
            (x0 + new_width, y0, new_width + width % 2, new_height),
            (x0, y0 + new_height, new_width, new_height + height % 2),
            (x0 + new_width, y0 + new_height, new_width + width % 2, new_height + height % 2)
        )

    with Tiles(args.tiles_dir, args.max_tile_size) as tiles:

        image_data = skimage.img_as_float32(load_image(args.image))

        boxes_temp = split_image_parts((0, 0) + image_data.shape[-2::-1])

        with Pool() as workers:
            res = workers.starmap(process_part, zip(itertools.repeat(image_data, len(boxes_temp))
                                                    , boxes_temp
                                                    , itertools.repeat(args.threshold, len(boxes_temp))
                                                    , itertools.repeat(args.k_near_tile, len(boxes_temp))
                                                    , itertools.repeat(tiles, len(boxes_temp))
                                                    , itertools.repeat(tuple(args.min_tile_size), len(boxes_temp))
                                                    , itertools.repeat(tuple(args.max_tile_size), len(boxes_temp))
                                                    )
                                )


    boxes = defaultdict(list)

    for item in res:
        for tile_index in item:
            boxes[tile_index] += item[tile_index]

    print("Build mosaic")

    mos = image_data.copy()
    hsi_image = util.rgb2hsi(image_data)

    i = 0
    total = sum(len(boxes[key]) for key in boxes)

    orig_image_intensity = 0.5

    for tile_index in boxes:
        tile = tiles[tile_index]

        for box in boxes[tile_index]:
            x, y, width, height = box
            resized = transform.resize(tile, (height, width))
            hsi_tile = util.rgb2hsi(resized)
            # Последний канал это интенсивность
            hsi_tile[..., -1] = orig_image_intensity * hsi_image[y: y + height, x: x + width, -1] \
                                + (1 - orig_image_intensity) * hsi_tile[..., -1]
            mos[y: y + height, x: x + width] = util.hsi2rgb(hsi_tile)

            i += 1
            print("Progress: {:.2%}".format(i / total), end="\r")

    alpha = args.alpha
    res = alpha * mos + (1 - alpha) * image_data
    res = np.clip(res, 0, 1)
    res = np.pad(res, ((0, 0), (0, 300), (0, 0)), mode="constant", constant_values=1)

    res = Image.fromarray(skimage.img_as_ubyte(res))

    draw_params(res, image_data.shape[1], args)

    res.save(args.out, quality=100)

    # io.imsave(args.out, res, plugin="pil", quality=100)
    print("A result saved to '{}'".format(args.out))

def draw_params(image, orig_width, args):
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except IOError as exc:
        warnings.warn("Could not load 'arial.ttf'", RuntimeWarning)

        try:
            font = ImageFont.truetype("liberationsans-regular.ttf", size=16)
        except IOError as exc:
            warnings.warn("Could not load 'liberationsans-regular.ttf'.\nCould not draw text.", RuntimeWarning)
            return

    black_color = (0, 0, 0)
    d = ImageDraw.Draw(image)
    texts = [f"Порог: {args.threshold}"
             , f"Максимальный размер: {' '.join(map(str, args.max_tile_size))}"
             , f"Минимальный размер: {' '.join(map(str, args.min_tile_size))}"
             , f"Коэффициент смешивания: {args.alpha}"
             ]

    origin = 10

    for text in texts:
        d.text((orig_width + 5, origin), text, font=font, fill=black_color)
        _, height = font.getsize(text)
        origin += height + 5


def create_out_path(path_to_img):
    name, ext = os.path.splitext(path_to_img)
    return "{}_out{}".format(name, ext)


def check_args(args):
    if not os.path.exists(args.image):
        print(f"'{args.image}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.tiles_dir):
        print(f"'A directory '{args.tiles_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.threshold <= 0:
        print("Threshold must be > 0", file=sys.stderr)
        sys.exit(1)

    if args.k_near_tile < 1:
        print("K-nearest title must be > 0", file=sys.stderr)
        sys.exit(1)

    if args.alpha < 0 or args.alpha > 1:
        print("Alpha must be in [0; 1]", file=sys.stderr)
        sys.exit(1)

    sizes = []

    for size in args.min_tile_size:
        try:
            sizes.append(int(size))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)

    args.min_tile_size = sizes.copy()

    sizes.clear()

    for size in args.max_tile_size:
        try:
            sizes.append(int(size))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)

    args.max_tile_size = sizes

    min_width, min_height = args.min_tile_size
    max_width, max_height = args.max_tile_size

    if max_width < min_width:
        print("A minimum width of tile must be less than a maximum width. Min width is {}, max width is {}"
              .format(min_width, max_width), file=sys.stderr)
        sys.exit(1)

    if min_width == max_width:
        warnings.warn("A minimum width of tile is equal to maximum width of tile.")

    if max_height < min_height:
        print("A minimum height of tile must be less than a maximum height. Min height is {}, max height is {}"
              .format(min_height, max_height), file=sys.stderr)
        sys.exit(1)

    if min_height == max_height:
        warnings.warn("A minimum height of tile is equal to maximum height of tile.")

    if args.out is None:
         args.out = create_out_path(args.image)


if __name__ == '__main__':
    arg_parsers = argparse.ArgumentParser()
    arg_parsers.add_argument("--image", "-i", type=str, help="A path to input image.", required=True)
    arg_parsers.add_argument("--tiles-dir", "-t", type=str, help="A directory with tiles.", required=True)
    arg_parsers.add_argument("--out", "-o", type=str, help="A path to output image.")
    arg_parsers.add_argument("--threshold", "-d", type=float, help="A threshold. It must be  > 0.", default=0.9)
    arg_parsers.add_argument("--max-tile-size", nargs=2, default=["75", "75"], help="A maximum size of tile on mosaic.")
    arg_parsers.add_argument("--min-tile-size", nargs=2, default=["10", "10"], help="A minimum size of tile on mosaic.")
    arg_parsers.add_argument("--k-near-tile", "-k", type=int, default=8)
    arg_parsers.add_argument("--alpha", "-a", type=float, default=0.7
                             , help="A coefficient of blending original image and mosaic. Must be in [0; 1].")

    args = arg_parsers.parse_args()

    check_args(args)

    mosaic(args)

