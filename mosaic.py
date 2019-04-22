import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage import io
from skimage import transform

from mosbuilder import MosaicBuilder, Box
from tile import Tiles

TILES = None
THRESHOLD = None
K_NEAR = None


def init(*args):
    global TILES
    global THRESHOLD
    global K_NEAR
    TILES = args[0]
    THRESHOLD = args[1]
    K_NEAR = args[2]


def process_part(image: np.ndarray, box: Box):
    builder = MosaicBuilder(THRESHOLD, K_NEAR, TILES, min_box_size=(5, 5), max_box_size=(25, 25))
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

    tiles = Tiles(args.tiles_dir)
    image_data = skimage.img_as_float(io.imread(args.image))

    boxes_temp = split_image_parts((0, 0) + image_data.shape[-2::-1])

    with Pool(initializer=init, initargs=(tiles, args.threshold, args.k_near_tile)) as workers:
        res = workers.starmap(process_part, zip([image_data] * 4, boxes_temp))

    boxes = []
    for item in res:
        boxes.extend(item)

    print("Build mosaic")

    mos = image_data.copy()

    for box_data in boxes:
        x, y, width, height = box_data["box"]
        index = box_data["tile_index"]
        resized = transform.resize(tiles[index], (height, width))
        mos[y: y + height, x: x + width] = resized

    alpha = 0.6
    res = alpha * mos + (1 - alpha) * image_data
    res = np.clip(res, 0, 1)
    plt.imshow(res)
    plt.show()


def create_out_path(path_to_img):
    name, ext = os.path.splitext(path_to_img)
    print(ext)
    return "{}_out{}".format(name, ext)


if __name__ == '__main__':
    arg_parsers = argparse.ArgumentParser()
    arg_parsers.add_argument("--image", "-i", type=str, help="An input image.", required=True)
    arg_parsers.add_argument("--tiles-dir", "-t", type=str, help="A directory with tiles.", required=True)
    arg_parsers.add_argument("--out", "-o", type=str, help="Path to output image.")
    arg_parsers.add_argument("--threshold", "-d", type=float, help="A threshold.", default=0.9)
    arg_parsers.add_argument("--max-tile-size", "-m", type=int, default=50, help="A maximum size of tile.")
    arg_parsers.add_argument("--k-near-tile", "-k", type=int, default=8)

    args = arg_parsers.parse_args()

    if not os.path.exists(args.image):
        print(f"'{args.image}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.tiles_dir):
        print(f"'A directory '{args.tiles_die}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.threshold <= 0:
        print("Threshold must be > 0", file=sys.stderr)
        sys.exit(1)

    if args.max_tile_size < 1:
        print("Max tile size must be > 1", file=sys.stderr)
        sys.exit(1)

    if args.k_near_tile < 1:
        print("K-nearest title must be > 0", file=sys.stderr)
        sys.exit(1)

    if args.out is None:
        args.out = create_out_path(args.image)

    mosaic(args)

