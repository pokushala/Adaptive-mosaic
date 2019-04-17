import sys

from TilesLoader import TilesLoader
from TargetLoader import TargetLoader
from MosaicBuilder import MosaicBuilder


def mosaic(img_path, tiles_path):
    tiles_data = TilesLoader(tiles_path).get_tiles()
    image_data = TargetLoader(img_path).get_image()
    builder = MosaicBuilder(image_data, tiles_data)
    builder.adaptive_image_partition()
    #builder.compose(image_data, tiles_data)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print ('Usage: %s <image> <tiles directory>\r' % (sys.argv[0],))
    else:
        mosaic(sys.argv[1], sys.argv[2])

