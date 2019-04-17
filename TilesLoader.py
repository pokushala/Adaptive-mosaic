import glob
from PIL import Image

from Tile import Tile

class TilesLoader:
    def __init__(self, tiles_directory):
        self.tiles_directory = tiles_directory
        self.tiles_count = 0


    def get_tiles(self):
        tile_paths = glob.glob(self.tiles_directory + '/*.jpg')
        print("Reading files from directory: ", self.tiles_directory)
        tiles = []
        for path in tile_paths:
            tile = Tile(path)
            if tile.tile:
                tiles.append(tile)
        
        self.tiles_count = len(tiles)
        print ('Loaded %s tiles.' % (self.tiles_count))

        return tiles