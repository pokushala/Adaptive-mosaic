from PIL import Image, ImageStat
import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

class Tile:
    def __init__(self, tile_path):
        self.tile_path = tile_path
        self.tile = self.__load_tile()
        self.mean_rgb_clr = self.__mean_rgb()
        self.mean_lab_clr = self.__mean_lab()

    def __load_tile(self):
        try:
            tile = Image.open(self.tile_path)		
            return tile.convert('RGB')
        except:
            return None

    def __mean_rgb(self):
        return np.array(ImageStat.Stat(self.tile).mean)/256
    
    def __mean_lab(self):
        return convert_color(sRGBColor(*self.mean_rgb_clr), LabColor)

    def scale_tile(self, new_w, new_h):
        pass