import numpy as np
import random
from copy import copy
from PIL import Image, ImageStat
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from itertools import combinations

from settings import K_NEAREST_TILES, PREC, OUT_FILE
from MosaicImage import MosaicImage

class MosaicBuilder:
    def __init__(self, image_data, tiles_data):
        self.image_data = image_data
        self.tiles_data = tiles_data
        self.mosaic_img = MosaicImage(image_data)

    # find k nearest tiles for a part and return one random
    def nearest_tile_index(self, part_data, k=K_NEAREST_TILES):
        # go through each tile in turn looking for the best match for the part of the image represented by 'img_data'
        part_mean_lab_clr = self.mean_lab_clr(part_data)
        clr_differences_between_part_and_tiles = np.array([self.color_distance_lab(tile.mean_lab_clr, part_mean_lab_clr) for tile in self.tiles_data])
        best_fit_tile_indexes = np.argpartition(clr_differences_between_part_and_tiles, k)[:k]
        tile_index = random.choice(best_fit_tile_indexes)
        return tile_index

    # return mean lab color for img
    def mean_lab_clr(self, img):
        mean_rgb = np.array(ImageStat.Stat(img).mean)/256
        return convert_color(sRGBColor(*mean_rgb), LabColor)
    
    # calculate color difference using CIE2000 standart
    def color_distance_rgb(self, clr1_rgb, clr2_rgb):
        # Convert from RGB to Lab Color Space
        clr1_lab = convert_color(clr1_rgb, LabColor)
        clr2_lab = convert_color(clr2_rgb, LabColor)
        # Find the color difference
        delta_e = delta_e_cie2000(clr1_lab, clr2_lab)
        return delta_e

    def color_distance_lab(self, clr1_lab, clr2_lab):
        # Find the color difference
        delta_e = delta_e_cie2000(clr1_lab, clr2_lab)
        return delta_e

    # division function
    def adaptive_image_partition(self):
        pocket_with_quarters = [(self.image_data, self.image_data.getbbox())]#self.quarter_image(self.image_data, np.array([0, 0, w, h]))
        current_img = copy(pocket_with_quarters)[0]
        result_slices = []
        
        while pocket_with_quarters:
            current_quarters = self.quarter_image(current_img[0], current_img[1])
            if (self.continue_division(current_quarters)):
                pocket_with_quarters.pop(0) #текущее изображение поделено
                pocket_with_quarters += current_quarters # его части добавили в карман для дальнейшего деления 
            else:
                for quarter in current_quarters:  
                    tile_to_paste = self.tiles_data[self.nearest_tile_index(quarter[0])]
                    self.mosaic_img.add_tile(tile_to_paste, quarter[1]) # иначе сразу вставляем четверти в мозайку
                result_slices += current_quarters
            current_img = pocket_with_quarters.pop(0)   
        self.mosaic_img.save(OUT_FILE)
        print ('\nFinished, output is in', OUT_FILE)
        return result_slices
        

    # split one image into 4 parts
    def quarter_image(self, src_img, relative_box):
        w, h = src_img.size
        src_box = [0, 0, w, h]
        delta_w = w//2
        delta_h = h//2
        left_top_shift = np.array([0, 0, -delta_w, -delta_h]) #(0, 0, delta_w, delta_h)
        right_top_shift = np.array([delta_w, 0, 0, -delta_h]) #(delta_w, 0, w, delta_h)
        left_bottom_shift = np.array([0, delta_h, -delta_w, 0]) #(0, delta_h, delta_w, h)
        right_bottom_shift = np.array([delta_w, delta_h, 0, 0])#(delta_w, delta_h, w, h)

        #print("img ", src_img)
        #print("src box", src_box)
        #print("left shift", left_top_shift)

        left_top = src_img.crop(src_box + left_top_shift)
        right_top = src_img.crop(src_box + right_top_shift)
        left_bottom = src_img.crop(src_box + left_bottom_shift)
        right_bottom = src_img.crop(src_box + right_bottom_shift)

        return [(left_top, relative_box + left_top_shift),
                (right_top, relative_box + right_top_shift),
                (left_bottom, relative_box + left_bottom_shift),
                (right_bottom, relative_box + right_bottom_shift)] 

    # check stop division condition
    def continue_division(self, quarters, prec=PREC):
        color_distances = []
        quarter_pairs = list(combinations(quarters, 2))
        for q_pair in quarter_pairs:
            color_distances.append(self.color_distance_lab(self.mean_lab_clr(q_pair[0][0]), self.mean_lab_clr(q_pair[1][0])))
        mean_distance = np.mean(color_distances)
        return mean_distance > prec

    

    

    


        