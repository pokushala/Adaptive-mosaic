from PIL import Image

class MosaicImage:
	def __init__(self, target_img):
		self.image = Image.new(target_img.mode, target_img.size)

	def add_tile(self, tile_data, coords):
		#img = Image.new('RGB', (coords[2]-coords[0], coords[3]-coords[1]))
		#img.putdata(tile_data.load())
		self.image.paste(tile_data, coords)

	def save(self, path):
		self.image.save(path)