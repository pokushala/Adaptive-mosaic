from PIL import Image

class MosaicImage:
	def __init__(self, target_img):
		self.image = Image.new(target_img.mode, target_img.size)

	def add_tile(self, tile_data, coords):
		#img = Image.new('RGB', (coords[2]-coords[0], coords[3]-coords[1]))
		#img.putdata(tile_data.load())
		print("coords", coords)
		print("size", coords[2]-coords[0], coords[3]-coords[1])
		tile_to_paste = tile_data.resize((coords[2]-coords[0], coords[3]-coords[1]))
		self.image.paste(tile_to_paste, coords) #(coords[0], coords[1], coords[3], coords[3])

	def save(self, path):
		self.image.save(path)