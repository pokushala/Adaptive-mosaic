from PIL import Image


class TargetLoader:
    def __init__(self, image_path):
        self.image_path = image_path

    def get_image(self):
        print('Processing main image...')
        img = Image.open(self.image_path)
        image_data = img.convert('RGB')
        print ('Main image processed.')

        return image_data