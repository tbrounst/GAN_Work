import os

from PIL import Image

box = (37, 76, 450, 377)

inputPath = 'C:\\Users\\Tom\\Desktop\\DisembodiedHands'
outputPath = 'C:\\Users\\Tom\\Desktop\\TrainGAN\\Output'

for counter, filename in enumerate(os.listdir(inputPath)):
    print(filename)
    img = Image.open(inputPath + '\\' + filename, 'r')
    crop = img.crop(box)
    crop.save(outputPath + '\\{}.jpg'.format(counter))
