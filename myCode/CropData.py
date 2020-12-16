import os

from PIL import Image

box = (37, 76, 450, 377)

inputPath = 'startOfPath\\GANStuff\\Lands'
outputPath = 'startOfPath\\GANStuff\\TrainGAN\\GANInput'

for counter, filename in enumerate(os.listdir(inputPath)):
    print(filename)
    img = Image.open(inputPath + '\\' + filename, 'r')
    crop = img.crop(box)
    crop.save('{}\\{}.jpg'.format(outputPath, counter))
    flip = crop.transpose(method=Image.FLIP_LEFT_RIGHT)
    flip.save('{}\\{}Flip.jpg'.format(outputPath, counter))
