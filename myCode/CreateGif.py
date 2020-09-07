import os
import re

import imageio

inputPath = 'C:\\Users\\Tom\\PycharmProjects\\pythonProject\\data\\images\\DConv-GConv-GANART\\MeV3'
outputPath = 'C:\\Users\\Tom\\Desktop\\movie2.gif'

files = os.listdir(inputPath)

filesICareAbout = [file for file in files if 'hori' not in file]
FILE_REGEX = re.compile(r"_epoch_(\d+)_batch_0.png")
# print(filesICareAbout)
filesICareAbout.sort(key=lambda x: int(FILE_REGEX.match(x).group(1)))
# print(filesICareAbout)

images = [imageio.imread(os.path.join(inputPath, file)) for file in filesICareAbout]
imageio.mimsave(outputPath, images)
