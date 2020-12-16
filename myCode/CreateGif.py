import os
import re

import imageio

inputPath = 'startOfPath\\PycharmProjects\\pythonProject\\myCode\\data\\images\\DConv-GConv-GANART\\Lands'
outputPath = 'startOfPath\\Desktop\\movie4.gif'

files = os.listdir(inputPath)

filesICareAbout = [file for file in files if 'hori' not in file]
FILE_REGEX = re.compile(r"_epoch_(\d+)_batch_")
# print(filesICareAbout)
filesICareAbout.sort(key=lambda x: int(FILE_REGEX.match(x).group(1)))
# print(filesICareAbout)

images = [imageio.imread(os.path.join(inputPath, file)) for file in filesICareAbout]
imageio.mimsave(outputPath, images)
