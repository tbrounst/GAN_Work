import time

import scrython
import urllib.request
from PIL import Image

# https://api.scryfall.com/cards/xln/96?format=image&version=art_crop
# https://api.scryfall.com/cards/search?q=t:land+unique:art?format=csv

def getImage(card, counter):
    time.sleep(0.1)
    artCropUrl = card['image_uris']['art_crop']
    name = str(counter)
    urllib.request.urlretrieve(artCropUrl, "startOfPath\\GANStuff\\WrapperFolder\\ScryfallArtCropCreatures\\{}.jpg".format(name))
    img = Image.open("startOfPath\\GANStuff\\WrapperFolder\\ScryfallArtCropCreatures\\{}.jpg".format(name), 'r')
    flip = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    flip.save('startOfPath\\GANStuff\\WrapperFolder\\ScryfallArtCropCreatures\\{}Flip.jpg'.format(name))

more = True
page = 1
counter = 0

while more:
    print(page)
    time.sleep(0.1)
    cards = scrython.cards.Search(q="(t:shaman OR t:druid OR t:artificer OR t:cleric OR t:dwarf OR t:elf OR t:giant "
                                    "OR t:human OR t:kithkin OR t:knight OR t:kor OR t:soldier OR t:vampire OR "
                                    "t:warrior OR t:wizard OR t:zombie) (c:w OR c:g) -c:m", unique="art", page=page)
    for card in cards.data():
        time.sleep(0.1)
        # if page == 4:
        #     print(card)
        if card['layout'] == 'normal' or card['layout'] == 'meld' or card['layout'] == 'leveler':
            getImage(card, counter)
        elif card['layout'] == 'flip':
            continue
        else:
            for subCard in card['card_faces']:
                if subCard['type_line'] == 'Creature':
                    getImage(subCard, counter)
        counter += 1
    page += 1
    more = cards.has_more()

# print(cards.data())

# urllib.request.urlretrieve("https://c1.scryfall.com/file/scryfall-cards/art_crop/front/2/f/2f28ecdc-a4f0-4327-a78c-340be41555ee.jpg?1562139726", "C:\\Users\\Tom\\Desktop\\test.jpg")

# for ii in range(1000):
#     print(ii)
#     print(cards.data()[ii]['image_uris']['normal'])
