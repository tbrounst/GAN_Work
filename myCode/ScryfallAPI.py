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
    urllib.request.urlretrieve(artCropUrl, "C:\\Users\\Tom\Desktop\\GANStuff\\WrapperFolder\\ScryfallArtCropCreatures\\{}.jpg".format(name))
    img = Image.open("C:\\Users\\Tom\\Desktop\\\GANStuff\\WrapperFolder\\ScryfallArtCropCreatures\\{}.jpg".format(name), 'r')
    flip = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    flip.save('C:\\Users\\Tom\\Desktop\\GANStuff\\WrapperFolder\\ScryfallArtCropCreatures\\{}Flip.jpg'.format(name))

more = True
page = 1
counter = 0

humonoidCreatures = "t:Advisor, Aetherborn, Ally, Artificer, Assassin, Azra, Barbarian, Berserker, Citizen, Cleric, " \
                    "Coward, Cyclops, Dauthi, Demigod, Demon, Deserter, Djinn, Druid, Dryad, Dwarf, Elf, Faerie, " \
                    "Flagbearer, Giant, Gnome, Goblin, God, Golem, Gorgon, Hag, Homarid, Human, Kithkin, Knight, " \
                    "Kobold, Kor, Mercenary, Merfolk, Minion, Minotaur, Monk, Moonfolk, Mystic, Naga, Ninja, Noble, " \
                    "Noggle, Nomad, Nymph, Ogre, Orc, Orgg, Ouphe, Peasant, Pilot, Pirate, Rebel, Rigger, Rogue, " \
                    "Samurai, Satyr, Scarecrow, Scout, Serf, Shaman, Siren, Skeleton, Slith, Soldier, Spellshaper, " \
                    "Survivor, Troll, Vampire, Vedalken, Viashino, Warlock, Warrior, Werewolf, Wizard, Wraith, Yeti, " \
                    "Zombie, Zubera"
query = humonoidCreatures.lower().replace(", ", " OR t:")
print(query)

while more:
    print(page)
    time.sleep(0.1)
    # cards = scrython.cards.Search(q="(t:shaman OR t:druid OR t:artificer OR t:cleric OR t:dwarf OR t:elf OR t:giant "
    #                                 "OR t:human OR t:kithkin OR t:knight OR t:kor OR t:soldier OR t:vampire OR "
    #                                 "t:warrior OR t:wizard OR t:zombie) (c:w OR c:g) -c:m", unique="art", page=page)
    cards = scrython.cards.Search(q="t:advisor OR t:aetherborn OR t:ally OR t:artificer OR t:assassin OR t:azra OR "
                                    "t:barbarian OR t:berserker OR t:citizen OR t:cleric OR t:coward OR t:cyclops OR "
                                    "t:dauthi OR t:demigod OR t:deserter OR t:djinn OR t:druid OR t:dryad "
                                    "OR t:dwarf OR t:elf OR t:faerie OR t:flagbearer OR t:giant OR t:gnome OR "
                                    "t:goblin OR t:god OR t:golem OR t:gorgon OR t:hag OR t:homarid OR t:human OR "
                                    "t:kithkin OR t:knight OR t:kobold OR t:kor OR t:mercenary OR t:merfolk OR "
                                    "t:minion OR t:minotaur OR t:monk OR t:moonfolk OR t:mystic OR t:naga OR t:ninja "
                                    "OR t:noble OR t:noggle OR t:nomad OR t:nymph OR t:ogre OR t:orc OR t:orgg OR "
                                    "t:ouphe OR t:peasant OR t:pilot OR t:pirate OR t:rebel OR t:rigger OR t:rogue OR "
                                    "t:samurai OR t:satyr OR t:scarecrow OR t:scout OR t:serf OR t:shaman OR t:siren "
                                    "OR t:skeleton OR t:slith OR t:soldier OR t:spellshaper OR t:survivor OR t:troll "
                                    "OR t:vampire OR t:vedalken OR t:viashino OR t:warlock OR t:warrior OR t:werewolf "
                                    "OR t:wizard OR t:wraith OR t:yeti OR t:zombie OR t:zubera", unique="art",
                                  page=page)

    for card in cards.data():
        # if page == 4:
        #     print(card)
        if card['layout'] == 'normal' or card['layout'] == 'meld' or card['layout'] == 'leveler':
            getImage(card, counter)
            # continue
        elif card['layout'] == 'flip' or card['layout'] == 'host' or card['layout'] == 'augment':
            continue
        else:
            print(card)
            for subCard in card['card_faces']:
                if subCard['type_line'] == 'Creature':
                    getImage(subCard, counter)
                    # continue
        counter += 1
    page += 1
    more = cards.has_more()

# print(cards.data())

# urllib.request.urlretrieve("https://c1.scryfall.com/file/scryfall-cards/art_crop/front/2/f/2f28ecdc-a4f0-4327-a78c-340be41555ee.jpg?1562139726", "C:\\Users\\Tom\\Desktop\\test.jpg")

# for ii in range(1000):
#     print(ii)
#     print(cards.data()[ii]['image_uris']['normal'])
