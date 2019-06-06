import random
import time
import numpy as np
import itertools

# https://en.wikipedia.org/wiki/Set_(card_game)
# I have created a script that generated the deck of cards,
# shuffles the deck and plays the game. Your task is to write the fastest
# possible set finder. I have added a performance counter.

# The properties
color = ["red  ", "green", "blue "]
shape = ["block ", "circle", "wave  "]
fill = ["open ", "close", "dot  "]
number = [1, 2, 3]

# Create all cards
deck = []
for c in color:
    for s in shape:
        for f in fill:
            for n in number:
                deck.append([c, s, f, n])

# Check that the deck has 81 cards.
# print("The deck has {:} cards.".format(len(deck)))

# Shuffle the deck. Fix the random seed to ensure that everybody
# has the same deck.
random.seed(0)
random.shuffle(deck)

# Print the shuffled cards.
# for card in deck:
#     print(card)

# Draw the 12 initial cards
cards_on_table = deck[0:12]
deck_position = 12

# Start the game.
end_game = False
sets_found = 0
exec_times = []

### FUNCTIONS TO IMPLEMENT ###
def process_cards(cards):
    cards_code = []
    for i in range(len(cards)):
        color_code  = color .index(cards[i][0])
        shape_code  = shape .index(cards[i][1])
        fill_code   = fill  .index(cards[i][2])
        number_code = number.index(cards[i][3])
        cards_code.append((color_code, shape_code, fill_code, number_code))
    return cards_code

def is_set(card0, card1, card2):
    for i in range(4):
        if not ( (card0[i] == card1[i] == card2[i]) or ( (card0[i] != card1[i]) and (card1[i] != card2[i]) and (card0[i] != card2[i]) ) ):
            return False
    return True

def find_set(cards):
    t_start = time.process_time()

    cards_code = process_cards(cards)

    # Sort the list increase set likelihood in adjacent cards.
    cards_code.sort()

    set_found_code = []

    for three_cards in itertools.combinations(cards_code, 3):
        if is_set(*three_cards):
            set_found_code = three_cards
            break

    set_found = []
    for card in set_found_code:
        set_found.append([color[card[0]], shape[card[1]], fill[card[2]], number[card[3]]])

    t_end = time.process_time()
    return set_found, t_end-t_start
### END FUNCTIONS TO IMPLEMENT ###

for i in range(len(deck)//3):
    print("###### ROUND {0} ######".format(i+1))
    # Print the cards on the table.
    print("Cards on the table = {0}, cards left = {1}".format(len(cards_on_table), len(deck)-deck_position))
    print()
    for card in cards_on_table:
        print(card)
    print()
    
    # Find the set.
    set_cards, exec_time = find_set(cards_on_table)
    exec_times.append(exec_time)
   
    # If set is found, remove cards.
    if set_cards != []:
        print("The found set is: {:}".format(set_cards))
        sets_found += 1

        # Remove the set from the cards on the table.
        for card in set_cards:
            cards_on_table.remove(card)

        # Only add new cards if less than 12 and sufficient left.
        if len(cards_on_table) <= 12:
            if (deck_position + 3) <= len(deck):
                cards_on_table.extend(deck[deck_position:deck_position+3])
                deck_position += 3

    # If set not found, add three cards. If not possible, end the game.
    else:
        if (deck_position + 3) <= len(deck):
            cards_on_table.extend(deck[deck_position:deck_position+3])
            deck_position += 3
        else:
            end_game = True

    print()

print("End of the game, {0} sets found, average time per set = {1} s".format(sets_found, sum(exec_times)/len(exec_times)))
