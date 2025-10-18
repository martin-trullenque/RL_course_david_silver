import numpy as np 
import random


def draw_card():
    card = random.randint(1, 10)
    if random.random() < 1/3:
        return -card  # Ace as -1
    else:
        return card
    
def step(state, action):
    dealer_card, player_sum = state
    if action == 'hit':
        card = draw_card()
        if card < 0:
            player_sum += card
        else:
            player_sum += card
        if player_sum < 1 or player_sum > 21:
            return (None, None), -1  # Player busts
        else:
            return (dealer_card, player_sum), 0  # Game continues
    elif action == 'stick':
        while dealer_card < 17:
            card = draw_card()
            if card < 0:
                dealer_card += card
            else:
                dealer_card += card
        if dealer_card < 1 or dealer_card > 21 or player_sum > dealer_card:
            return (None, None), 1  # Player wins
        elif player_sum < dealer_card:
            return (None, None), -1  # Dealer wins
        else:
            return (None, None), 0  # Draw
        
def reset():
    dealer_card = draw_card()
    player_sum = draw_card() + draw_card()
    return (dealer_card, player_sum)

def is_terminal(state):
    return state == (None, None)

def get_valid_actions(state):
    return ['hit', 'stick']

def render(state):
    dealer_card, player_sum = state
    if is_terminal(state):
        print("Game over")
    else:
        print(f"Dealer's card: {dealer_card}, Player's sum: {player_sum}")
        