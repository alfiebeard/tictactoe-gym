# tictactoe-gym

This is an OpenAI gym environment for playing Tic Tac Toe (or Noughts and Crosses).

## Installation
```
pip install tictactoe_gym
```

## Overview
Each player takes it in turn to mark a position in a square grid (e.g., 3x3), until they form a horizontal, vertical or diagonal line across the grid (e.g., 3 in a row), in which case they win. If no further moves can be made and there is no winner then the game is a draw.

### Action Space
The action is an `integer` which can take values $\{0, 1, ..., n^{2} - 1\}$, where $n$ is the size of the grid, starting at grid position $[0, 0]$, which is action $0$, then on to $[0, 1]$ which is action $1$, and moving through row by row, until action $n^{2} - 1$, 
which is $[n - 1, n - 1]$.

For example, action 4 is the centre position, $[1, 1]$, in a $3 \times 3$ game.

### Observation Space
The observation is a `ndarray` with shape `(n, n)`, where n is the grid size. Each entry can take the following values:

| Value | Meaning                                |
|-------|----------------------------------------|
| 0     | No mark here yet, free to mark         |
| 1     | Player 1 has placed a mark here        |
| -1    | Player 2 has placed a mark here        |

### Rewards
A reward of +1 is given to the winning player with a reward of -1 for the losing player. If it's a draw both players get a reward of 0.

### Starting State
$n \times n$ grid of zeros.

### Episode End
The episode ends if any one of the following occurs:
1. Termination: A player gets $n$ successive marks in a row, column or diagonal for an $n \times n$ grid.
2. Termination: No more moves can be played (i.e., every grid position is marked).

### Arguments
```
gym.make('tictactoe-v0')
```

No additional arguments are currently supported.

## Environment

### Attributes

**size (int):** The size of the grid.

**observation_space (gym.spaces.Box):** The observation space, an $n \times n$ grid.

**action_space (gym.spaces.Discrete):** The action space, an $n^{2} - 1$ list.

**reward_range (int):** The reward range, -1 for a loss, 0 for a draw, +1 for a win.

**_player (int):** The current player, 1/-1.

**_state_size (tuple):** A (1, 3) tuple of the state size - (size, size, 1).

**_action_to_index_map (dict):** A mapping from actions to indices, e.g., {0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [1, 0], ..., 8: [2, 2]}.

**_history (list) :** The history of actions, e.g., [0, 5, 1, 3, 2].

**_terminal (bool):** Whether the game is finished or not.

**_winner (int):** The winning player, 1 for player 1, -1 for player 2 and 0 for a draw.

### Methods

**__init__:** Initialises environment and all attributes.

**get_observation:** Gets the observation for a player.

**get_actions:** Get possible actions, i.e., positions where marks have not yet been made.

**get_result:** Get result for a player. If the player is 1/-1 and winner is 1/-1, then return 1, as the requested player has won. Otherwise if the player is 1/-1 and winner is -1/1, then return -1, as the player has lost. If a draw, then self._winner is 0, so returns 0.

**_get_action_to_index_map:** Returns the action_to_index_map. This is more efficient as it saves having to compute indices every time.

**_is_valid_action:** Checks an action is valid.

**_row_winner:** Returns winner of any row from an observation.

**_col_winner:** Returns winner of any column from an observation.

**_main_diag_winner:** Returns winner of main diagonal from an observation.

**_reverse_main_diag_winner:** Returns winner of reverse main diagonal from an observation.

**_is_game_over:** Returns true if game is over and false otherwise. Also sets the winning player as the _winner or 0 for draw.

**_get_info:** Get information on the game.

**step:** Step game using action and returns new observation, winner, game over indicator, truncated (always False here) and info.

**reset:** Reset the game.

**clone:** Clone the game.

**render:** Renders the current observation in the terminal as a string.
