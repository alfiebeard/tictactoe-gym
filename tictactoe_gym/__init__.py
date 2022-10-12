from gym.envs.registration import register


register(
    id='tictactoe_gym/tictactoe-v0',
    entry_point='tictactoe_gym.envs:TicTacToeEnv',
)