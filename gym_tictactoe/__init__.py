from gym.envs.registration import register


register(
    id='gym_tictactoe/tictactoe-v0',
    entry_point='gym_tictactoe.envs:TicTacToeEnv',
)