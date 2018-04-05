from q_snake.game import SnakeGame
from q_snake.agent import DeepQLearningAgent
import time
import os


if __name__ == '__main__':
    episodes = 1000

    game = SnakeGame(framerate=8)

    model_path = None
    if os.path.isfile('model.h5'):
        model_path = 'model.h5'
    agent = DeepQLearningAgent(game.ACTION_SPACE, image_shape=(*game.image_size, 1), network_path=model_path, gamma=0.9)

    game.print_help()

    for episode in range(episodes):
        print(f'Starting episode number {episode}')

        state, info = game.reset()

        while True:
            action = agent.choose_action(state, game.ACTION_SPACE, epsilon=(0.9 if info['settings']['explore'] else 0.0))
            next_state, reward, done, info = game.step(action)

            if info['settings']['learn']:
                t = time.time()
                agent.learn(state, action, reward, next_state, done)
                print(f'Learning time: {time.time() - t} s')

            if done:
                agent.Q.save('model.h5')
                state, info = game.reset()
                break

            state = next_state
