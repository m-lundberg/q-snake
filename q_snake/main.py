from q_snake.game import SnakeGame
from q_snake.agent import DeepQLearningAgent


if __name__ == '__main__':
    episodes = 1000

    game = SnakeGame(framerate=8)
    agent = DeepQLearningAgent(len(game.ACTION_SPACE), gamma=0.9)

    for episode in range(episodes):
        print(f'Starting episode number {episode}')

        state = game.reset()

        while True:

            action = agent.choose_action(state, game.ACTION_SPACE, epsilon=0.9)
            next_state, reward, done, info = game.step(action)

            if done:
                # TODO: Do more things. Save trained network?
                game.reset()
                break

            agent.learn(state, action, reward, next_state)
            state = next_state
