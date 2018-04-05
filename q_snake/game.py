import pygame
from PIL import Image
import numpy as np
import random
import sys

BLACK = (0, 0, 0)
SNAKE_COLOR = (255, 255, 255)
APPLE_COLOR = (255, 0, 0)


class SnakeGame:
    def __init__(self, rewards=(1, 0, -1), screen_size=(300, 300), tile_size=(20, 20), image_size=(15, 15), framerate=0):
        self.apple_reward = rewards[0]
        self.neutral_reward = rewards[1]
        self.death_reward = rewards[2]

        self.screen_size = screen_size
        self.tile_size = tile_size
        self.image_size = image_size
        self.framerate = framerate

        # settings that user can control through keyboard shortcuts that need to be handled outside the game
        self.settings = {
            'explore': True,
            'learn': True,
        }

        self.fast_mode = False
        self.manual_mode = False

        self.ACTION_SPACE = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption('Q-Snake')
        self.clock = pygame.time.Clock()

        self.snake = None
        self.snake_positions = []

        self.direction = None

        self.apple = None
        self.apple_position = None

        self.reset()

    def reset(self):
        self.snake = pygame.Surface(self.tile_size)
        self.snake.fill(SNAKE_COLOR)
        self.snake_positions = [tuple(random.randint(0, screen/tile - 1) * tile for screen, tile in zip(self.screen_size, self.tile_size))]

        self.direction = (0, 1)

        self.apple = pygame.Surface(self.tile_size)
        self.apple.fill(APPLE_COLOR)
        self.apple_position = self.place_apple()

        self.draw()
        return self.take_screenshot(), {'settings': self.settings}

    def step(self, action):
        reward = 0
        done = False

        self.clock.tick(self.framerate if not self.fast_mode else 0)

        # update direction according to input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

            elif event.type == pygame.KEYUP and event.key == pygame.K_a and self.direction != (1, 0):
                self.direction = (-1, 0)
            elif event.type == pygame.KEYUP and event.key == pygame.K_d and self.direction != (-1, 0):
                self.direction = (1, 0)
            elif event.type == pygame.KEYUP and event.key == pygame.K_w and self.direction != (0, 1):
                self.direction = (0, -1)
            elif event.type == pygame.KEYUP and event.key == pygame.K_s and self.direction != (0, -1):
                self.direction = (0, 1)

            elif event.type == pygame.KEYUP and event.key == pygame.K_r:
                self.settings['explore'] = not self.settings['explore']
            elif event.type == pygame.KEYUP and event.key == pygame.K_f:
                self.fast_mode = not self.fast_mode
            elif event.type == pygame.KEYUP and event.key == pygame.K_t:
                self.settings['learn'] = not self.settings['learn']
            elif event.type == pygame.KEYUP and event.key == pygame.K_m:
                self.manual_mode = not self.manual_mode
                if self.manual_mode:
                    self.settings['learn'] = False
            elif event.type == pygame.KEYUP and event.key == pygame.K_h:
                self.print_help()

        if action and not self.manual_mode:
            # make sure that we can't turn 180 degrees
            if [-x for x in self.direction] != action:
                self.direction = action

        # get the next position of the snakes head
        next_position = tuple(self.snake_positions[0][i] + d * tile for i, (d, tile) in enumerate(zip(self.direction, self.tile_size)))

        # check if snake will eat itself
        if any(self.collision(next_position, p) for p in self.snake_positions):
            reward = self.death_reward
            done = True

        # check if snake is going outside the grid
        if not 0 <= next_position[0] <= self.screen_size[0] - self.tile_size[0] or \
                not 0 <= next_position[1] <= self.screen_size[1] - self.tile_size[1]:
            reward = self.death_reward
            done = True

        # check if snake will eat apple
        if self.collision(next_position, self.apple_position):
            self.snake_positions = [next_position] + self.snake_positions
            self.apple_position = self.place_apple()

            reward = self.apple_reward
        else:
            self.snake_positions = [next_position] + self.snake_positions[:-1]

            if reward is not self.death_reward:
                # we have not died in this step
                reward = self.neutral_reward

        self.draw()

        return self.take_screenshot(), reward, done, {'settings': self.settings}

    def draw(self):
        self.screen.fill(BLACK)

        for pos in self.snake_positions:
            self.screen.blit(self.snake, pos)
        self.screen.blit(self.apple, self.apple_position)

        pygame.display.flip()

    def take_screenshot(self):
        data = pygame.image.tostring(self.screen, 'RGB')

        image = Image.frombytes('RGB', self.screen_size, data)

        # convert image to greyscale and resize
        image = image.convert('L')
        image = image.resize(self.image_size)

        matrix = np.asarray(image.getdata(), dtype=np.int8)
        return matrix.reshape(image.size)

    def place_apple(self):
        while True:
            candidate = tuple(random.randint(0, screen/tile - 1) * tile for screen, tile in zip(self.screen_size, self.tile_size))
            if not any(self.collision(p, candidate) for p in self.snake_positions):
                return candidate

    def collision(self, p1, p2):
        return p1[0] < p2[0] + self.tile_size[0] and \
               p1[0] + self.tile_size[0] > p2[0] and \
               p1[1] < p2[1] + self.tile_size[1] and \
               p1[1] + self.tile_size[1] > p2[1]

    @staticmethod
    def print_help():
        print('The snake can be controlled using WASD')
        print('Keyboard shortcuts:')
        print('  r - toggle random exploration (default True)')
        print('  f - toggle fast mode (in fast mode, the game runs at max speed) (default False)')
        print('  t - toggle training, that is whether the agent should learn from experiences or not (default True)')
        print('  m - toggle manual input mode, and disable learning if manual mode is enabled (default False)')
        print('  h - show this help text')
