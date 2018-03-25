import sys
import numpy as np
import pygame
from maple.games.utils.vec2d import vec2d
from maple.games.utils import percent_round_int

from maple.games.base.pygamewrapper import PyGameWrapper

class Ball(pygame.sprite.Sprite):

    def __init__(self, radius, speed, rng,
                 pos_init, SCREEN_WIDTH, SCREEN_HEIGHT):

        pygame.sprite.Sprite.__init__(self)

        self.rng = rng
        self.radius = radius
        self.speed = speed
        self.pos = vec2d(pos_init)
        self.pos_before = vec2d(pos_init)
        self.vel = vec2d((speed, -1.0 * speed))

        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        image = pygame.Surface((radius * 2, radius * 2))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.circle(
            image,
            (255, 255, 255),
            (radius, radius),
            radius,
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def line_intersection(self, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):

        s1_x = p1_x - p0_x
        s1_y = p1_y - p0_y
        s2_x = p3_x - p2_x
        s2_y = p3_y - p2_y

        s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y)
        t = (s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y)

        return (s >= 0 and s <= 1 and t >= 0 and t <= 1)

    def update(self, agentPlayer, cpuPlayer, dt):

        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt

        is_pad_hit = False

        if self.pos.x <= agentPlayer.pos.x + agentPlayer.rect_width:
            if self.line_intersection(self.pos_before.x, self.pos_before.y, self.pos.x, self.pos.y, agentPlayer.pos.x + agentPlayer.rect_width / 2, agentPlayer.pos.y - agentPlayer.rect_height / 2, agentPlayer.pos.x + agentPlayer.rect_width / 2, agentPlayer.pos.y + agentPlayer.rect_height / 2):
                self.pos.x = max(0, self.pos.x)
                self.vel.x = -1 * (self.vel.x + self.speed * 0.05)
                self.vel.y += agentPlayer.vel.y * 2.0
                self.pos.x += self.radius
                is_pad_hit = True

        if self.pos.x >= cpuPlayer.pos.x - cpuPlayer.rect_width:
            if self.line_intersection(self.pos_before.x, self.pos_before.y, self.pos.x, self.pos.y, cpuPlayer.pos.x - cpuPlayer.rect_width / 2, cpuPlayer.pos.y - cpuPlayer.rect_height / 2, cpuPlayer.pos.x - cpuPlayer.rect_width / 2, cpuPlayer.pos.y + cpuPlayer.rect_height / 2):
                self.pos.x = min(self.SCREEN_WIDTH, self.pos.x)
                self.vel.x = -1 * (self.vel.x + self.speed * 0.05)
                self.vel.y += cpuPlayer.vel.y * 0.006
                self.pos.x -= self.radius
                is_pad_hit = True

        # Little randomness in order not to stuck in a static loop
        if is_pad_hit:
            self.vel.y += self.rng.random_sample() * 0.001 - 0.0005

        if self.pos.y - self.radius <= 0:
            self.vel.y *= -0.99
            self.pos.y += 1.0

        if self.pos.y + self.radius >= self.SCREEN_HEIGHT:
            self.vel.y *= -0.99
            self.pos.y -= 1.0

        self.pos_before.x = self.pos.x
        self.pos_before.y = self.pos.y

        self.rect.center = (self.pos.x, self.pos.y)


class Player(pygame.sprite.Sprite):

    def __init__(self, speed, rect_width, rect_height,
                 pos_init, SCREEN_WIDTH, SCREEN_HEIGHT):

        pygame.sprite.Sprite.__init__(self)

        self.speed = speed
        self.pos = vec2d(pos_init)
        self.vel = vec2d((0, 0))

        self.rect_height = rect_height
        self.rect_width = rect_width
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        image = pygame.Surface((rect_width, rect_height))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.rect(
            image,
            (255, 255, 255),
            (0, 0, rect_width, rect_height),
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, dy, dt):
        self.vel.y += dy * dt
        self.vel.y *= 0.9

        self.pos.y += self.vel.y

        if self.pos.y - self.rect_height / 2 <= 0:
            self.pos.y = self.rect_height / 2
            self.vel.y = 0.0

        if self.pos.y + self.rect_height / 2 >= self.SCREEN_HEIGHT:
            self.pos.y = self.SCREEN_HEIGHT - self.rect_height / 2
            self.vel.y = 0.0

        self.rect.center = (self.pos.x, self.pos.y)



class Pong(PyGameWrapper):
    """
    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    MAX_SCORE : int (default: 11)
        The max number of points the agent or cpu need to score to cause a terminal state.
        
    cpu_speed_ratio: float (default: 0.5)
        Speed of opponent (useful for curriculum learning)
        
    players_speed_ratio: float (default: 0.25)
        Speed of player (useful for curriculum learning)

    ball_speed_ratio: float (default: 0.75)
        Speed of ball (useful for curriculum learning)

    """

    def __init__(self, width=64, height=48, players_speed_ratio = 0.4, ball_speed_ratio=0.75,  MAX_SCORE=11):

        actions_set = [["p1_noop", "p1_up", "p1_down"], ["p2_noop", "p2_up", "p2_down"]]

        PyGameWrapper.__init__(self, width, height, n_agents=2, actions_set=actions_set)

        # the %'s come from original values, wanted to keep same ratio when you
        # increase the resolution.
        self.ball_radius = percent_round_int(height, 0.03)

        self.ball_speed_ratio = ball_speed_ratio
        self.players_speed_ratio = players_speed_ratio

        self.paddle_width = percent_round_int(width, 0.023)
        self.paddle_height = percent_round_int(height, 0.15)
        self.paddle_dist_to_wall = percent_round_int(width, 0.0625)
        self.MAX_SCORE = MAX_SCORE

        self.dys = [0.0, 0.0]
        self.score_counts = np.zeros(self.n_agents)

    def _handle_players_actions(self):
        self.dys = [0.0, 0.0]

        for i in range(self.n_agents):
            if self.current_actions[i] == 1:  # up
                self.dys[i] = -self.players[i].speed
            elif self.current_actions[i] == 2: # down
                self.dys[i] = self.players[i].speed

    def _handle_events(self):        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.event.pump()

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        Returns
        -------

        dict
            * players y position.
            * players velocity.
            * ball x position.
            * ball y position.
            * ball x velocity.
            * ball y velocity.

            See code for structure.

        """
        state = {
            "player1_y": self.players[0].pos.y,
            "player1_velocity": self.players[0].vel.y,
            "player2_y": self.players[1].pos.y,
            "player2_velocity": self.players[1].vel.y,
            "ball_x": self.ball.pos.x,
            "ball_y": self.ball.pos.y,
            "ball_velocity_x": self.ball.vel.x,
            "ball_velocity_y": self.ball.vel.y
        }

        return state

    def getScores(self):
        return self.scores 

    def game_over(self):
        # pong used 11 as max score
        return (self.score_count == self.MAX_SCORE)

    def init(self):
        self.score_count = 0
        self.scores = np.zeros(2)

        self.ball = Ball(
            self.ball_radius,
            self.ball_speed_ratio * self.height,
            self.rng,
            (self.width / 2, self.height / 2),
            self.width,
            self.height
        )

        self.players = [Player(
            self.players_speed_ratio * self.height, self.paddle_width,
            self.paddle_height, (self.paddle_dist_to_wall, self.height / 2),
            self.width, self.height),
                        Player(
            self.players_speed_ratio * self.height, self.paddle_width,
            self.paddle_height, (self.width - self.paddle_dist_to_wall, self.height / 2),
            self.width, self.height)]

        self.players_group = pygame.sprite.Group()
        self.players_group.add(self.players[0])
        self.players_group.add(self.players[1])

        self.ball_group = pygame.sprite.Group()
        self.ball_group.add(self.ball)


    def reset(self):
        self.init()
        # after game over set random direction of ball otherwise it will always be the same
        self._reset_ball(1 if self.rng.random_sample() > 0.5 else -1)


    def _reset_ball(self, direction):
        self.ball.pos.x = self.width / 2  # move it to the center

        # we go in the same direction that they lost in but at starting vel.
        self.ball.vel.x = self.ball.speed * direction
        self.ball.vel.y = (self.rng.random_sample() *
                           self.ball.speed) - self.ball.speed * 0.5

    def step(self, dt):
        dt /= 1000.0
        self.screen.fill((0, 0, 0))

        self.players[0].speed = self.players_speed_ratio * self.height
        self.players[1].speed = self.players_speed_ratio * self.height
        self.ball.speed = self.ball_speed_ratio * self.height

        self._handle_players_actions()
        self._handle_events()

        self.ball.update(self.players[0], self.players[1], dt)

        is_terminal_state = False

        # logic
        if self.ball.pos.x <= 0:
            self.score_count += 1
            self.scores += np.array([self.rewards["negative"],self.rewards["positive"]])
            self._reset_ball(-1)
            is_terminal_state = True

        if self.ball.pos.x >= self.width:
            self.score_count += 1
            self.scores += np.array([self.rewards["positive"],self.rewards["negative"]])
            self._reset_ball(1)
            is_terminal_state = True

        if not is_terminal_state:
            self.players[0].update(self.dys[0], dt)
            self.players[1].update(self.dys[1], dt)

        self.players_group.draw(self.screen)
        self.ball_group.draw(self.screen)
        
    def expert_policy(self, i):
        action = 0
        if self.players[i].pos.y > self.ball.pos.y:
            action = 1
        else:
            action = 2
        return action


if __name__ == "__main__":
    pygame.init()
    game = Pong(width=256, height=200)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(60)
        game.step(dt)
        pygame.display.update()
