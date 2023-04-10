"""
Bounce but with invariances
"""

import math

from play2grade import utils_seeding as seeding
from play2grade.graphics import Canvas
import random
import time

import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'

import json

screen_width = 400
screen_height = 400
distance_from_boundary = 17
fps = 50

wall_thickness = 15

speed_text_available = ['random', 'very slow', 'slow', 'normal', 'fast', 'very fast']
speed_choices = ['very slow', 'slow', 'normal', 'fast', 'very fast']  #
speed_dict = {
    'very slow': 4, 'slow': 6, 'normal': 8, 'fast': 10, 'very fast': 12
}

paddle_speed_dict = {
    'very slow': 10, 'slow': 15, 'normal': 20, 'fast': 25, 'very fast': 30
}

game_name = "Bounce"
PAUSE = 'Pause'

MOVE_LEFT = "move left"
MOVE_RIGHT = "move right"
BOUNCE_BALL = "bounce ball"
SCORE_POINT = "score point"
SCORE_OPPO_POINT = "score opponent point"
LAUNCH_NEW_BALL = "launch new ball"

WHEN_RUN = 'when run'
WHEN_LEFT_ARROW = 'when left arrow'
WHEN_RIGHT_ARROW = 'when right arrow'

BALL_HIT_PADDLE = "when ball hits paddle"
BALL_HIT_WALL = "when ball hits wall"
BALL_IN_GOAL = "when ball in goal"
BALL_MISS_PADDLE = "when ball misses paddle"

BALL_DESTROYED = "ball destroyed"
BALL_ALREADY_IN_GOAL = "ball already in goal"

# Radius of the ball in pixels
BALL_RADIUS = 10
VELOCITY_Y = 8.0

# The ball's minimum and maximum horizontal velocity; the bounds of the
# initial random velocity that you should choose (randomly +/-).
VELOCITY_X_MIN = 2.0
VELOCITY_X_MAX = 8.0

# Dimensions of the paddle
PADDLE_WIDTH = 60
PADDLE_HEIGHT = 10

# Offset of the paddle up from the bottom
PADDLE_Y_OFFSET = 30

total_wall_width = 7

# Number of turns
NTURNS = 5

# Max balls on canvas at a time
MAX_NUM_BALLS = 2

CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400


class Config(object):
    def update(self):
        """
        This updates the self.config_dict, and set up attributes for the class
        :return:
        """
        for k, v in self.config_dict.items():
            self.__setattr__(k.replace(' ', "_"), v)

    def __getitem__(self, y):
        if "_" in y:
            y = y.replace("_", " ")
        return self.config_dict[y]

    def __setitem__(self, key, value):
        """
        :param key:
        :param value: be careful with value, it overrides things
        :return:
        """
        if "_" in key:
            key = key.replace("_", " ")
        self.config_dict[key] = value

    def save(self, file_name):
        json.dump(self.config_dict, open(file_name, "w"))

    def load(self, file_name):
        # this method overrides
        self.config_dict = json.load(open(file_name))
        self.update()

    def loads(self, json_obj):
        if type(json_obj) == str:
            result = None
            try:
                result = json.loads(json_obj)
            except:
                pass

            try:
                # we assume this is
                result = json.load(open(json_obj))
            except:
                pass

            assert result is not None, "We are not able to parse the obj you sent in"
            json_obj = result

        assert type(json_obj) == dict
        for key, values in json_obj.items():
            self.config_dict[key] = values
        self.update()

class Program(Config):
    def __init__(self):
        self.config_dict = {
            "when run": [],
            "when left arrow": [],
            "when right arrow": [],
            "when ball hits paddle": [],
            "when ball hits wall": [],
            "when ball in goal": [],
            "when ball misses paddle": []
        }
        self.update()
        self.cond_order = [WHEN_RUN, WHEN_LEFT_ARROW, WHEN_RIGHT_ARROW,
                           BALL_HIT_PADDLE, BALL_HIT_WALL, BALL_IN_GOAL, BALL_MISS_PADDLE]

        self.no_invar_command_order = [MOVE_LEFT, MOVE_RIGHT, BOUNCE_BALL, SCORE_POINT, SCORE_OPPO_POINT,
                                       LAUNCH_NEW_BALL]
        self.speed_invar_command_order = [f"set '{speed}' paddle speed" for speed in ['random', 'very slow', 'slow', 'normal', 'fast', 'very fast']]
        self.speed_invar_command_order += [f"set '{speed}' ball speed" for speed in ['random', 'very slow', 'slow', 'normal', 'fast', 'very fast']]

        self.theme_invar_command_order = [f"set '{theme}' scene" for theme in ['hardcourt', 'retro', 'random']]
        self.theme_invar_command_order += [f"set '{theme}' ball" for theme in ['hardcourt', 'retro', 'random']]
        self.theme_invar_command_order += [f"set '{theme}' paddle" for theme in ['hardcourt', 'retro', 'random']]

    def set_correct(self):
        # we generate a correct program
        self.config_dict['when run'].append(LAUNCH_NEW_BALL)
        self.config_dict['when left arrow'].append(MOVE_LEFT)
        self.config_dict['when right arrow'].append(MOVE_RIGHT)
        self.config_dict['when ball hits paddle'].append(BOUNCE_BALL)
        self.config_dict['when ball hits wall'].append(BOUNCE_BALL)
        self.config_dict['when ball in goal'].append(SCORE_POINT)
        self.config_dict['when ball in goal'].append(LAUNCH_NEW_BALL)
        self.config_dict['when ball misses paddle'].append(SCORE_OPPO_POINT)
        self.config_dict['when ball misses paddle'].append(LAUNCH_NEW_BALL)

    def to_features(self, no_invariance=False, no_theme_invariance=True):
        # compile the program into a one-hot encoding
        # used for code-as-text classifier and
        # also can be used to compress/condense programs (hashing feature)
        # this will generate the cardinality:
        # invariance: 7 x 27
        # no_invariance: 7 x 6
        if no_invariance:
            feat = [0] * 42
            for i, cond in enumerate(self.cond_order):
                for j, cmd in enumerate(self.no_invar_command_order):
                    if cmd in self.config_dict[cond]:
                        feat[i * len(self.no_invar_command_order) + j] = self.config_dict[cond].count(cmd)
        elif no_theme_invariance:
            feat = [0] * 7 * (6 + 12)
            for i, cond in enumerate(self.cond_order):
                for j, cmd in enumerate(self.no_invar_command_order + self.speed_invar_command_order):
                    if cmd in self.config_dict[cond]:
                        feat[i * len(self.no_invar_command_order + self.speed_invar_command_order) + j] = self.config_dict[cond].count(cmd)
        else:
            feat = [0] * 7 * (6 + 12 + 9)
            for i, cond in enumerate(self.cond_order):
                for j, cmd in enumerate(self.no_invar_command_order + self.speed_invar_command_order + self.theme_invar_command_order):
                    if cmd in self.config_dict[cond]:
                        feat[i * len(self.no_invar_command_order + self.speed_invar_command_order + self.theme_invar_command_order) + j] = self.config_dict[cond].count(cmd)

        return feat


class Paddle(object):
    def __init__(self, canvas: Canvas):
        self.canvas = canvas
        self.obj = self.setup_paddle()
        self.speed_offset = 20

    def setup_paddle(self):
        """
        Creates the paddle on screen at the location and size specified in
        the paddle constants. Returns the paddle.
        """
        canvas = self.canvas
        paddle_x = (canvas.get_canvas_width() - PADDLE_WIDTH) / 2
        paddle_y = canvas.get_canvas_height() - PADDLE_Y_OFFSET - PADDLE_HEIGHT
        paddle = canvas.create_rectangle(paddle_x, paddle_y,
                                         paddle_x + PADDLE_WIDTH, paddle_y + PADDLE_HEIGHT, 'black')
        return paddle

    def set_speed(self, new_speed, context=None):
        self.speed_offset = paddle_speed_dict[new_speed]

    def move_left(self):
        """
        """
        canvas = self.canvas
        paddle = self.obj
        old_paddle_x = canvas.get_left_x(paddle)
        new_paddle_x = old_paddle_x - self.speed_offset

        # handle out of boundary situation
        new_paddle_x = max(total_wall_width,
                           min((canvas.get_canvas_width() - total_wall_width) - PADDLE_WIDTH, new_paddle_x))

        canvas.moveto(paddle, new_paddle_x, canvas.get_top_y(paddle))

    def move_right(self):
        """
        """
        canvas = self.canvas
        paddle = self.obj
        old_paddle_x = canvas.get_left_x(paddle)
        new_paddle_x = old_paddle_x + self.speed_offset

        # handle out of boundary situation
        new_paddle_x = max(total_wall_width,
                           min((canvas.get_canvas_width() - total_wall_width) - PADDLE_WIDTH, new_paddle_x))

        canvas.moveto(paddle, new_paddle_x, canvas.get_top_y(paddle))

    # mouse testing
    def move_paddle(self):
        """
        Updates the paddle location to track the location of the mouse.  Specifically,
        sets the paddle location to keep the same y coordinate, but set the x coordinate
        such that the paddle is centered around the mouse.  Constrains the paddle to be
        entirely onscreen at all times.
        """
        raise Exception("this function can't work in PyGame")

        canvas, paddle = self.canvas, self.obj
        new_paddle_x = canvas.get_mouse_x() - PADDLE_WIDTH / 2
        new_paddle_x = max(total_wall_width,
                           min((canvas.get_canvas_width() - total_wall_width) - PADDLE_WIDTH, new_paddle_x))
        canvas.moveto(paddle, new_paddle_x, canvas.get_top_y(paddle))


class Ball(object):
    def __init__(self, canvas: Canvas, np_random):
        self.canvas = canvas
        self.np_random = np_random

        # degree out of 360
        self.direction = None
        self.vertical_speed = -VELOCITY_Y
        self.horizontal_speed = -VELOCITY_X_MAX  # random.uniform(VELOCITY_X_MIN, VELOCITY_X_MAX)

        self.obj: Rect = self.setup_ball()
        self.velocity_x, self.velocity_y = self.initialize_ball_velocity()

        self.x, self.y = self.init_x, self.init_y

        self.ball_already_in_goal = False
        self.ball_destroyed = False

    def move(self):
        # move to a displacement (change/delta)
        self.canvas.move(self.obj, self.velocity_x, self.velocity_y)
        # update position
        self.x, self.y = self.canvas.get_left_x(self.obj), self.canvas.get_top_y(self.obj)

    def reset(self):
        # move ball back to original position
        # a destroyed ball can also be reset

        self.ball_already_in_goal = False
        self.ball_destroyed = False
        self.canvas.moveto(self.obj, self.init_x, self.init_y)
        self.horizontal_speed = - VELOCITY_X_MAX
        self.vertical_speed = - VELOCITY_Y

        self.velocity_x, self.velocity_y = self.initialize_ball_velocity()

    def destroy(self):
        # a ball is ONLY destroyed if there's no "launch new ball" in
        # "when ball in goal" or "when ball misses paddle"
        if self.ball_destroyed:
            return

        self.ball_destroyed = True
        self.ball_already_in_goal = False
        self.x, self.y = -50, -50
        self.velocity_x, self.velocity_y = 0, 0
        self.canvas.moveto(self.obj, self.x, self.y)
        self.horizontal_speed = - VELOCITY_X_MAX
        self.vertical_speed = - VELOCITY_Y

    @property
    def tkinter_obj(self):
        return self.obj

    def initialize_ball_velocity(self):
        """
        Returns an initial x velocity value and y velocity value.  The x velocity
        is a random value between the min and max x velocity, and randomly positive or
        negative.  The y velocity is VELOCITY_Y.
        """
        self.direction = random.randrange(-30, 30)
        self.direction += 180

        direction_radians = math.radians(self.direction)
        velocity_x = self.horizontal_speed * math.sin(direction_radians)
        velocity_y = self.vertical_speed * math.cos(direction_radians)

        return velocity_x, velocity_y

    def set_speed(self, new_speed, context=None):
        """
        Set speed based on text
        We set self.horizontal_speed and self.vertical_speed
        and recalculate velocity, update it
        """
        speed = speed_dict[new_speed]

        # both vertical horizontal speeds are the same
        self.vertical_speed = -speed
        self.horizontal_speed = -speed

        direction_radians = math.radians(self.direction)
        velocity_x = self.horizontal_speed * math.sin(direction_radians)
        velocity_y = self.vertical_speed * math.cos(direction_radians)

        self.velocity_x, self.velocity_y = velocity_x, velocity_y

        if context == 'when_run':
            global VELOCITY_X_MAX
            global VELOCITY_Y
            VELOCITY_X_MAX = speed
            VELOCITY_Y = speed

    def setup_ball(self):
        """
        Creates the ball on-screen, centered, with the size
        as specified by BALL_RADIUS.  Returns the ball.
        """
        canvas = self.canvas

        # location is fixed, but velocity isn't
        ball_x = canvas.get_canvas_width() / 2 - BALL_RADIUS
        ball_y = canvas.get_canvas_height() / 2 - BALL_RADIUS

        ball = canvas.create_oval(ball_x, ball_y,
                                  ball_x + 2 * BALL_RADIUS, ball_y + 2 * BALL_RADIUS, BALL_RADIUS, 'black')

        self.init_x, self.init_y = ball_x, ball_y

        return ball

    def infer_ball_on_screen(self):
        if self.ball_destroyed:
            return False

        ball_y_top = self.canvas.get_top_y(self.obj)
        ball_y_bottom = ball_y_top + self.canvas.get_height(self.obj)

        on_screen = True
        if ball_y_bottom <= 0:
            on_screen = False
        elif ball_y_top >= CANVAS_HEIGHT:
            on_screen = False

        return on_screen

    def infer_ball_position(self, paddle: Paddle, program: Program):

        if self.ball_already_in_goal:
            return BALL_ALREADY_IN_GOAL

        canvas = self.canvas
        ball_x_right = canvas.get_right_x(self.obj)
        ball_y_top = canvas.get_top_y(self.obj)

        if canvas.get_top_y(self.obj) < total_wall_width and canvas.get_left_x(
                self.obj) > 100 - 1 and ball_x_right < 300 + 1:
            if BOUNCE_BALL not in program['when ball in goal']:
                self.ball_already_in_goal = True
            return BALL_IN_GOAL
        elif ball_y_top >= canvas.get_canvas_height():
            return BALL_MISS_PADDLE
        # left wall, right wall
        elif canvas.get_left_x(
                self.obj) < total_wall_width or ball_x_right >= canvas.get_canvas_width() - total_wall_width:
            return BALL_HIT_WALL
        # top wall
        elif canvas.get_top_y(self.obj) < total_wall_width:
            return BALL_HIT_WALL

        # instead of finding overlapping, we directly detect if this collide with paddle rect
        collide = self.obj.colliderect(paddle.obj)
        if collide:
            return BALL_HIT_PADDLE

        # None means it doesn't trigger anything
        return None

    def update_direction(self, velocity_x, velocity_y):
        # compute degree/direction from velocity_x, velocity_y
        # we rescale from speed_x, speed_y first (unit circle)
        unit_v_x = velocity_x / self.horizontal_speed
        unit_v_y = velocity_y / self.vertical_speed

        theta = math.degrees(math.atan2(unit_v_x, unit_v_y))

        self.direction = theta

    def bounce(self, condition, paddle: Paddle=None):
        canvas = self.canvas
        if condition == BALL_HIT_WALL:
            # if ball already in goal, we just let it go...
            if not self.ball_already_in_goal:
                ball_x_right = canvas.get_left_x(self.obj) + canvas.get_width(self.obj)
                # left wall
                if canvas.get_left_x(self.obj) <= total_wall_width:
                    self.velocity_x = math.fabs(self.velocity_x)
                    self.update_direction(self.velocity_x, self.velocity_y)
                    # self.velocity_x *= -1
                # right wall
                elif ball_x_right >= canvas.get_canvas_width() - total_wall_width:
                    self.velocity_x = - math.fabs(self.velocity_x)
                    self.update_direction(self.velocity_x, self.velocity_y)
                    # self.velocity_x *= -1
                # top wall
                elif canvas.get_top_y(self.obj) < total_wall_width:
                    self.velocity_y = math.fabs(self.velocity_y)
                    self.update_direction(self.velocity_x, self.velocity_y)
                    # self.velocity_y *= -1
        elif condition == BALL_HIT_PADDLE:
            assert paddle is not None
            # paddle.obj.top
            if self.velocity_y > 0:
                # let's allow paddle to exert force

                diff = (paddle.obj.left + paddle.obj.width / 2) - (self.obj.left + self.obj.width / 2)
                new_direction = (180 - self.direction) % 360
                new_direction -= diff

                direction_radians = math.radians(new_direction)
                self.velocity_x = self.horizontal_speed * math.sin(direction_radians)
                self.velocity_y = self.vertical_speed * math.cos(direction_radians)

                self.direction = new_direction

        elif condition == BALL_IN_GOAL:
            # just need to flip this one open
            self.ball_already_in_goal = False
            # self.velocity_y *= -1
            self.velocity_y = math.fabs(self.velocity_y)
            self.update_direction(self.velocity_x, self.velocity_y)
        elif condition == BALL_MISS_PADDLE:
            self.velocity_y = -math.fabs(self.velocity_y)
            self.update_direction(self.velocity_x, self.velocity_y)

class ScoreBoard(object):
    def __init__(self, canvas: Canvas, win_points=3):
        self.own = 0
        self.opponent = 0
        self.win_points = win_points

        self.canvas = canvas

        self.key = 'scoreboard_text'
        self.forced_game_over = False

    def score_point(self):
        self.own += 1
        self.update()

    def score_opponent_point(self):
        self.opponent += 1
        self.update()

    def draw(self):
        text = "Score " + str(self.own) + " : " + str(self.opponent)
        x_pos = (screen_width - 150) // 2

        self.canvas.create_text(x_pos, 50, text, "comicsansms", 25, self.key)

    def update(self):
        text = "Score " + str(self.own) + " : " + str(self.opponent)
        self.canvas.set_text(self.key, text)

    def set_game_over(self):
        self.forced_game_over = True

    def game_over(self):
        if self.forced_game_over:
            return True

        if self.own >= self.win_points or self.opponent >= self.win_points:
            return True
        else:
            return False


class RNG(object):
    def __init__(self):
        self.np_random = None
        self.curr_seed = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.curr_seed = seed
        return [seed]

    def choice(self, a, size=None, replace=True, p=None):
        return self.np_random.choice(a, size, replace, p)

    def randint(self, low, high=None, size=None, dtype=int):
        return self.np_random.randint(low, high, size, dtype)


def setup_walls(canvas: Canvas):
    # left wall
    brick = canvas.create_rectangle(0, 0, total_wall_width, 400, 'red')

    # right wall
    brick = canvas.create_rectangle(CANVAS_WIDTH - total_wall_width, 0, CANVAS_WIDTH, 400, 'red')

    # top left wall
    brick = canvas.create_rectangle(total_wall_width, 0, 100, total_wall_width, 'red')

    # top right wall
    brick = canvas.create_rectangle(300, 0, CANVAS_WIDTH - total_wall_width, total_wall_width, 'red')


import pygame
from pygame.locals import *

class Bounce(object):
    """
    """

    def __init__(self, program):
        self.rng = RNG()
        self.seed()

        assert program is not None, "Specify program as None."

        self.program = program

        canvas = Canvas(CANVAS_WIDTH, CANVAS_HEIGHT)
        canvas.set_canvas_background_color('floral white')

        setup_walls(canvas)

        self.paddle = Paddle(canvas)

        self.balls = []  # [Ball(canvas, self.rng)]

        self.score_board = ScoreBoard(canvas)

        self.canvas = canvas

        self.executable_cmds = [MOVE_LEFT, MOVE_RIGHT, SCORE_OPPO_POINT, SCORE_POINT, LAUNCH_NEW_BALL]

        self.fresh_run = True
        self.action_cmds = [pygame.K_RIGHT, pygame.K_LEFT, PAUSE]  # "None" action correspond to pause

        self.when_run_execute()

    def seed(self, seed=None):
        # we use a class object, so that if we update seed here, it broadcasts into everywhere
        return self.rng.seed(seed)

    def remove_bounce(self, cmds):
        # generate new list
        return [c for c in cmds if c != BOUNCE_BALL]

    def remove_set_cmds(self, cmds):
        # generate new list
        return [c for c in cmds if 'theme' not in c]

    def remove_bounce_and_set_cmds(self, cmds):
        # generate new list
        return [c for c in cmds if c != BOUNCE_BALL and 'theme' not in c]

    def execute(self, cmd, context=None):
        # we handle bounce inside Ball class
        assert cmd != "bounce ball"
        if cmd in self.executable_cmds:
            if " " in cmd:
                cmd = cmd.replace(" ", "_")
            eval("self." + cmd + "()")
        elif "ball speed" in cmd:
            self.set_ball_speed(cmd, context)
        elif "paddle speed" in cmd:
            self.set_paddle_speed(cmd, context)

    def extract_speed(self, cmd):
        speed_text = cmd.split("'")[1]
        assert speed_text in speed_text_available, "the speed test used is {}".format(speed_text)

        if speed_text == 'random':
            speed_text = self.rng.choice(speed_choices)  # choose a non-random option

        assert speed_text in speed_choices
        return speed_text

    def set_paddle_speed(self, cmd, context=None):
        speed = self.extract_speed(cmd)
        self.paddle.set_speed(speed, context)

    def set_ball_speed(self, cmd, context=None):
        speed = self.extract_speed(cmd)
        for ball in self.balls:
            ball.set_speed(speed, context)

    def when_run_execute(self):
        cmds = self.program[WHEN_RUN]
        cmds = self.remove_bounce_and_set_cmds(cmds)
        for cmd in cmds:
            self.execute(cmd, context='when_run')

    def when_left_arrow(self):
        cmds = self.program[WHEN_LEFT_ARROW]
        cmds = self.remove_bounce_and_set_cmds(cmds)
        for cmd in cmds:
            self.execute(cmd)

    def when_right_arrow(self):
        cmds = self.program[WHEN_RIGHT_ARROW]
        cmds = self.remove_bounce_and_set_cmds(cmds)
        for cmd in cmds:
            self.execute(cmd)

    def move_left(self):
        self.paddle.move_left()

    def move_right(self):
        self.paddle.move_right()

    def score_point(self):
        self.score_board.score_point()

    def score_opponent_point(self):
        self.score_board.score_opponent_point()

    def launch_new_ball(self):
        # we can have up to MAX_NUM_BALLS balls at any one point
        # we don't have a "remove" ball, we just reset them

        created = False
        for ball in self.balls:
            if ball.ball_destroyed:
                ball.reset()
                created = True
                break

        
        if len(self.balls) < MAX_NUM_BALLS and not created:
            self.balls.append(Ball(self.canvas, self.rng))

    def run(self, debug=False):

        self.score_board.draw()

        while not self.score_board.game_over():

            # pygame specific
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.score_board.set_game_over()

            # update ball
            for ball in self.balls:
                if ball.ball_destroyed:
                    continue
                ball.move()

            # arrow key based
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.when_left_arrow()
            elif keys[pygame.K_RIGHT]:
                self.when_right_arrow()

            # collision

            # 4 ball-related conditions
            # ====== new modularized code ======
            for ball in self.balls:
                if ball.ball_destroyed:
                    continue

                ball_condition = ball.infer_ball_position(self.paddle, self.program)

                if ball_condition == BALL_IN_GOAL:
                    cmds = self.program[BALL_IN_GOAL]
                    cmds = self.remove_set_cmds(cmds)
                    # If ball bounces off goal, nothing else happens
                    if BOUNCE_BALL in cmds:
                        ball.bounce(ball_condition)
                    else:
                        for cmd in cmds:
                            if cmd == LAUNCH_NEW_BALL:
                                # yeah, ball immediately disappears
                                # but that's ok...
                                if not self.score_board.game_over():
                                    ball.reset()
                            else:
                                # the reason is that, unless the ball is reset
                                # once in goal or miss paddle, it's gone
                                self.execute(cmd)
                elif ball_condition == BALL_MISS_PADDLE:
                    cmds = self.program[BALL_MISS_PADDLE]
                    cmds = self.remove_set_cmds(cmds)
                    for cmd in cmds:
                        if cmd == BOUNCE_BALL:
                            pass
                            # ball.bounce(ball_condition)
                        elif cmd == LAUNCH_NEW_BALL:
                            # yeah, ball immediately disappears
                            # but that's ok...
                            if not self.score_board.game_over():
                                ball.reset()
                        else:
                            self.execute(cmd)
                elif ball_condition == BALL_HIT_WALL:
                    cmds = self.program[BALL_HIT_WALL]
                    cmds = self.remove_set_cmds(cmds)
                    for cmd in cmds:
                        if cmd == BOUNCE_BALL:
                            ball.bounce(ball_condition)
                        else:
                            self.execute(cmd)
                elif ball_condition == BALL_HIT_PADDLE:
                    cmds = self.program[BALL_HIT_PADDLE]
                    cmds = self.remove_set_cmds(cmds)
                    for cmd in cmds:
                        if cmd == BOUNCE_BALL:
                            ball.bounce(ball_condition, self.paddle)
                        else:
                            self.execute(cmd)

                # add a safeguard here...
                # for out-of-boundary balls
                # if a ball is reset, it will be on screen
                ball_on_screen = ball.infer_ball_on_screen()
                if not ball_on_screen:
                    ball.destroy()

            self.canvas.update()
            time.sleep(1 / 50)

    def prefill_keys(self):
        # used for agents
        keys = {}
        for k in self.action_cmds:
            keys[k] = False
        return keys

    def act(self, action):
        self.score_board.draw()

        keys = self.prefill_keys()
        keys[action] = True

        # update ball
        for ball in self.balls:
            if ball.ball_destroyed:
                continue
            ball.move()


        # arrow key based
        if keys[pygame.K_LEFT]:
            self.when_left_arrow()
        elif keys[pygame.K_RIGHT]:
            self.when_right_arrow()

        # collision

        # 4 ball-related conditions
        # ====== modularized code ======
        for ball in self.balls:
            if ball.ball_destroyed:
                continue

            ball_condition = ball.infer_ball_position(self.paddle, self.program)

            if ball_condition == BALL_IN_GOAL:
                cmds = self.program[BALL_IN_GOAL]
                cmds = self.remove_set_cmds(cmds)
                if BOUNCE_BALL in cmds:
                    ball.bounce(ball_condition)
                else:
                    for cmd in cmds:
                        if cmd == LAUNCH_NEW_BALL:
                            # Note that we don't serve more balls when the game is over...
                            if not self.score_board.game_over():
                                ball.reset()
                        else:
                            self.execute(cmd)
            elif ball_condition == BALL_MISS_PADDLE:
                cmds = self.program[BALL_MISS_PADDLE]
                cmds = self.remove_set_cmds(cmds)
                for cmd in cmds:
                    if cmd == BOUNCE_BALL:
                        ball.bounce(ball_condition)
                    elif cmd == LAUNCH_NEW_BALL:
                        if not self.score_board.game_over():
                            ball.reset()
                    else:
                        self.execute(cmd)
            elif ball_condition == BALL_HIT_WALL:
                cmds = self.program[BALL_HIT_WALL]
                cmds = self.remove_set_cmds(cmds)
                for cmd in cmds:
                    if cmd == BOUNCE_BALL:
                        ball.bounce(ball_condition)
                    else:
                        self.execute(cmd)
            elif ball_condition == BALL_HIT_PADDLE:
                cmds = self.program[BALL_HIT_PADDLE]
                cmds = self.remove_set_cmds(cmds)
                for cmd in cmds:
                    if cmd == BOUNCE_BALL:
                        ball.bounce(ball_condition, self.paddle)
                    else:
                        self.execute(cmd)

            ball_on_screen = ball.infer_ball_on_screen()
            if not ball_on_screen:
                ball.destroy()

        self.canvas.update()

import PIL
import numpy as np
import gymnasium as gym
from gym import spaces
import PIL.Image

ONLY_SELF_SCORE = "only_self_score"
SELF_MINUS_HALF_OPPO = "self_minus_half_oppo"

def define_action_space(action_set):
    return spaces.Discrete(len(action_set))


def define_observation_space(screen_height, screen_width):
    return spaces.Box(low=0, high=255,
                      shape=(screen_height, screen_width, 3), dtype=np.uint8)


def define_object_observation_space(shape):
    # (x, y, direction)
    return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float)


class BounceEnv(gym.Env):
    """
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, program: Program, reward_type, num_balls_to_win=1, finish_reward=0,
                 info_bug=None):
        assert reward_type in {ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO}
        self.reward_type = reward_type

        self.viewer = None
        self.num_balls_to_win = num_balls_to_win
        self.finish_reward = finish_reward

        self.program = program
        self.bounce = None
        self.info_bug = info_bug

        BALL_LIMIT = 2  # only gonna take first 2 balls in the self.balls
        self.state_size = 3 + 6 * BALL_LIMIT
        self.observation_space = define_object_observation_space(shape=(self.state_size,))

    def seed(self, seed=None):
        return self.bounce.seed(seed)

    def get_state(self):
        # (paddle_left, paddle_right, paddle_top)
        feat = np.zeros(self.state_size)
        paddle_pos = [self.bounce.paddle.obj.left, self.bounce.paddle.obj.right, self.bounce.paddle.obj.top]

        feat[:3] = paddle_pos
        # (ball_left, ball_bottom, ball_top, ball_right, ball_vel_x, ball_vel_y)
        BALL_LIMIT = 2
        for i, ball in enumerate(self.bounce.balls[:BALL_LIMIT]):
            feat[3+i*6:3+(i+1)*6] = [ball.obj.left, ball.obj.bottom, ball.obj.top, ball.obj.right, ball.velocity_x,
                             ball.velocity_y]
        return feat

    def step(self, action):
        if action >= len(self.bounce.action_cmds):
            raise Exception("Can't choose an action based on index {}".format(action))

        prev_score = self.bounce.score_board.own
        prev_oppo_score = self.bounce.score_board.opponent

        # map action into key
        action_key = self.bounce.action_cmds[int(action)]

        self.bounce.act(action_key)

        score = self.bounce.score_board.own
        oppo_score = self.bounce.score_board.opponent

        score_diff = score - prev_score
        oppo_score_diff = oppo_score - prev_oppo_score

        done = self.bounce.score_board.game_over()

        if self.reward_type == SELF_MINUS_HALF_OPPO:
            reward = score_diff * 20 - oppo_score_diff * 10
            reward = max(-10, reward)
        else:
            reward = score_diff * 20

        info = {'bug_state': False}
        if len(self.bounce.balls) == 1:
            ball_condition = self.bounce.balls[0].infer_ball_position(self.bounce.paddle, self.bounce.program)
            if ball_condition == BALL_IN_GOAL and self.info_bug == BALL_IN_GOAL:
                info['bug_state'] = True
            elif ball_condition == BALL_MISS_PADDLE and self.info_bug == BALL_MISS_PADDLE:
                info['bug_state'] = True
            elif ball_condition == BALL_HIT_WALL and self.info_bug == BALL_HIT_WALL:
                info['bug_state'] = True
            elif ball_condition == BALL_HIT_PADDLE and self.info_bug == BALL_HIT_PADDLE:
                info['bug_state'] = True

        info["score"] = score
        info["oppo_score"] = oppo_score

        return self.get_state(), reward, done, info

    def reset(self):
        self.bounce = Bounce(self.program)
        self.bounce.score_board.win_points = self.num_balls_to_win
        self.action_space = define_action_space(self.bounce.action_cmds)
        return self.get_state()

    def render(self, mode='human'):
        img = self.get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        return

    def get_image(self):
        """
        https://mail.python.org/pipermail/python-list/2006-August/371647.html
        https://pillow.readthedocs.io/en/stable/reference/Image.html
        :return:
        """
        image_str = pygame.image.tostring(self.bounce.canvas.screen, 'RGB')
        image = PIL.Image.frombytes(mode='RGB', size=(screen_height, screen_width), data=image_str)
        image_np_array = np.array(image)
        return image_np_array

if __name__ == '__main__':
    program = Program()
    program.set_correct()
    # program.load("programs/correct_speed_change.json")
    program.load("programs/miss_paddle_no_launch_ball.json")

    # program.load("programs/hit_goal_no_point.json")
    # program.load("programs/empty.json")
    # program.load("programs/multi_ball.json")
    # program.load("programs/ball_through_wall.json")
    # program.load("programs/goal_bounce.json")
    # program.load("programs/multi_ball2.json")
    # program.load("programs/paddle_not_bounce.json")

    game = Bounce(program)
    game.run()

    # test RL environment
    # env = BounceEnv(program, SELF_MINUS_HALF_OPPO)
    # env.reset()
    # for i in range(500):
    #     state, reward, done, info = env.step(2)
    #     print(env.bounce.score_board.opponent, env.bounce.score_board.game_over())
    #     print(reward)
    #     if done:
    #         break
