# Implement PyGame Graphics

import math
import pygame
from pygame.rect import Rect


pts = ('topleft', 'topright', 'bottomleft', 'bottomright',
        'midtop', 'midright', 'midbottom', 'midleft', 'center')

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)

BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)

# by extension, we can also have a no-graphics canvas
class Canvas(object):
    def __init__(self, width, height):
        pygame.init()

        self.width, self.height = width, height

        self.screen = pygame.display.set_mode((width, height), 0, 32)  #
        self.clock = pygame.time.Clock()

        # sprites need to be added here
        self.all_sprites = pygame.sprite.Group()

        self.rectangle_objs = []
        self.circle_objs = []
        self.text_objs = {}

    def get_canvas_width(self):
        return self.width

    def get_canvas_height(self):
        return self.height

    def set_color(self, obj):
        raise Exception("color needs to be set during creation")

    def create_rectangle(self, x1, y1, x2, y2, color: str):
        color = eval(color.upper())
        width = math.fabs(x2 - x1)
        height = math.fabs(y2 - y1)
        # Rect(left, top, width, height)
        rect = Rect(x1, y1, width, height)
        self.rectangle_objs.append((rect, color))
        # pygame.draw.rect(self.screen, BLACK, rect)
        return rect

    def create_oval(self, x1, y1, x2, y2, radius, color: str):
        color = eval(color.upper())
        width = math.fabs(x2 - x1)
        height = math.fabs(y2 - y1)
        rect = Rect(x1, y1, width, height)
        self.circle_objs.append((rect, radius, color))

        return rect

    def create_text(self, x, y, text, font, size, store_key):
        pygame_font = pygame.font.SysFont(font, size)
        img = pygame_font.render(text, True, BLACK)
        self.text_objs[store_key] = (img, x, y, pygame_font, size)

    def set_text(self, store_key, text):
        # just update the thing
        if store_key in self.text_objs:
            _, x, y, font, size = self.text_objs[store_key]
            img = font.render(text, True, BLACK)
            self.text_objs[store_key] = (img, x, y, font, size)
            # self.create_text(x, y, text, font, size, store_key)

    def coords(self, obj: Rect):
        # based on Tkinter tradition 4 corners
        raise Exception("You shouldn't call this function")

    def find_overlapping(self, left, top, right, bottom):
        raise Exception("Resolve collision in PyGame way")

    def get_left_x(self, obj: Rect):
        # return one number
        return obj.midleft[0]

    def get_top_y(self, obj: Rect):
        # return one number
        return obj.midtop[1]

    def get_bottom_y(self, obj: Rect):
        return obj.midbottom[1]

    def get_right_x(self, obj: Rect):
        return obj.midright[0]

    def get_height(self, obj: Rect):
        return obj.height

    def get_width(self, obj: Rect):
        return obj.width

    def move_to(self, obj, new_x, new_y):
        old_x = self.get_left_x(obj)
        old_y = self.get_top_y(obj)
        self.move(obj, new_x - old_x, new_y - old_y)

    def moveto(self, obj: Rect, x, y):
        self.move_to(obj, x, y)

    def move(self, obj: Rect, dx, dy):
        obj.move_ip(dx, dy)

    def set_canvas_background_color(self, color):
        assert color == 'floral white', "We can only set this one color"
        self.screen.fill((255, 250, 240))

    def update(self):
        self.set_canvas_background_color('floral white')

        for rect, color in self.rectangle_objs:
            pygame.draw.rect(self.screen, color, rect)

        for rect, radius, color in self.circle_objs:
            pygame.draw.circle(self.screen, color, rect.center, radius)

        for key, (img, x, y, font, size) in self.text_objs.items():
            self.screen.blit(img, (x, y))

        pygame.display.flip()