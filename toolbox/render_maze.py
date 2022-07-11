from math import atan2, cos, sin
from typing import List, Optional, Tuple

import numpy as np
import pygame
from PIL import Image
from pygame import Color, Surface, gfxdraw

SCREEN_DIM = 500
BOUND = 13
SCALE = SCREEN_DIM / (BOUND * 2)
OFFSET = SCREEN_DIM // 2

BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)
GRAY = Color(192, 192, 192)
GREEN = Color(0, 200, 0)
RED = Color(255, 0, 0)
BLUE = Color(0, 0, 255)


def draw_filled_circle(surf: Surface, pos: np.ndarray, radius: int = 1, color: Color = RED):
    """
    Draw a filled circle on the surface

    :param surf: The surface
    :type surf: Surface
    :param pos: Position of the surface
    :type pos: np.ndarray
    :param radius: Radius of the circle, defaults to 1
    :type radius: int, optional
    :param color: Color of the filling, defaults to RED
    :type color: Color, optional
    """
    x, y = pos * SCALE + OFFSET
    gfxdraw.filled_circle(surf, int(x), int(y), radius, color)


def draw_circle(surf: Surface, pos: np.ndarray, radius: int = 3, color: Color = GREEN):
    """
    Draw a circle on the surface

    :param surf: The surface
    :type surf: Surface
    :param pos: Position of the surface
    :type pos: np.ndarray
    :param radius: Radius of the circle, defaults to 3
    :type radius: int, optional
    :param color: Color of the line, defaults to GREEN
    :type color: Color, optional
    """
    x, y = pos * SCALE + OFFSET
    gfxdraw.circle(surf, int(x), int(y), radius, color)


def draw_line(surf: Surface, point_a: np.ndarray, point_b: np.ndarray, thickness: float = 2, color: Color = BLACK):
    """
    Draw a line on the surface

    :param surf: The surface
    :type surf: Surface
    :param point_a: One point
    :type point_a: np.ndarray
    :param point_b: The other point
    :type point_b: np.ndarray
    :param thickness: Thickness of the line
    :type: float
    :param color: Color of the line, defaults to BLACK
    :type color: Color, optional
    """
    point_a = point_a * SCALE + OFFSET
    point_b = point_b * SCALE + OFFSET
    # gfxdraw.filled_polygon(surf, ((0, 0), (100, 100), (50, 30)), BLACK)
    # gfxdraw.line(surf, int(x1), int(y1), int(x2), int(y2), color)

    center = (point_a + point_b) / 2.0

    # Then find the slope (angle) of the line:
    length = np.linalg.norm(point_a - point_b)  # Total length of line
    angle = atan2(point_a[1] - point_b[1], point_a[0] - point_b[0])

    # Using the slope and the shape parameters you can calculate the following
    # coordinates of the box ends:
    UL = (
        center[0] + (length / 2.0) * cos(angle) - (thickness / 2.0) * sin(angle),
        center[1] + (thickness / 2.0) * cos(angle) + (length / 2.0) * sin(angle),
    )
    UR = (
        center[0] - (length / 2.0) * cos(angle) - (thickness / 2.0) * sin(angle),
        center[1] + (thickness / 2.0) * cos(angle) - (length / 2.0) * sin(angle),
    )
    BL = (
        center[0] + (length / 2.0) * cos(angle) + (thickness / 2.0) * sin(angle),
        center[1] - (thickness / 2.0) * cos(angle) + (length / 2.0) * sin(angle),
    )
    BR = (
        center[0] - (length / 2.0) * cos(angle) + (thickness / 2.0) * sin(angle),
        center[1] - (thickness / 2.0) * cos(angle) - (length / 2.0) * sin(angle),
    )

    # Using the computed coordinates, we draw an unfilled anti-aliased polygon
    # (thanks to @martineau) and then fill it as suggested in the documentation
    # of pygame's gfxdraw module for drawing shapes.
    pygame.gfxdraw.aapolygon(surf, (UL, UR, BR, BL), color)
    pygame.gfxdraw.filled_polygon(surf, (UL, UR, BR, BL), color)


def rot(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Return the input point rotated around (0, 0) by a given angle.

    :param x: Input array of shape (2,)
    :type x: np.ndarray
    :param theta: Angle in degrees
    :type theta: float
    :return: The result
    :rtype: np.ndarray
    """
    r = np.radians(theta)
    c, s = np.cos(r), np.sin(r)
    r = np.array(((c, -s), (s, c)))
    return np.matmul(r, x)


def render_and_save(
    all_pos: np.ndarray,
    walls: np.ndarray,
    filename: str,
    goals: np.ndarray = None,
    trajectories: List[np.ndarray] = None,
    bg: Color = BLACK,
    grid: Optional[Tuple[int, int, int]] = None,
):
    """
    Render and save a maze.

    :param all_pos: A matrix of shape (num_pos x 2) containng all the visited positions
    :type all_pos: np.ndarray
    :param walls: The wall as a array of shape (num_walls x 2 x 2)
    :type walls: np.ndarray
    :param filename: The filename of the output image
    :type filename: str
    :param goals: The goals as an array of shape (num_pos x 2), defaults to None
    :type goals: np.ndarray, optional
    :param trajectories: The trajectories as a list of array of shape (num_pos x 2), , defaults to None
    :type trajectories: np.ndarray, optional
    :param bg: Background color, defaults to BLACK
    :type bg: Color, optional
    :param grid: A grid, as a tuple (origin, width, angle), defaults to None
    :type grid: Optional[Tuple[int, int, int]], optional
    """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_DIM, SCREEN_DIM))
    surf = Surface((SCREEN_DIM, SCREEN_DIM))

    # Background
    surf.fill(bg)

    if grid is not None:
        origin, width, angle = grid
        for i in range(400):
            draw_line(
                surf,
                rot(np.array([-20, (i - 200 + origin) * width]), angle),
                rot(np.array([20, (i - 200 + origin) * width]), angle),
                color=GRAY,
                thickness=1,
            )
            draw_line(
                surf,
                rot(np.array([(i - 200 + origin) * width, -20]), angle),
                rot(np.array([(i - 200 + origin) * width, 20]), angle),
                color=GRAY,
                thickness=1,
            )

    # Draw visited points
    for pos in all_pos:
        draw_filled_circle(surf, pos)

    # Draw goals if any
    goals = [] if goals is None else goals
    for pos in goals:
        draw_filled_circle(surf, pos, color=GREEN)

    trajectories = [] if trajectories is None else trajectories
    for trajectory in trajectories:
        for i in range(len(trajectory) - 1):
            draw_line(surf, trajectory[i], trajectory[i + 1], color=GREEN)

    # Draw walls
    for point_a, point_b in walls:
        draw_line(surf, point_a, point_b, color=BLACK if bg == WHITE else WHITE)

    surf = pygame.transform.flip(surf, flip_x=False, flip_y=True)
    screen.blit(surf, (0, 0))
    im = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    im = Image.fromarray(im)
    im.save(filename)
