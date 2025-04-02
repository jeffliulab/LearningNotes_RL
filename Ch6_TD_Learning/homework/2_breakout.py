"""
secondary question:

breakout用qlearning尝试解

将breakout视为一个NxN的grid图，然后分析：

1、动作：


"""

import pygame
import sys
import random

# 初始化 pygame
pygame.init()

# 游戏设置：10x10 网格，每格 50x50 像素，窗口 500x500
GRID_COLS, GRID_ROWS = 10, 10      
GRID_SIZE = 50                    
WIDTH, HEIGHT = GRID_COLS * GRID_SIZE, GRID_ROWS * GRID_SIZE
FPS = 60                         

# 定义颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)
GRAY  = (200, 200, 200)

# 游戏屏幕
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("简化版 Breakout (10x10 网格)")

clock = pygame.time.Clock()

# 砖块设置：共10个砖块，均匀分布在最上方，每个砖块占 1 格宽、1 格高
BRICK_COUNT = 10
BRICK_WIDTH = 1  # 每个砖块占1格宽
BRICK_HEIGHT = 1
bricks = []
for i in range(BRICK_COUNT):
    brick_x = i  # 每个砖块占1格
    brick_y = 0
    bricks.append(pygame.Rect(brick_x, brick_y, BRICK_WIDTH, BRICK_HEIGHT))

# 挡板设置：宽 3 格，高 1 格，位于最底部（第 10 行，索引 9）
PADDLE_WIDTH = 3
PADDLE_HEIGHT = 1
paddle_x = (GRID_COLS - PADDLE_WIDTH) // 2  # 初始挡板左侧格
paddle_y = GRID_ROWS - PADDLE_HEIGHT        # 位于底部

# 定义小球参数：占 1 格，初始位置位于挡板上方1格内（随机横向位置在挡板范围内）
BALL_SIZE = 1   # 占1个格子的大小
def reset_ball():
    ball_x = random.randint(paddle_x, paddle_x + PADDLE_WIDTH - 1)
    ball_y = paddle_y - 1
    # 初始速度：向上发射，水平随机（确保不为0）
    vx = random.choice([-1, 0, 1])
    if vx == 0:
        vx = 1
    vy = -1
    return float(ball_x), float(ball_y), vx, vy

ball_x, ball_y, vx, vy = reset_ball()
ball_speed = 9.0  # 小球速度：9格/秒

# 绘制砖块：砖块为红色矩形（每个砖块占满一个格子）
def draw_bricks():
    for brick in bricks:
        rect = pygame.Rect(brick.x * GRID_SIZE, brick.y * GRID_SIZE, 
                           brick.width * GRID_SIZE, brick.height * GRID_SIZE)
        pygame.draw.rect(screen, RED, rect)
        pygame.draw.rect(screen, BLACK, rect, 1)

# 绘制挡板：蓝色矩形
def draw_paddle(x):
    rect = pygame.Rect(x * GRID_SIZE, paddle_y * GRID_SIZE, 
                       PADDLE_WIDTH * GRID_SIZE, PADDLE_HEIGHT * GRID_SIZE)
    pygame.draw.rect(screen, BLUE, rect)

# 绘制小球：改为圆形，圆心在所在格的中心，半径为半个格子的宽度
def draw_ball(x, y):
    center = (int(x * GRID_SIZE + GRID_SIZE/2), int(y * GRID_SIZE + GRID_SIZE/2))
    radius = GRID_SIZE // 2
    pygame.draw.circle(screen, RED, center, radius)

def bounce_horizontal(vx):
    return -vx

def bounce_vertical(vy):
    return -vy

def point_in_rect(px, py, rect):
    # px, py 为小球中心的坐标（以格为单位），rect 为 pygame.Rect（以格为单位）
    return rect.collidepoint(px, py)

game_over = False
win = False

while not game_over:
    dt = clock.tick(FPS) / 1000.0  # 时间步长（秒）
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # 仅在 KEYDOWN 时响应，每次按键移动1格
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                paddle_x = max(0, paddle_x - 1)
            elif event.key == pygame.K_RIGHT:
                paddle_x = min(GRID_COLS - PADDLE_WIDTH, paddle_x + 1)
    
    # 更新小球位置（按 dt 时间步长移动）
    ball_x += vx * ball_speed * dt
    ball_y += vy * ball_speed * dt

    # 碰撞检测：左右边界（0 到 GRID_COLS - BALL_SIZE）
    if ball_x < 0:
        ball_x = 0
        vx = bounce_horizontal(vx)
    elif ball_x > GRID_COLS - BALL_SIZE:
        ball_x = GRID_COLS - BALL_SIZE
        vx = bounce_horizontal(vx)
    
    # 上边界碰撞
    if ball_y < 0:
        ball_y = 0
        vy = bounce_vertical(vy)
    
    # 检测与砖块的碰撞：判断小球中心是否进入砖块区域
    ball_center_x = ball_x + BALL_SIZE / 2
    ball_center_y = ball_y + BALL_SIZE / 2
    brick_hit = None
    for brick in bricks:
        if point_in_rect(ball_center_x, ball_center_y, brick):
            brick_hit = brick
            break
    if brick_hit:
        bricks.remove(brick_hit)
        vy = bounce_vertical(vy)
    
    # 检测与挡板的碰撞
    paddle_rect = pygame.Rect(paddle_x, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
    if vy > 0 and point_in_rect(ball_center_x, ball_center_y, paddle_rect):
        ball_y = paddle_y - BALL_SIZE
        vy = bounce_vertical(vy)
    
    # 游戏结束条件判断：若砖块全部消除则胜利，若小球出界则失败
    if len(bricks) == 0:
        print("Victory! You cleared all bricks.")
        win = True
        game_over = True
    if ball_y > GRID_ROWS:
        print("Game Over! Ball out of bounds.")
        game_over = True

    # 绘制画面
    screen.fill(WHITE)
    draw_bricks()
    draw_paddle(paddle_x)
    draw_ball(ball_x, ball_y)
    
    pygame.display.flip()

# 游戏结束后显示提示信息
font = pygame.font.SysFont(None, 48)
if win:
    text = font.render("Victory!", True, BLUE)
else:
    text = font.render("Game Over!", True, RED)
text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
screen.blit(text, text_rect)
pygame.display.flip()
pygame.time.wait(3000)

pygame.quit()
sys.exit()
