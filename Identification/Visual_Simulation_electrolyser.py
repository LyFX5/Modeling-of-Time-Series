
from threading import Thread
import pygame, sys
from electrolyser_T import Electrolyser
import random
import time

import numpy as np

public_variable_U = 0
public_variable_point = [0.,0.]
public_variable_point_Temperature = [0.,0.]
public_variable_color = [255, 255, 255]
public_variable_pixels_per_time_step = 0.2

class Visualise_Tread(Thread):
    def __init__(self, name):
        """Инициализация потока"""
        Thread.__init__(self)
        self.name = name

    def run(self):
        global public_variable_U, public_variable_point, public_variable_color, public_variable_point_Temperature
        """Запуск потока"""
        amount = random.randint(3, 9)  # 15
        time.sleep(amount)

        pygame.init()

        point_radius = 5

        height_of_screen = 700
        width_of_screen = 1500
        screen = pygame.display.set_mode([width_of_screen, height_of_screen])
        font = pygame.font.Font(None, 32)
        screen.fill([255, 255, 255])
        running = True
        Pos = [0, 0]

        input_box = pygame.Rect(100, 100, 140, 32)
        color_inactive = pygame.Color('lightskyblue3')
        color_active = pygame.Color('dodgerblue2')
        color = color_inactive
        text = ''

        active = False

        while running:
            time.sleep(0.05/5)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # забеливаем прошлое место
                    pygame.draw.circle(screen, [255, 255, 255], center=Pos, radius=point_radius)
                    Pos = event.pos
                    if input_box.collidepoint(event.pos):
                        # Toggle the active variable.
                        active = not active
                    else:
                        active = False
                    # Change the current color of the input box.
                    color = color_active if active else color_inactive
                if event.type == pygame.KEYDOWN:
                    if active:
                        if event.key == pygame.K_RETURN:
                            public_variable_U = int(text)*1.
                            print(text)
                            text = ''
                        elif event.key == pygame.K_BACKSPACE:
                            text = text[:-1]
                        else:
                            text += event.unicode
                    # screen.blit(food, Pos)
                    # Food = obs.SystemObjekt([foodPos[0] / 10, foodPos[1] / 10], "food", 5)

            screen.fill((255, 255, 255), input_box)
            #pygame.draw.rect(screen, color, input_box, 2)

            # draw y
            point_y = [public_variable_point[0]*public_variable_pixels_per_time_step, height_of_screen-public_variable_point[1] - point_radius]
            pygame.draw.circle(screen, public_variable_color, center=point_y, radius=point_radius) # рисую точку тут

            # draw T
            point_T = [public_variable_point_Temperature[0] * public_variable_pixels_per_time_step,
                     height_of_screen - public_variable_point_Temperature[1] - point_radius]
            pygame.draw.circle(screen, [255, 100, 100], center=point_T, radius=point_radius)  # рисую точку тут

            # Render the current text.
            txt_surface = font.render(text, True, color)
            # Resize the box if the text is too long.
            width = max(200, txt_surface.get_width() + 10)
            input_box.w = width
            # Blit the text.
            screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
            # Blit the input_box rect.
            pygame.draw.rect(screen, color, input_box, 2)

            pygame.display.flip()
            #clock.tick(30)
            # if Food.eatten:
            #   pygame.draw.rect(screen,[255,255,255],[foodPos[0],foodPos[1],90,90],0)

        pygame.quit()


class Integration_Thread(Thread):
    def __init__(self, name):
        """Инициализация потока"""
        Thread.__init__(self)
        self.name = name

    def run(self):
        global public_variable_U, public_variable_point, public_variable_color, public_variable_point_Temperature
        """Запуск потока"""
        amount = random.randint(3, 15)
        time.sleep(amount)

        H2_amount_in_dimamics = []

        delta_t = 0.5  # time step size (seconds)
        t_max = (1500 - 20)/public_variable_pixels_per_time_step # width_of_screen = 1500, 1 time steps =  pixels / public_variable_pixels_per_time_step

        time_work = np.linspace(0, t_max, int(t_max // delta_t))

        ID = 1

        elec1 = Electrolyser(ID, delta_t)

        for i in range(len(time_work)):
            time.sleep(0.1/5)

            U = public_variable_U
            elec1.apply_control_signal_in_moment(U)
            [y, yd, ydd] = elec1.getDinamics()

            [Temper, dTemper] = elec1.TemperatureDinamics

            y_momentum = y

            H2_amount_in_dimamics.append(y_momentum)

            # отображение тока
            public_variable_point = [time_work[i], y_momentum*500]
            # умножаю на 500, потому что высота фрейма равна 700, а выход лизера это значение от 0 до 1, где 1 это 53 А

            # отображение температуры
            public_variable_point_Temperature = [time_work[i], Temper*500/53]

            # раскрашиваю в зависимости от состояния
            if elec1.state == 'idle':
                public_variable_color = [0, 255, 255]
            elif elec1.state == 'hydration':
                public_variable_color = [255, 0, 0]
            elif elec1.state == 'ramp_up_1':
                public_variable_color = [0, 255, 0]
            elif elec1.state == 'ramp_up_2':
                public_variable_color = [100, 30, 160]
            elif elec1.state == 'steady':
                public_variable_color = [255, 255, 0]
            elif elec1.state == 'ramp_down_1':
                public_variable_color = [180, 190, 0]
            elif elec1.state == 'ramp_down_2':
                public_variable_color = [0, 100, 50]


def create_threads():
    integration = Integration_Thread("Integration")
    integration.start()
    visual = Visualise_Tread("visual")
    visual.start()


if __name__ == "__main__":
    create_threads()