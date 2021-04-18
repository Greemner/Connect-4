import pygame
import os
import json
import numpy as np
from kaggle_environments import make
from moviepy.editor import VideoFileClip
from MCTS import MCTS_agent as agent3
from Minimax import agent as agent2
from operator import itemgetter

# Считывание данных пользователей из файла. Если первый запуск программы - создаем пустой файл для записи
try:
    DATA = json.load(open('data.txt'))
except FileNotFoundError:
    DATA = {}

pygame.init()

# Инициализируем геометрические параметры окна и задаем ограничение на частоту кадров
WIDTH = 800
HEIGHT = 600
FPS = 30

COLOR = pygame.Color(153, 99, 90)  # Цвет символов при вводе имени пользователя
FONT = pygame.font.Font(None, 28)
DIFFICULTY = None  # Сложность: 0 - negamax(kaggle implementation), 1 - minimax, 2 - MCTS
USER = None  # Имя пользователя

# Создаем иконку, называем окно и вызываем показ клипа перед запуском
display = pygame.display.set_mode((WIDTH, HEIGHT))
icon = pygame.image.load(os.path.join("Картинки", "icon.png"))
pygame.display.set_icon(icon)
pygame.display.set_caption("Connect 4")
clip = VideoFileClip("Заставка.mp4")

# Для музыки
pygame.mixer.music.load("koto.mp3")
music1 = pygame.image.load(os.path.join("Картинки", "Переключатель 1.png"))
music2 = pygame.image.load(os.path.join("Картинки", "Переключатель 4.png"))
on_off_music = True  # Изначательно музыка включена
pygame.mixer.music.play(loops=100)  # Делаем loops для возобновления песни после ее завершения
display.blit(music2, (675, 100))  # Отрисовываем включенное состояние переключателя

# Загрузка необходимых картинок
empty_name = pygame.image.load(os.path.join("Картинки", "Пустое поле ввода.png"))
full_name = pygame.image.load(os.path.join("Картинки", "Поле ввода.png"))
mark = pygame.image.load(os.path.join("Картинки", "Галка.png"))
wrong = pygame.image.load(os.path.join("Картинки", "Ошибка.png"))
warning = pygame.image.load(os.path.join("Картинки", "Предупреждение.png"))
lose = pygame.image.load(os.path.join("Картинки", "Проигрыш.png"))
win = pygame.image.load(os.path.join("Картинки", "Выигрыш.png"))
tie = pygame.image.load(os.path.join("Картинки", "Ничья.png"))

# Для ограничения кадров
clock = pygame.time.Clock()


class GameBoard:
    global DATA, USER, DIFFICULTY

    def __init__(self, agent):
        self.agent = agent  # Алгоритм, против которого играем
        self.human_mark = pygame.image.load(os.path.join("Картинки", "Фишка 1.png"))  # Фишка первого игрока
        self.computer_mark = pygame.image.load(os.path.join("Картинки", "Фишка 2.png"))  # Фишка второго игрока
        self.env = make("connectx", debug=True)  # Симулятор игровой среды
        self.trainer = self.env.train([None, self.agent])
        self.observation = self.trainer.reset()  # self.observation.board - текущее состояние доски (массив 1*42)
        self.old_observation = self.observation  # Необходимо для вычисления выбранной алгоритмом колонки
        self.agent_turn = False
        self.human_column = None  # Выбранная игроком колонка
        self.computer_column = None  # Выбранная компьютером колонка
        self.cell = None  # Незаполненная ячейка в выбранной для хода колонки
        self.fall = False  # Падает ли фишка
        self.x = 0  # Для прорисовки падения фишки
        self.y = 0  # Для прорисовки падения фишки
        self.first = True  # Если первый кадр отрисовки падения
        self.done = False  # Закончилась ли игра
        self.frames = 0  # Сколько кадров отображать после завершения FIXME
        self.points = 0  # Сколько очков получили за игру
        self.show_hint = False  # Показывать ли подсказки
        self.result = ''  # Win/Lose/Tie
        self.reward = 0  # Награда, возвращаемая симулятором среды
        self.hint_column = 3  # Рекомендуемая подсказкой колонка

    def draw(self):
        # Если нужно отрисовать падение фишки компьютера. Grid не содержит последнего хода компьютера после trainer.step
        if self.fall and not self.agent_turn:
            grid = np.asarray(self.old_observation.board).reshape(6, 7)
        # Если нужно отрисовать падение фишки человека или просто отрисовать все фишки. Grid содержит все фишки
        elif (self.fall and self.agent_turn) or not self.fall:
            grid = np.asarray(self.observation.board).reshape(6, 7)

        # Отрисовываем все фишки из grid
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i][j] == 1:
                    display.blit(self.human_mark, (100 + 87.5 * j, 62.5 + i * 87.5))
                elif grid[i][j] == 2:
                    display.blit(self.computer_mark, (100 + 87.5 * j, 62.5 + i * 87.5))

        # Считываем колонку, на которую нажал человек
        if not self.fall and not self.done:
            click = pygame.mouse.get_pressed()
            if click[0]:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for i in range(7):
                    if 100 + 87.5 * i < mouse_x < 175 + 87.5 * i and 62.5 < mouse_y < 575:
                        self.human_column = i
                        self.agent_turn = True  # Следующий ход компьютера
                        self.fall = True  # Флаг для запуска процесса отрисовки падения фишки
                        break

        if self.cell is None and self.fall:
            # Если agent_turn == true, то это значит, что человек выбрал колонку, поэтому нужно отрисовать процесс
            # падения фишки человека. Если false, то компьютер сходил и необходимо отрисовать его фишку
            if self.agent_turn:
                column = self.human_column
            else:
                column = self.computer_column
            self.cell = 5
            # Находим строку, в которую должна упасть новая фишка
            for j in range(6):
                if grid[j][column] != 0:
                    self.cell = j - 1
                    break

        # Падение фишки человека
        if self.fall and self.agent_turn:
            self.mark_fall(self.cell, self.human_mark, self.human_column)
        # Падение фишки компьютера
        elif self.fall and not self.agent_turn:
            self.mark_fall(self.cell, self.computer_mark, self.computer_column)
        elif not self.done:
            # Если ход компьютера
            if self.agent_turn:
                self.old_observation = self.observation
                self.observation, self.reward, self.done, info = self.trainer.step(self.human_column)
                # observation содержит 2 новых значения (колонка человека и компьютера)
                # Для корректного отображения падения фишки компьютера нужно найти где стоит новое значение человека
                for i in range(len(self.observation.board)):
                    if self.old_observation.board[i] != self.observation.board[i]:
                        if self.observation.board[i] == 1:
                            self.old_observation.board[i] = 1
                        elif self.observation.board[i] == 2:
                            self.computer_column = i % 7
                        self.fall = True

                if self.show_hint:
                    self.hint()
                self.agent_turn = False

        # Если проигрыш/выигрыш/ничья
        if self.done:
            self.when_done()

    def hint(self):
        self.hint_column = agent3(self.observation, None, t=2)  # t - ограничение на время поиска оптимального хода

    def when_done(self):
        if self.points == 0:
            if self.reward == 0:
                self.points = max(-10 ** (DIFFICULTY + 1), -20)  # -10/-20/-20
                self.result = 'Lose'
            elif self.reward == 1:
                self.points = 10 ** (DIFFICULTY + 1)  # 10/100/1000
                self.result = 'Win'
            else:
                self.points = 0.1 * 10 ** (DIFFICULTY + 1)  # 1/10/100
                self.result = 'Tie'

            DATA[USER] += self.points
            json.dump(DATA, open('data.txt', 'w'))

        self.show_hint = False
        # Отрисовывка игрового поля до тех пор, пока не нажата кнопка "домой" или "еще раз"
        grid = np.asarray(self.observation.board).reshape(6, 7)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i][j] == 1:
                    display.blit(self.human_mark, (100 + 87.5 * j, 62.5 + i * 87.5))
                elif grid[i][j] == 2:
                    display.blit(self.computer_mark, (100 + 87.5 * j, 62.5 + i * 87.5))
        # Координаты кнопок house x = 325, y = 325, w = 26, h = 37   circle x = 450, y = 337, w = 26, h = 26
        if self.result == 'Lose':
            display.blit(lose, (300, 225))
        elif self.result == 'Win':
            display.blit(win, (300, 225))
        elif self.result == 'Tie':
            display.blit(tie, (300, 225))
        # Обновляем поле и все переменные
        click = pygame.mouse.get_pressed()
        if click[0]:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if 325 < mouse_x < 351 and 325 < mouse_y < 362:  # Если нажали "домой"
                show_menu()
            elif 450 < mouse_x < 476 and 337 < mouse_y < 363:  # Если нажали "еще раз"
                pygame.time.delay(500)
                self.__init__(self.agent)


    def mark_fall(self, cell, mark, player_column):
        # Если первый кадр отрисовки падения, инициализируем начальное состояние исходя из номера колонки
        if self.first:
            self.x = 100 + 87.5 * player_column
            self.y = 62.5
            self.first = False
        else:
            self.y += 13  # Опускаем фишку по оси у
        # Если опустили на нужный уровень
        if self.y >= 62.5 + 87.5 * cell:
            self.cell = None
            self.fall = False
            self.first = True
        else:
            display.blit(mark, (self.x, self.y))


# Класс для введения символов
class InputBox:

    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)  # Поле ввода имени пользователя
        self.active_pic = empty_name  # Пустой фон поля ввода
        self.inactive_pic = full_name  # На фоне поля ввода написано "Введите имя"
        self.color = COLOR  # Цвет текста
        self.text = text  # Текст, введенный в поле ввода
        self.txt_surface = FONT.render(text, True, self.color)
        self.active = False  # Состояние поля ввода. True - нажали и вводим имя, False - отображается "Введите имя"
        self.flag_name = None
        self.ok = Button(61, 41, 'Ок')

    def handle_event(self, event):
        global USER, DATA
        if event.type == pygame.MOUSEBUTTONDOWN:  # Если пользователь кликнул
            if self.rect.collidepoint(event.pos):  # Если кликнул на поле ввода (актив -> неактив, неактив -> актив)
                if self.active:  # Если поле было активно, отрисовываем пустой текст, чтобы не было наложения "Введите имя" и текста
                    self.txt_surface = FONT.render('', True, self.color)
                else:
                    self.txt_surface = FONT.render(self.text, True, self.color)  # Если поле было неактивно, отображаем текст
                self.active = not self.active
            else:
                self.txt_surface = FONT.render(self.text, True, self.color)

        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:  # Если нажали Enter
                    if self.text not in DATA.keys():  # Если введенного имени нет в базе данных
                        self.flag_name = True
                        USER = self.text
                        DATA[USER] = 0  # self.text - имя пользователя. Записываем имя в файл
                        json.dump(DATA, open('data.txt', 'w'), ensure_ascii=False)
                    else:
                        self.flag_name = False

                elif event.key == pygame.K_BACKSPACE:  # Если стираем символы
                    self.text = self.text[:-1]
                else:
                    if len(self.text) < 20:  # Имя не может быть длиннее 20 символов
                        self.text += event.unicode
                # Ререндерим текстовое поле
                self.txt_surface = FONT.render(self.text, True, self.color)

    def draw(self, screen):
        global USER
        if self.active:
            screen.blit(self.active_pic, (195, 100))
        else:
            screen.blit(self.inactive_pic, (195, 100))

        if self.flag_name and self.flag_name is not None:  # Если введеного имени нет в базе данных, отображаем галочку
            screen.blit(mark, (549, 101))
        elif not self.flag_name and self.flag_name is not None:  # Если имя есть в базе данных
            mouse_x, mouse_y = pygame.mouse.get_pos()
            self.ok.draw(620, 105)
            screen.blit(wrong, (549, 101))
            if 549 < mouse_x < 604 and 101 < mouse_y < 151:  # Отображаем предупреждение, если мышка наведена на !
                screen.blit(warning, (580, 140))
            if 620 < mouse_x < 681 and 105 < mouse_y < 146:
                click = pygame.mouse.get_pressed()
                if click[0]:  # Если кликнули на ОК
                    self.flag_name = True
                    USER = self.text

        screen.blit(self.txt_surface, (self.rect.x, self.rect.y+5))


# Класс для кнопок
class Button:
    def __init__(self, width_img, height_img, name, number=None, special=False):
        self.width = width_img
        self.height = height_img
        self.normal = pygame.image.load(os.path.join("Картинки", name + " нормал.png"))
        self.hover = pygame.image.load(os.path.join("Картинки", name + " покрыто.png"))
        self.pressed = pygame.image.load(os.path.join("Картинки", name + " нажато.png"))
        self.number = number  # Необходимо для радио-кнопок выбора уровня сложности
        self.name = name
        self.special = special  # Необходимо для кнопки "вперед" в окне ввода имени и выбора сложности

    def draw(self, x, y, action=None):
        global DIFFICULTY, USER
        mouse_x, mouse_y = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        if (x < mouse_x < x + self.width) and (y < mouse_y < y + self.height):  # Если навели на область кнопки
            display.blit(self.hover, (x, y))  # Отрисовываем покрытое состояние кнопки

            if click[0]:
                display.blit(self.pressed, (x, y))  # Отрисовываем нажатое состояние кнопки
                if self.number is not None:
                    DIFFICULTY = self.number
                if self.name == 'Назад':   # Если это кнопка "назад", то обнуляем сложность и имя пользователя
                    DIFFICULTY = None
                    USER = None
                if action is not None:  # Если кнопка что-то вызывает, то вызываем
                    if self.special and DIFFICULTY is not None and USER is not None:  # Для срабатывания специальной кнопки необходимо, чтобы была выбрана сложность и введено имя пользователя
                        action()
                    elif not self.special:
                        action()

        elif self.number is not None:
            if self.number == DIFFICULTY:
                display.blit(self.pressed, (x, y))
            else:
                display.blit(self.normal, (x, y))

        else:
            display.blit(self.normal, (x, y))


# Функция для отрисовки переключателя музыки
def music():
    global on_off_music

    mouse_x, mouse_y = pygame.mouse.get_pos()
    on_off_coord_x = 675
    on_off_coord_y = 100

    # Отрисовываем состояние переключателя, когда мышка не наведена
    if on_off_music:
        display.blit(music2, (on_off_coord_x, on_off_coord_y))
    if not on_off_music:
        display.blit(music1, (on_off_coord_x, on_off_coord_y))

    # Меняем состояние переключателя, когда мышка наведена
    if on_off_coord_x < mouse_x < 738 and on_off_coord_y < mouse_y < 130:
        if on_off_music:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    display.blit(music1, (on_off_coord_x, on_off_coord_y))
                    on_off_music = False
                    pygame.mixer.music.pause()

        else:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    display.blit(music2, (on_off_coord_x, on_off_coord_y))
                    on_off_music = True
                    pygame.mixer.music.unpause()


# Функция для отрисовки главного меню
def show_menu():
    global on_off_music, music1, music2
    show = True
    menu_back = pygame.image.load(os.path.join("Картинки", "Фон меню.png"))
    play = Button(323, 79, 'Играть')
    rules = Button(323, 79, 'Правила')
    rating = Button(323, 79, 'Рейтинг')

    while show:
        # Проверяем не нажат ли крестик (закрытие приложения)
        quit_func()
        display.blit(menu_back, (0, 0))
        play.draw(238.5, 100, show_play)
        rules.draw(238.5, 260, show_rules)
        rating.draw(238.5, 420, show_rating)

        # Проверяем состояние переключателя
        music()

        pygame.display.flip()
        clock.tick(FPS)


# Функция для отрисовки меню под кнопкой ИГРАТЬ
def show_play():
    show = True
    name_box = InputBox(220, 112.5, 350, 25)
    lite_diff = Button(116, 26, 'Легкий', number=0)
    medium_diff = Button(131, 31, 'Средний', number=1)
    hard_diff = Button(140, 26, 'Сложный', number=2)

    play_back = pygame.image.load(os.path.join("Картинки", "Фон играть.png"))
    back = Button(85, 33, 'Назад')
    next = Button(85, 33, 'Вперед', special=True)

    while show:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            name_box.handle_event(event)

        display.blit(play_back, (0, 0))
        name_box.draw(display)
        back.draw(10, 10, show_menu)
        next.draw(705, 557, show_board)
        lite_diff.draw(330, 350)
        medium_diff.draw(330, 400)
        hard_diff.draw(330, 450)

        pygame.display.flip()
        clock.tick(FPS)

# Функция для отрисовки рейтинга
def show_rating():
    show = True
    rating_back = pygame.image.load(os.path.join("Картинки", "Рейтинг фон.png"))
    next_5 = pygame.image.load(os.path.join("Картинки", "Стрелка вправо.png"))
    prev_5 = pygame.image.load(os.path.join("Картинки", "Стрелка влево.png"))
    back = Button(85, 33, 'Назад')
    color = pygame.Color(144, 108, 126)
    list_of_points = list(DATA.items())
    if len(list_of_points) % 5 == 0:  # Если количество пользователей в базе данных кратно 5
        num_pages = len(list_of_points) // 5
    else:
        num_pages = len(list_of_points) // 5 + 1
    page = 0  # Номер страницы для отображения

    while show:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if 200 < mouse_x < 244 and 520 < mouse_y < 570 and page != 0:  # Если нажали стрелку назад и это не первая страница
                    page -= 1
                elif 560 < mouse_x < 604 and 520 < mouse_y < 570 and page != num_pages - 1:  # Если нажали стрелку вперед и это не последняя страница
                    page += 1
        k = 0
        display.blit(rating_back, (0, 0))
        back.draw(10, 10, show_menu)
        FONT_for_rating = pygame.font.Font(None, 36)

        # Сортировка
        points_back = sorted(list_of_points, key=itemgetter(1), reverse=True)

        if (page + 1) * 5 > len(list_of_points):  # Для последней страницы
            endpoint = len(list_of_points)
        else:
            endpoint = (page + 1) * 5

        for i in range(page * 5, endpoint):
            txt_surface = FONT_for_rating.render(str(i + 1), True, color)  # Место пользователя
            display.blit(txt_surface, (205, 152 + k))
            txt_surface = FONT_for_rating.render(str(points_back[i][0]), True, color)  # Имя пользователя
            display.blit(txt_surface, (262.5, 152 + k))
            txt_surface = FONT_for_rating.render(str(int(points_back[i][1])), True, color)  # Баллы пользователя
            display.blit(txt_surface, (558.5, 152 + k))
            k += 75  # Перемещаемся к следующему месту на странице

        # Отображение кнопок вперед и назад
        if page == 0:
            display.blit(next_5, (560, 520))
        elif page == num_pages - 1:
            display.blit(prev_5, (200, 520))
        else:
            display.blit(prev_5, (200, 520))
            display.blit(next_5, (560, 520))

        pygame.display.flip()
        clock.tick(FPS)


# Функция для отрисовки правил
def show_rules():
    show = True
    rules_back = pygame.image.load(os.path.join("Картинки", "Правила текст.png"))
    back = Button(85, 33, 'Назад')

    while show:
        quit_func()
        display.blit(rules_back, (0, 0))
        back.draw(10, 10, show_menu)

        pygame.display.flip()
        clock.tick(FPS)


def show_board():
    show = True
    board_back = pygame.image.load(os.path.join("Картинки", "Фон поле.png"))
    hint_on = pygame.image.load(os.path.join("Картинки", "Лампочка вкл.png"))
    hint_off = pygame.image.load(os.path.join("Картинки", "Лампочка выкл.png"))
    pointer = pygame.image.load(os.path.join("Картинки", "Указка.png"))
    back_board = Button(85, 33, 'Назад')

    if DIFFICULTY == 0:
        board = GameBoard('negamax')
    elif DIFFICULTY == 1:
        board = GameBoard(agent2)
    elif DIFFICULTY == 2:
        board = GameBoard(agent3)

    while show:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if 721 < mouse_x < 721 + 41 and 21 < mouse_y < 21 + 63:
                    if not board.show_hint:
                        board.hint()
                    board.show_hint = not board.show_hint  # Если нажали на лампочку меняем состояние show_hint на противоположное

        display.blit(board_back, (0, 0))

        if not board.show_hint:
            display.blit(hint_off, (721, 21))
        else:
            display.blit(pointer, (126.5 + 87.5 * board.hint_column, 13))  # Указатель на лучшую колонку
            display.blit(hint_on, (680, 5))

        board.draw()
        back_board.draw(10, 10, show_menu)
        pygame.display.flip()
        clock.tick(FPS)


def quit_func():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()


clip.preview()
show_menu()
