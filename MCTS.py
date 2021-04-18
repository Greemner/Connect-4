import random
import math
import time


def MCTS_agent(observation, configuration, t=1.5):
    global current_state

    init_time = time.time()
    EMPTY = 0
    T_max = t  # время на ход
    Cp_default = 1

    def play(board, column, mark, config):
        columns = 7
        rows = 6
        # находим самую верхнюю строчку в которой у нас есть свободное место в колонке
        row = max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])
        # закидываем в свободное место фишку
        board[column + (row * columns)] = mark

    def is_win(board, column, mark, config):
        # Детектируем 4 фишки стоящих в ряд
        columns = 7
        rows = 6
        inarow = 3
        row = min([r for r in range(rows) if board[column + (r * columns)] == mark])

        def count(offset_row, offset_column):
            for i in range(1, inarow + 1):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                        r < 0
                        or r >= rows
                        or c < 0
                        or c >= columns
                        or board[c + (r * columns)] != mark
                ):
                    return i - 1
            return inarow

        return (
                count(1, 0) >= inarow  # вертикально
                or (count(0, 1) + count(0, -1)) >= inarow  # горизонтально в ряд
                or (count(-1, -1) + count(1, 1)) >= inarow  # диагонально слева
                or (count(-1, 1) + count(1, -1)) >= inarow  # диагонально справа
        )

    def is_tie(board):
        # Проверяем наличие ничьи
        return not (any(mark == EMPTY for mark in board))

    def check_finish_and_score(board, column, mark, config):
        # Возвращаем значение игры - продолжается или закончилась, и награду
        if is_win(board, column, mark, config):
            return (True, 1)
        if is_tie(board):
            return (True, 0.5)
        else:
            return (False, None)

    def uct_score(node_total_score, node_total_visits, parent_total_visits, Cp=Cp_default):
        # считаем метрику
        if node_total_visits == 0:
            return math.inf
        return node_total_score / node_total_visits + Cp * math.sqrt(
            2 * math.log(parent_total_visits) / node_total_visits)

    def opponent_mark(mark):
        # показывает какой игрок делает ход
        return 3 - mark

    def opponent_score(score):
        return 1 - score

    def random_action(board, config):
        return random.choice([c for c in range(7) if board[c] == EMPTY])

    def default_policy_simulation(board, mark, config):
        original_mark = mark
        board = board.copy()
        column = random_action(board, config)
        play(board, column, mark, config)
        is_finish, score = check_finish_and_score(board, column, mark, config)
        while not is_finish:
            mark = opponent_mark(mark)
            column = random_action(board, config)
            play(board, column, mark, config)
            is_finish, score = check_finish_and_score(board, column, mark, config)
        if mark == original_mark:
            return score
        return opponent_score(score)

    def find_action_taken_by_opponent(new_board, old_board, config):
        # Учитывая новое состояние и предыдущее находит какой ход был сделан
        for i, piece in enumerate(new_board):
            if piece != old_board[i]:
                return i % 7
        return -1  # shouldn't get here

    class State():

        def __init__(self, board, mark, config, parent=None, is_terminal=False, terminal_score=None, action_taken=None):
            self.board = board.copy()
            self.mark = mark
            self.config = config
            self.children = []
            self.parent = parent
            self.node_total_score = 0
            self.node_total_visits = 0
            self.available_moves = [c for c in range(7) if board[c] == EMPTY]
            self.expandable_moves = self.available_moves.copy()
            self.is_terminal = is_terminal
            self.terminal_score = terminal_score
            self.action_taken = action_taken

        def is_expandable(self):
            # проверяет есть ли у узла неисследованные ветки
            return (not self.is_terminal) and (len(self.expandable_moves) > 0)

        def expand_and_simulate_child(self):
            # выбираем колонку из свободных
            column = random.choice(self.expandable_moves)
            child_board = self.board.copy()
            # делаем ход
            play(child_board, column, self.mark, self.config)
            # считаем заработанные баллы
            is_terminal, terminal_score = check_finish_and_score(child_board, column, self.mark, self.config)
            # добавляем новое состояние
            self.children.append(State(child_board, opponent_mark(self.mark),
                                       self.config, parent=self,
                                       is_terminal=is_terminal,
                                       terminal_score=terminal_score,
                                       action_taken=column
                                       ))
            # производим симуляцию - запускаем игру с этого узла чтобы просчитать информацию о победителе
            simulation_score = self.children[-1].simulate()
            # распространяем информацию о сыгранной игре вверх по пройденным узлам и меняем в них значения
            self.children[-1].backpropagate(simulation_score)
            # удаляем колонку
            self.expandable_moves.remove(column)

        def choose_strongest_child(self, Cp):
            # выбираем узел с наибольшим значением UCB1
            children_scores = [uct_score(child.node_total_score,
                                         child.node_total_visits,
                                         self.node_total_visits,
                                         Cp) for child in self.children]
            max_score = max(children_scores)
            best_child_index = children_scores.index(max_score)
            return self.children[best_child_index]

        def choose_play_child(self):
            # вытаскиваем из наших узлов информацию о статистике побед
            children_scores = [child.node_total_score for child in self.children]
            # находим состояние с максимальным значением побед
            max_score = max(children_scores)
            best_child_index = children_scores.index(max_score)
            return self.children[best_child_index]

        def tree_single_run(self):
            # Заходим в один из этапов метода в зависимости от текущего состояния
            if self.is_terminal:
                self.backpropagate(self.terminal_score)
                return
            if self.is_expandable():
                self.expand_and_simulate_child()
                return
            # просчитываем дальше по найденному наилучшему узлу
            self.choose_strongest_child(Cp_default).tree_single_run()

        def simulate(self):
            """Запускаем симуляцию с текущего состояния. Если игра закончилась на этом ходе, то баллы засчитываются текущему игроку,
            если нет, то игра моделируется до конца"""
            if self.is_terminal:
                return self.terminal_score
            return opponent_score(default_policy_simulation(self.board, self.mark, self.config))

        def backpropagate(self, simulation_score):
            self.node_total_score += simulation_score
            self.node_total_visits += 1
            if self.parent is not None:
                self.parent.backpropagate(opponent_score(simulation_score))

        def choose_child_via_action(self, action):
            # Возвращает узел по действию
            for child in self.children:
                if child.action_taken == action:
                    return child
            return None

    board = observation.board
    mark = observation.mark

    # Если игра продолжается, то ищем нужное состояние чтобы сделать ход
    try:
        current_state = current_state.choose_child_via_action(
            find_action_taken_by_opponent(board, current_state.board, configuration))
        current_state.parent = None

    except:
        current_state = State(board, mark, configuration, parent=None, is_terminal=False, terminal_score=None,
                              action_taken=None)

    while time.time() - init_time <= T_max:
        current_state.tree_single_run()

    current_state = current_state.choose_play_child()
    return current_state.action_taken