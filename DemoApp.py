import sys
from DemoInterface import Ui_MainWindow
from PyQt5 import QtWidgets
from kaggle_environments import make
from Minimax import agent
#from MCTS import MCTS_agent as agent


class App(QtWidgets.QMainWindow):

    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.env = make("connectx", debug=True)
        self.push_buttons()
        self.init_columns()
        self.trainer = self.env.train([None, agent])
        self.observation = self.trainer.reset()
        self.player_action = None

    def init_columns(self):
        self.column_0 = [self.ui.pushButton_0_0, self.ui.pushButton_0_1, self.ui.pushButton_0_2,
                         self.ui.pushButton_0_3, self.ui.pushButton_0_4, self.ui.pushButton_0_5]
        self.column_1 = [self.ui.pushButton_1_0, self.ui.pushButton_1_1, self.ui.pushButton_1_2,
                         self.ui.pushButton_1_3, self.ui.pushButton_1_4, self.ui.pushButton_1_5]
        self.column_2 = [self.ui.pushButton_2_0, self.ui.pushButton_2_1, self.ui.pushButton_2_2,
                         self.ui.pushButton_2_3, self.ui.pushButton_2_4, self.ui.pushButton_2_5]
        self.column_3 = [self.ui.pushButton_3_0, self.ui.pushButton_3_1, self.ui.pushButton_3_2,
                         self.ui.pushButton_3_3, self.ui.pushButton_3_4, self.ui.pushButton_3_5]
        self.column_4 = [self.ui.pushButton_4_0, self.ui.pushButton_4_1, self.ui.pushButton_4_2,
                         self.ui.pushButton_4_3, self.ui.pushButton_4_4, self.ui.pushButton_4_5]
        self.column_5 = [self.ui.pushButton_5_0, self.ui.pushButton_5_1, self.ui.pushButton_5_2,
                         self.ui.pushButton_5_3, self.ui.pushButton_5_4, self.ui.pushButton_5_5]
        self.column_6 = [self.ui.pushButton_6_0, self.ui.pushButton_6_1, self.ui.pushButton_6_2,
                         self.ui.pushButton_6_3, self.ui.pushButton_6_4, self.ui.pushButton_6_5]

        self.columns = [self.column_0, self.column_1, self.column_2, self.column_3,
                        self.column_4, self.column_5, self.column_6]

    def push_buttons(self):
        self.ui.pushButton_43.clicked.connect(lambda: self.column_choose(0))
        self.ui.pushButton_44.clicked.connect(lambda: self.column_choose(1))
        self.ui.pushButton_45.clicked.connect(lambda: self.column_choose(2))
        self.ui.pushButton_46.clicked.connect(lambda: self.column_choose(3))
        self.ui.pushButton_47.clicked.connect(lambda: self.column_choose(4))
        self.ui.pushButton_48.clicked.connect(lambda: self.column_choose(5))
        self.ui.pushButton_49.clicked.connect(lambda: self.column_choose(6))
        self.ui.pushButton_50.clicked.connect(lambda: self.clear_table())

    def column_choose(self, column_number):
        old_observation = self.observation
        self.player_action = column_number
        self.columns[column_number][0].setStyleSheet("background: #0000fe;")
        self.columns[column_number].pop(0)
        self.observation, reward, done, info = self.trainer.step(self.player_action)

        agent_action = -1
        for i in range(len(self.observation.board)):
            if old_observation.board[i] != self.observation.board[i] and self.observation.board[i] == 2:
                agent_action = i % 7
        self.columns[agent_action][0].setStyleSheet("background: #fe0000;")
        self.columns[agent_action].pop(0)

        if reward == 0:
            QtWidgets.QMessageBox.about(self, ' ', 'Вы проиграли(')
            self.clear_table()
        elif reward == 1:
            QtWidgets.QMessageBox.about(self, ' ', 'Вы выиграли!!')
            self.clear_table()
        elif done:
            QtWidgets.QMessageBox.about(self, ' ', 'Ничья -_-')
            self.clear_table()

    def clear_table(self):
        self.init_columns()
        for column in self.columns:
            for button in column:
                button.setStyleSheet("background: #fff;")

        self.trainer = self.env.train([None, agent])
        self.observation = self.trainer.reset()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    application = App()
    application.show()
    sys.exit(app.exec_())
