# -*- coding: utf-8 -*-
# @Author : CatfishW🚀
# @Time : 2023/5/1
import sys
from PySide6.QtWidgets import QApplication, QWidget
from utils.id_dialog import id_form


class id_Window(QWidget, id_form):
    def __init__(self):
        super(id_Window, self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = id_Window()
    window.show()
    sys.exit(app.exec())
