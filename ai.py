"""A simple artificial intelligence written in Python."""


import sys
import os


from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMenu, QMenuBar,
                                QAction, QFileDialog, QMessageBox, QDesktopWidget,
                                QStatusBar, QToolBar, QWidget, QGridLayout,
                                QPushButton, QLabel, QSizePolicy, QSpacerItem,
                                QStyle, QStyleOption, QStylePainter)

class HexGame(QMainWindow):
    
        def __init__(self, parent=None):
            super(HexGame, self).__init__(parent)
    
            self.initUI()
    
        def initUI(self):
            self.setWindowTitle('Hex Game')
            self.setWindowIcon(QIcon('hex.png'))
            self.setFixedSize(600, 600)
    
            self.statusBar = QStatusBar()
            self.setStatusBar(self.statusBar)
    
            self.center()
            self.show()
    
        def center(self):
            frame = self.frameGeometry()
            centerPoint = QDesktopWidget().availableGeometry().center()
            frame.moveCenter(centerPoint)
            self.move(frame.topLeft())
    
        def closeEvent(self, event):
            reply = QMessageBox.question(self, 'Message',
                "Are you sure to quit?", QMessageBox.Yes |
                QMessageBox.No, QMessageBox.No)
    
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    hexgame = HexGame()
    sys.exit(app.exec_())
