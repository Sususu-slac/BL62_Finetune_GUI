from __future__ import annotations
import sys
from pathlib import Path
from PySide6.QtGui import QIcon
from PySide6.QtCore import QProcess,QTimer,Qt
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QTabWidget,QMessageBox
from .review import ReviewPage
from .train_page import TrainPage
from .training import wsl_string

STYLE='''
QWidget{background:#0b1321;color:#e5edf8;font-family:Segoe UI;font-size:14px}
QGroupBox{border:1px solid #2c3c53;border-radius:8px;margin-top:12px;padding:12px 8px 8px;font-weight:600}
QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px}
QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QPlainTextEdit,QListWidget{background:#101e30;border:1px solid #34445c;border-radius:5px;padding:6px;selection-background-color:#24735f}
QPushButton{background:#1a2b40;border:1px solid #354b68;border-radius:6px;padding:8px 12px}
QPushButton:hover{background:#263e56;border-color:#4de0bb} QPushButton:checked,QPushButton#primary{background:#207e68;border-color:#53dbb9;color:white}
QPushButton:disabled{color:#6c7b91;background:#152030} QSplitter::handle{background:#26384b}
QLabel#title{font-size:24px;font-weight:700} QLabel#state{font-size:21px;color:#72e4c4}
QTabWidget::pane{border:0} QTabBar::tab{padding:11px 30px;background:#152338;margin:2px;border-radius:5px}
QTabBar::tab:selected{background:#246e5e;color:white} QProgressBar{border:1px solid #34445c;border-radius:5px;text-align:center;min-height:22px} QProgressBar::chunk{background:#27896f}
QScrollBar:vertical{background:#0d1a2a;width:12px;margin:2px;border-radius:6px}
QScrollBar::handle:vertical{background:#3d6079;min-height:38px;border-radius:5px}
QScrollBar::handle:vertical:hover{background:#42c2a6} QScrollBar::handle:vertical:pressed{background:#68e4c8}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0px;border:0;background:none}
QScrollBar::up-arrow:vertical,QScrollBar::down-arrow:vertical{width:0px;height:0px}
QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:none}
QScrollArea{border:0} QCheckBox{spacing:8px} QToolTip{background:#20344d;color:white;border:1px solid #58708d;padding:5px}
'''
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('BL62 SAM3 Fine-tune Studio');self.resize(1560,1000);self.setWindowIcon(QIcon(str(Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent.parent))/'assets/BL62_Finetune.ico')));root=QWidget();layout=QVBoxLayout(root);header=QHBoxLayout();title=QLabel('SAM3 Fine-tune Studio');title.setObjectName('title');header.addWidget(title);header.addStretch();self.save_state=QLabel('');header.addWidget(self.save_state);header.addSpacing(16);self.runtime_info=QLabel();self.runtime_info.setAlignment(Qt.AlignRight|Qt.AlignVCenter);self.runtime_info.setTextFormat(Qt.PlainText);self.runtime_info.setTextInteractionFlags(Qt.TextSelectableByMouse);self.runtime_info.setWordWrap(True);self.runtime_info.setMaximumWidth(850);header.addWidget(self.runtime_info,1);layout.addLayout(header)
        self.tabs=QTabWidget();self.training=TrainPage();self.data=ReviewPage(lambda:self.tabs.setCurrentIndex(1));self.data.saved.connect(self.dataset_saved);self.data.status.connect(self.show_status);self.tabs.addTab(self.data,'Data');self.tabs.addTab(self.training,'Training');layout.addWidget(self.tabs,1);self.setCentralWidget(root);self.setStyleSheet(STYLE);self.statusBar().showMessage('Ready')
        for field in (self.training.distro,self.training.repository,self.training.python):field.textChanged.connect(self.update_runtime_info)
        self.update_runtime_info()
        self.toast=QLabel(self);self.toast.setStyleSheet('background:#245c50;color:white;border:1px solid #5ad7b6;border-radius:7px;padding:12px 20px');self.toast.hide();self.toast_timer=QTimer(self);self.toast_timer.setSingleShot(True);self.toast_timer.timeout.connect(self.toast.hide)
        self._close_after_stop=False;self.close_timer=QTimer(self);self.close_timer.setInterval(500);self.close_timer.timeout.connect(self.close_when_stopped)
    def update_runtime_info(self):
        python=wsl_string(Path(self.training.python.text()))
        repository=wsl_string(Path(self.training.repository.text()))
        environment=python.split('/envs/',1)[1].split('/')[0] if '/envs/' in python else 'Python'
        gui='Windows app' if getattr(sys,'frozen',False) and sys.platform=='win32' else Path(sys.prefix).name
        self.runtime_info.setText(f'GUI: {gui}  ·  Training: {environment}  ·  WSL: {self.training.distro.text()}\nPython: {python}\nSAM3: {repository}')
        self.runtime_info.setToolTip(self.runtime_info.text())
    def show_status(self,text):
        self.statusBar().showMessage(text,8000)
        if text.startswith(('Label changed','Label renamed','Label merged','Dataset saved')):
            self.toast.setText(text);self.toast.adjustSize();self.toast.move(max(12,(self.width()-self.toast.width())//2),self.height()-110);self.toast.raise_();self.toast.show();self.toast_timer.start(3500)
    def dataset_saved(self,path):self.training.set_dataset(path);self.save_state.setText('Dataset saved')
    def close_when_stopped(self):
        if not self.training.running:self.close_timer.stop();self.close()
    def closeEvent(self,event):
        if self.data.export_thread:QMessageBox.information(self,'Saving','Wait for dataset export to finish.');event.ignore();return
        if self.training.running:
            if not self._close_after_stop:
                if QMessageBox.question(self,'Training Running','Stop training safely and close when it exits?',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:event.ignore();return
                self._close_after_stop=True;self.training.stop_training();self.close_timer.start()
            event.ignore();return
        if not self.data.discard_ok():self._close_after_stop=False;event.ignore();return
        if self.training.env_process.state()!=QProcess.NotRunning:self.training.env_process.kill();self.training.env_process.waitForFinished(3000)
        self.training.save_prefs();event.accept()

def run():
    app=QApplication.instance() or QApplication(sys.argv);app.setApplicationName('BL62 Fine-tune Studio');window=MainWindow();window.show();return app.exec()
