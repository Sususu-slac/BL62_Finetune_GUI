import os,sys,traceback
from pathlib import Path
from finetune_gui.window import run

def report_error(kind,value,tb):
    folder=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'BL62_Finetune_GUI';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'error.log').open('a',encoding='utf-8') as f:traceback.print_exception(kind,value,tb,file=f)
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.critical(None,'BL62 Fine-tune GUI',str(value)+'\nDetails: '+str(folder/'error.log'))

if __name__=='__main__':
    sys.excepthook=report_error
    if len(sys.argv)>2 and sys.argv[1]=='--self-test':
        from PySide6.QtWidgets import QApplication
        from finetune_gui.window import MainWindow
        from finetune_gui.selftest import run as selftest
        app=QApplication([]);window=MainWindow()
        try:selftest(window,sys.argv[2]);window.close()
        except Exception:
            target=Path(sys.argv[2]);target.mkdir(parents=True,exist_ok=True);(target/'packaged_error.txt').write_text(traceback.format_exc());sys.exit(1)
    else:raise SystemExit(run())
