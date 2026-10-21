"""Local integration test; writes only to a separate test directory."""
import os,sys,time,json
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication,QMessageBox
from finetune_gui.data import DataProject
from finetune_gui.window import MainWindow
app=QApplication([]);w=MainWindow();w.show();app.processEvents()
root=Path('D:/BL62_Finetune_Test_20261020');root.mkdir(exist_ok=True)
source=Path('C:/Users/subo/Desktop/Science_data/Labeled_data/sample_2/training_dataset')
p=DataProject.load(source);p.images=p.images[:4];p.apply_split(['train','train','val','val'])
dataset=root/'dataset'
if not dataset.exists():p.export(dataset)
t=w.training;t.dataset.setText(str(dataset));t.gpu_ids.setText('1');t.epochs.setValue(1);t.workers.setValue(0);t.val_every.setValue(1);run=root/('train_smoke_'+time.strftime('%H%M%S'));t.output.setText(str(run))
errors=[]
def warning(parent,title,text,*args):errors.append(title+': '+str(text));print(errors[-1],flush=True);return QMessageBox.Ok
QMessageBox.warning=warning
print('VALIDATING',flush=True);t.validate_settings();t.check_environment();deadline=time.time()+90
while t.env_process.state()!=QProcess.NotRunning and time.time()<deadline:app.processEvents();time.sleep(.05)
print(t.log.toPlainText(),flush=True)
if t.env_process.state()!=QProcess.NotRunning or t.env_process.exitCode()!=0:raise RuntimeError('Environment check failed')
print('START TRAINING',flush=True);t.start_training();deadline=time.time()+900
while t.running and time.time()<deadline:
 app.processEvents();time.sleep(.1)
 (root/'training_log.txt').write_text(t.log.toPlainText(),encoding='utf-8')
if t.running:t.stop_training();raise RuntimeError('Training timed out')
(root/'training_log.txt').write_text(t.log.toPlainText(),encoding='utf-8')
result={'state':t.state.text(),'exit_code':t.process.exitCode(),'errors':errors,'checkpoint':(run/'checkpoints/checkpoint.pt').is_file(), 'output':str(run)}
(root/'training_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);w.data.project.dirty=False;w.close()
if errors or result['exit_code']!=0 or not result['checkpoint']:raise RuntimeError('Training integration failed; inspect training_log.txt')
