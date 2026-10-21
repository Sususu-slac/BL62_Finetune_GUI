"""Exercise the actual launcher and safe-stop group without loading a model."""
import os,sys,time,tempfile,unittest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from unittest.mock import patch
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from finetune_gui.train_page import TrainPage
from finetune_gui.training import TrainingSettings,wsl_string,TRAIN_PYTHON
app=QApplication.instance() or QApplication([])

class ProcessControlTests(unittest.TestCase):
    def test_start_stop_only_own_process_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);repo=root/'repo';out=root/'output';repo.mkdir();template=root/'template.yaml';template.write_text('roboflow_vl_100_root: old\nexperiment_log_dir: old\nsupercategory: old\nmax_epochs: 1\nnum_train_workers: 1\nlr_scale: 1\ntrain_batch_size: 1\nval_epoch_freq: 1\ncheckpoint_path: old\nsave_freq: 0\ngpus_per_node: 1\nuse_cluster: true\n')
            fake=root/'fake_training.py';fake.write_text("import time, pathlib, signal, sys\npathlib.Path(sys.argv[1]).touch()\nwhile True: time.sleep(.05)\n")
            with patch('finetune_gui.train_page.QSettings',return_value=QSettings(str(root/'prefs.ini'),QSettings.IniFormat)):page=TrainPage()
            interpreter=TRAIN_PYTHON if os.name=='nt' else Path(sys.executable)
            settings=TrainingSettings(config=template,output=out,repository=repo,python=interpreter,gpus=1)
            page.gpu_ids.setText('1');ready=root/'ready'
            fake_command=[wsl_string(interpreter),wsl_string(fake),wsl_string(ready),'-c','config.yaml']
            with patch.object(page,'validate_settings',return_value=settings),patch.object(page,'env_script',return_value='true'),patch.object(page,'save_prefs'),patch('finetune_gui.train_page.command',return_value=fake_command):page.start_training()
            deadline=time.time()+8
            try:
                while not ready.exists() and time.time()<deadline:app.processEvents();time.sleep(.02)
                self.assertTrue(ready.exists(),page.log.toPlainText());pid=int((out/'gui_training.pid').read_text())
                if os.name!='nt':self.assertEqual(os.getpgid(pid),pid)
                page.stop_training()
                while page.running and time.time()<deadline:app.processEvents();time.sleep(.02)
                self.assertFalse(page.running,page.log.toPlainText());self.assertEqual(page.state.text(),'Stopped');self.assertFalse((out/'gui_training.pid').exists())
            finally:
                if page.running:
                    try:
                        if os.name!='nt':os.killpg(pid,15)
                        else:page.shell_start(page.helper,'kill -TERM -- -'+str(pid));page.helper.waitForFinished(3000)
                    except ProcessLookupError:pass
                    page.process.waitForFinished(3000)
                page.timer.stop();page.stop_timer.stop();page.deleteLater();app.processEvents()
if __name__=='__main__':unittest.main()
