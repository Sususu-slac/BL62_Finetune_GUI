import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile,unittest,json,time,sys
from pathlib import Path
from unittest.mock import patch
import torch
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication
from finetune_gui.convert_checkpoint import convert
from finetune_gui.train_page import TrainPage
from finetune_gui.training import TrainingSettings
app=QApplication.instance() or QApplication([])

class ConversionTests(unittest.TestCase):
    def test_export_preserves_trained_weights_and_tracker_without_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base.pt';train=root/'checkpoint.pt';out=root/'inference.pt'
            torch.save({'detector.layer':torch.zeros(3),'tracker.layer':torch.ones(2),'detector.backbone.vision_backbone.sam2_convs.a':torch.ones(1)},base)
            torch.save({'model':{'layer':torch.full((3,),7.)},'optimizer':{'state':torch.ones(30)},'epoch':2},train)
            original=train.read_bytes();report=convert(base,train,out)
            saved=torch.load(out,weights_only=True);self.assertEqual(set(saved),{'detector.layer','tracker.layer','detector.backbone.vision_backbone.sam2_convs.a'});self.assertTrue(torch.equal(saved['detector.layer'],torch.full((3,),7.)));self.assertTrue(torch.equal(saved['tracker.layer'],torch.ones(2)))
            self.assertEqual(train.read_bytes(),original);self.assertEqual(report['matched'],1);self.assertTrue(out.with_suffix('.report.json').is_file())
            with self.assertRaisesRegex(ValueError,'already exists'):convert(base,train,out)
    def test_mismatch_missing_weights_and_write_failure_publish_no_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base.pt';train=root/'train.pt';out=root/'out.pt'
            torch.save({'detector.a':torch.zeros(2),'detector.b':torch.zeros(1)},base)
            for weights in ({'a':torch.ones(3),'b':torch.ones(1)},{'a':torch.ones(2)},{'wrong':torch.ones(2)}):
                torch.save({'model':weights},train)
                with self.assertRaisesRegex(ValueError,'Incompatible'):convert(base,train,out)
                self.assertFalse(out.exists())
            torch.save({'model':{'a':torch.ones(2),'b':torch.ones(1)}},train)
            with patch('torch.save',side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError,'disk full'):convert(base,train,out)
            self.assertFalse(out.exists());self.assertFalse(list(root.glob('*.tmp')))
    def test_fallback_template_and_asset_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base.pt';template=root/'sam3.pt';out=root/'out.pt'
            torch.save({'model':{'a':torch.ones(2)}},base);torch.save({'detector.a':torch.zeros(2),'tracker.a':torch.ones(1)},template)
            report=convert(base,base,out,template);self.assertEqual(report['base'],str(template))
        repo=Path(__file__).resolve().parent.parent
        self.assertEqual((repo/'assets/convert_checkpoint.py').read_bytes(),(repo/'finetune_gui/convert_checkpoint.py').read_bytes())
    def test_success_starts_conversion_without_confirmation_and_locks_setup(self):
        page=TrainPage()
        with tempfile.TemporaryDirectory() as tmp:
            page.active_settings=TrainingSettings(output=Path(tmp),export_inference=True);page.last_output=Path(tmp);page.running=True
            try:
                with patch.object(page,'shell_start') as launch,patch('finetune_gui.train_page.QMessageBox.information') as info:
                    page.finished(0,QProcess.NormalExit);self.assertTrue(page.converting);info.assert_not_called();launch.assert_called_once()
                    self.assertEqual(page.conversion_dialog.message.text(),'Training complete. Converting…');self.assertFalse(page.start.isEnabled());self.assertTrue(page.open_btn.isEnabled())
                    page.conversion_dialog.reject();self.assertTrue(page.conversion_dialog.isVisible())
                    page.conversion_line('BL62_CONVERSION '+json.dumps({'progress':50,'message':'Writing inference model…'}));self.assertEqual(page.conversion_dialog.progress.value(),50)
                    (Path(tmp)/'checkpoints').mkdir();page.local_path(str(page.inference_output)).write_bytes(b'fixture')
                    page.conversion_finished(0,QProcess.NormalExit);self.assertFalse(page.converting);self.assertTrue(page.start.isEnabled());self.assertIsNone(page.conversion_dialog);info.assert_called_once();self.assertIn('Inference Model:',page.artifacts.text())
            finally:page.gpu_status.stop();page.close();page.deleteLater();app.processEvents()
    def test_real_worker_process_reports_progress_and_keeps_ui_responsive(self):
        from finetune_gui.training import TRAIN_PYTHON
        from PySide6.QtCore import QTimer
        page=TrainPage()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoints').mkdir();base=root/'base.pt'
            torch.save({'detector.a':torch.zeros(2),'tracker.a':torch.ones(1)},base)
            torch.save({'model':{'a':torch.full((2,),3.)},'optimizer':{},'epoch':1},root/'checkpoints/checkpoint.pt')
            page.active_settings=TrainingSettings(output=root,python=TRAIN_PYTHON if os.name=='nt' else Path(sys.executable),inference_base=base,export_inference=True);page.last_output=root;page.running=True
            ticks=[];pulse=QTimer();pulse.setInterval(10);pulse.timeout.connect(lambda:ticks.append(1));pulse.start()
            try:
                with patch('finetune_gui.train_page.QMessageBox.information') as info,patch('finetune_gui.train_page.QMessageBox.warning') as warning:
                    page.finished(0,QProcess.NormalExit);deadline=time.monotonic()+30
                    while page.converting and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
                    self.assertFalse(page.converting,page.log.toPlainText());warning.assert_not_called();info.assert_called_once();self.assertGreater(len(ticks),3)
                    result=torch.load(root/'checkpoints/checkpoint_sam3_format.pt',weights_only=True)
                    self.assertTrue(torch.equal(result['detector.a'],torch.full((2,),3.)));self.assertIn('Verifying saved model',page.log.toPlainText())
            finally:
                pulse.stop()
                if page.conversion_process.state()!=QProcess.NotRunning:page.conversion_process.kill();page.conversion_process.waitForFinished(3000)
                page.gpu_status.stop();page.close();page.deleteLater();app.processEvents()
    def test_failure_preserves_training_result_and_stopped_runs_do_not_convert(self):
        page=TrainPage()
        with tempfile.TemporaryDirectory() as tmp:
            page.active_settings=TrainingSettings(output=Path(tmp),export_inference=True);page.last_output=Path(tmp)
            try:
                for code,stopping in ((1,False),(0,True)):
                    page.running=True;page.stopping=stopping
                    with patch.object(page,'begin_conversion') as begin:page.finished(code,QProcess.NormalExit);begin.assert_not_called()
                page.stopping=False
                with patch.object(page,'shell_start'),patch('finetune_gui.train_page.QMessageBox.warning') as warning:
                    page.begin_conversion();page.conversion_error='disk full';page.conversion_finished(1,QProcess.NormalExit)
                    self.assertEqual(page.state.text(),'Training complete. Conversion failed');self.assertFalse(page.converting);self.assertTrue(page.start.isEnabled());self.assertIn('preserved',warning.call_args.args[2])
            finally:page.gpu_status.stop();page.close();page.deleteLater();app.processEvents()
if __name__=='__main__':unittest.main()
