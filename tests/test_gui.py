import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import copy,tempfile,unittest,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt,QPoint,QSettings
from PySide6.QtWidgets import QApplication,QMessageBox
from PySide6.QtTest import QTest
from finetune_gui.window import MainWindow

app=QApplication.instance() or QApplication([])
class GuiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for i in range(4):Image.new('RGB',(80,80),'gray').save(self.root/f'{i}.png')
        with patch('finetune_gui.train_page.QSettings',return_value=QSettings(str(self.root/'preferences.ini'),QSettings.IniFormat)):self.w=MainWindow()
        self.w.show();self.w.data.load(self.root);app.processEvents();self.d=self.w.data
    def tearDown(self):self.d.project.dirty=False;self.w.close();self.w.deleteLater();app.processEvents();self.temp.cleanup()
    def brush(self,start,end):
        self.d.view.set_mode('brush');a=self.d.view.mapFromScene(*start);b=self.d.view.mapFromScene(*end);QTest.mousePress(self.d.view.viewport(),Qt.LeftButton,Qt.NoModifier,a);QTest.mouseMove(self.d.view.viewport(),b,50);QTest.mouseRelease(self.d.view.viewport(),Qt.LeftButton,Qt.NoModifier,b);app.processEvents()
    def test_brush_erase_undo_switch_and_export(self):
        self.brush((30,30),(35,35));self.assertEqual(len(self.d.view.masks),1);painted=self.d.view.masks[0]['mask'].copy();self.assertTrue(painted.any());self.d.view.set_mode('erase');pos=self.d.view.mapFromScene(30,30);QTest.mouseClick(self.d.view.viewport(),Qt.LeftButton,Qt.NoModifier,pos);app.processEvents();self.assertLess(self.d.view.masks[0]['mask'].sum(),painted.sum());self.d.undo();np.testing.assert_array_equal(self.d.view.masks[0]['mask'],painted);self.d.redo();self.d.undo();self.d.assign('train');self.d.navigate(1);self.d.assign('val');self.d.navigate(-1);np.testing.assert_array_equal(self.d.view.masks[0]['mask'],painted);self.d.commit();self.d.project.export(self.root/'export');self.w.dataset_saved(str(self.root/'export'));self.assertEqual(self.w.training.dataset.text(),str(self.root/'export'))
    def test_instance_relabel_undo_and_global_rename(self):
        self.brush((30,30),(30,30));self.d.project.categories.append({'id':2,'name':'fragment','supercategory':'fragment'});self.d.refresh_categories();self.d.select(0);self.d.instance_label.setCurrentIndex(1);self.assertEqual(self.d.view.masks[0]['category_id'],2);self.d.undo();self.assertEqual(self.d.view.masks[0]['category_id'],1)
        with patch('finetune_gui.review.QInputDialog.getText',return_value=('grain',True)):self.d.rename()
        self.assertEqual(self.d.project.categories[0]['name'],'grain');self.assertEqual(self.d.view.masks[0]['noun_phrase'],'grain')
    def test_delete_redo_mask_toggle_fit(self):
        self.brush((30,30),(30,30));self.d.mask_toggle.setChecked(False);self.assertFalse(self.d.view.show_labels);self.d.fit();self.d.select(0);self.d.delete();self.assertEqual(len(self.d.view.masks),0);self.d.undo();self.assertEqual(len(self.d.view.masks),1);self.d.redo();self.assertEqual(len(self.d.view.masks),0)
    def test_random_filter_and_manual_move(self):
        self.d.scope.setCurrentIndex(0);self.d.random_split(False);self.assertTrue(all(r['split'] for r in self.d.project.images));self.d.assign('val');self.d.filter.setCurrentIndex(2);self.assertEqual(self.d.files.count(),sum(r['split']=='val' for r in self.d.project.images))
    def test_background_save_and_auto_dataset_path(self):
        self.d.project.apply_split(['train','train','val','val'])
        with patch('finetune_gui.review.QFileDialog.getExistingDirectory',return_value=str(self.root)),patch('finetune_gui.review.QInputDialog.getText',return_value=('saved_dataset',True)):
            self.d.save_dataset()
        deadline=time.time()+5
        while self.d.export_thread is not None and time.time()<deadline:app.processEvents();time.sleep(.01)
        self.assertIsNone(self.d.export_thread);self.assertTrue((self.root/'saved_dataset/train/annotations.json').is_file());self.assertEqual(self.w.training.dataset.text(),str(self.root/'saved_dataset'));self.assertFalse(self.d.project.dirty)
    def test_training_launcher_paths_and_linux_newlines(self):
        self.d.project.apply_split(['train','train','val','val']);dataset=self.root/'dataset';self.d.project.export(dataset)
        repo=self.root/'repo';(repo/'sam3/train').mkdir(parents=True);(repo/'sam3/train/train.py').write_text('')
        template=self.root/'template.yaml';template.write_text('roboflow_vl_100_root: old\nexperiment_log_dir: old\nsupercategory: old\nmax_epochs: 1\nnum_train_workers: 1\nlr_scale: 1\ntrain_batch_size: 1\nval_epoch_freq: 1\ncheckpoint_path: old\nsave_freq: 0\ngpus_per_node: 1\nuse_cluster: true\n')
        checkpoint=self.root/'weights.pt';checkpoint.write_bytes(b'fixture');out=self.root/'experiment'
        t=self.w.training;t.dataset.setText(str(dataset));t.repository.setText(str(repo));t.config.setText(str(template));t.checkpoint.setText(str(checkpoint));t.output.setText(str(out));t.gpu_ids.setText('1')
        with patch.object(t,'shell_start') as launch,patch.object(t,'save_prefs'):
            t.start_training();self.assertTrue(t.running);launch.assert_called_once()
        script=(out/'launch.sh').read_bytes();self.assertNotIn(b'\r',script);self.assertIn(b'setsid ',script);self.assertNotIn(b"'\\home",script)
        self.assertFalse(t.setup.isEnabled());self.assertEqual(t.active_settings.gpus,1);t.running=False;t.active_settings=None;t.last_output=None
    def test_polygon_merge_and_undo(self):
        self.d.view.polygon=[(2,2),(20,2),(20,20),(2,20)];self.d.view.finish_polygon();self.d.view.polygon=[(30,30),(40,30),(40,40)];self.d.view.finish_polygon();self.assertEqual(len(self.d.view.masks),2);self.d.select(0)
        with patch('finetune_gui.review.QInputDialog.getItem',return_value=('2: particle',True)):self.d.merge()
        self.assertEqual(len(self.d.view.masks),1);self.d.undo();self.assertEqual(len(self.d.view.masks),2)
if __name__=='__main__':unittest.main()
