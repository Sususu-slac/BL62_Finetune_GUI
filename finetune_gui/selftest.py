"""Packaged application smoke test; no SAM3 or user data is required."""
import json,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from .data import DataProject

def run(window,destination):
    from PySide6.QtWidgets import QApplication
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);images=root/'images';images.mkdir()
        for name in ('a.png','b.png'):Image.new('RGB',(64,64),'gray').save(images/name)
        project=DataProject.load(images);mask=np.zeros((64,64),bool);mask[8:48,8:48]=True;mask[20:30,20:30]=False
        project.commit(0,[{'category_id':1,'mask':mask}]);project.apply_split(['train','val']);project.export(root/'dataset');loaded=DataProject.load(root/'dataset');np.testing.assert_array_equal(loaded.masks(0)[0]['mask'],mask)
        window.data.load(root/'dataset');window.dataset_saved(str(root/'dataset'));window.show();QApplication.processEvents();window.data.fit();QApplication.processEvents();window.grab().save(str(destination/'packaged_data.png'));window.tabs.setCurrentIndex(1);QApplication.processEvents();window.grab().save(str(destination/'packaged_training.png'));window.data.project.dirty=False
        result={'ok':True,'pages':window.tabs.count(),'dataset_auto_loaded':window.training.dataset.text()==str(root/'dataset'),'lossless_mask_roundtrip':True}
        (destination/'packaged_test.json').write_text(json.dumps(result,indent=2));return result


def run_conversion(window,destination):
    """Exercise the bundled conversion worker on externally prepared tiny fixtures."""
    import time
    from PySide6.QtCore import QProcess,QTimer
    from PySide6.QtWidgets import QApplication,QMessageBox
    from .training import TrainingSettings,wsl_string
    destination=Path(destination);page=window.training
    if not (destination/'base.pt').is_file() or not (destination/'checkpoints/checkpoint.pt').is_file():raise ValueError('Prepare conversion fixtures first')
    page.active_settings=TrainingSettings(output=Path(wsl_string(destination)),inference_base=Path(wsl_string(destination/'base.pt')),export_inference=True)
    page.last_output=page.active_settings.output;page.running=True;window.show();window.tabs.setCurrentIndex(1)
    stages=[];notifications=[];ticks=[];original=page.conversion_line
    def line(text):
        original(text)
        if text.startswith('BL62_CONVERSION '):
            payload=json.loads(text[len('BL62_CONVERSION '):]);stages.append(payload.get('progress'))
            if payload.get('progress')==50:page.conversion_dialog.grab().save(str(destination/'conversion_progress.png'))
    page.conversion_line=line
    def tick():
        ticks.append(1)
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget,QMessageBox) and widget.isVisible():
                notifications.append(widget.text());widget.grab().save(str(destination/'conversion_complete.png'));widget.accept()
    timer=QTimer();timer.setInterval(20);timer.timeout.connect(tick);timer.start()
    try:
        page.finished(0,QProcess.NormalExit);deadline=time.monotonic()+45
        while page.converting and time.monotonic()<deadline:QApplication.processEvents();time.sleep(.005)
        if page.converting:raise TimeoutError('Conversion worker did not finish')
        assert stages==[5,20,35,50,85,100],stages
        assert notifications and notifications[-1].startswith('Inference model saved:'),notifications
        assert (destination/'checkpoints/checkpoint_sam3_format.pt').is_file()
        assert len(ticks)>3
        result={'ok':True,'stages':stages,'qt_heartbeat':len(ticks),'notification':notifications[-1],'state':page.state.text(),'worker_asset':(destination/'convert_checkpoint.py').is_file()}
        (destination/'conversion_test.json').write_text(json.dumps(result,indent=2));return result
    finally:timer.stop();page.conversion_line=original
