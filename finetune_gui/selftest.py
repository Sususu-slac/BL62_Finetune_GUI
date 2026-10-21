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
