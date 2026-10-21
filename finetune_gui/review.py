from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QWidget,QHBoxLayout,QVBoxLayout,QFormLayout,QGroupBox,QSplitter,QPushButton,QLabel,QComboBox,QListWidget,QListWidgetItem,QSpinBox,QCheckBox,QFileDialog,QMessageBox,QInputDialog,QDialog,QDialogButtonBox,QProgressDialog,QButtonGroup,QScrollArea)
from .data import DataProject
from .canvas import AnnotationView

class ExportWorker(QObject):
    done=Signal(str);failed=Signal(str);progress=Signal(int,int)
    def __init__(self,project,path): super().__init__();self.project=copy.deepcopy(project);self.path=path
    def run(self):
        try: self.done.emit(str(self.project.export(self.path,self.progress.emit)))
        except Exception as e: self.failed.emit(str(e))

class ReviewPage(QWidget):
    saved=Signal(str);status=Signal(str)
    def __init__(self,training_callback):
        super().__init__();self.project=DataProject();self.current=-1;self.undo_stack=[];self.redo_stack=[];self.export_thread=None;self.pending_plan=None;self._updating=False;self.image_dirty=False
        root=QHBoxLayout(self);root.setContentsMargins(0,0,0,0);split=QSplitter();root.addWidget(split)
        left=QWidget();left.setMinimumWidth(310);shell=QVBoxLayout(left);settings=QWidget();lv=QVBoxLayout(settings);scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(settings);shell.addWidget(scroll,1)
        source=QGroupBox('Dataset');sl=QVBoxLayout(source);row=QHBoxLayout()
        for title,fn in [('Open Folder',self.open_folder),('Open JSON File',self.open_json)]: b=QPushButton(title);b.clicked.connect(fn);row.addWidget(b)
        sl.addLayout(row);self.source_label=QLabel('No dataset');self.source_label.setWordWrap(True);sl.addWidget(self.source_label)
        self.filter=QComboBox();self.filter.addItems(['All Images','Training','Validation','Unassigned']);self.filter.currentIndexChanged.connect(self.populate);sl.addWidget(self.filter)
        self.counts=QLabel();self.counts.setWordWrap(True);sl.addWidget(self.counts);self.files=QListWidget();self.files.currentItemChanged.connect(self.choose_item);sl.addWidget(self.files,1);lv.addWidget(source,1)
        labels=QGroupBox('Labels');ll=QHBoxLayout(labels);self.global_label=QComboBox();ll.addWidget(self.global_label,1)
        rename=QPushButton('Rename');rename.clicked.connect(self.rename);ll.addWidget(rename);lv.addWidget(labels)
        random=QGroupBox('Random Split');rl=QFormLayout(random);self.scope=QComboBox();self.scope.addItems(['All Images','Unassigned Only']);self.scope.setCurrentIndex(1);rl.addRow('Scope',self.scope)
        self.train_pct=QSpinBox();self.train_pct.setRange(1,99);self.train_pct.setValue(80);self.val_pct=QSpinBox();self.val_pct.setRange(1,99);self.val_pct.setValue(20)
        self.train_pct.valueChanged.connect(lambda n:self.val_pct.setValue(100-n));self.val_pct.valueChanged.connect(lambda n:self.train_pct.setValue(100-n));rl.addRow('Train %',self.train_pct);rl.addRow('Validation %',self.val_pct)
        self.seed=QSpinBox();self.seed.setRange(0,2147483647);self.seed.setValue(42);rl.addRow('Seed',self.seed)
        rr=QHBoxLayout();preview=QPushButton('Preview Split');preview.clicked.connect(lambda:self.random_split(True));apply=QPushButton('Apply Split');apply.clicked.connect(lambda:self.random_split(False));rr.addWidget(preview);rr.addWidget(apply);rl.addRow(rr);lv.addWidget(random)
        bottom=QHBoxLayout();self.save_button=QPushButton('Save Dataset');self.save_button.setObjectName('primary');self.save_button.clicked.connect(self.save_dataset);bottom.addWidget(self.save_button);go=QPushButton('Training →');go.clicked.connect(training_callback);bottom.addWidget(go);shell.addLayout(bottom);split.addWidget(left)
        right=QWidget();rv=QVBoxLayout(right);self.image_name=QLabel('Open a dataset to review');rv.addWidget(self.image_name)
        tools=QHBoxLayout();self.tool_group=QButtonGroup(self)
        for label,mode in [('Select','select'),('Pan','pan'),('Brush','brush'),('Eraser','erase'),('Polygon','polygon')]:
            button=QPushButton(label);button.setCheckable(True);button.clicked.connect(lambda checked=False,m=mode:self.view.set_mode(m));self.tool_group.addButton(button);tools.addWidget(button)
            if mode=='select': self.select_tool=button;button.setChecked(True)
        self.size=QSpinBox();self.size.setRange(1,500);self.size.setValue(20);self.size.setPrefix('Size ');tools.addWidget(self.size);tools.addWidget(QLabel('Label'));self.instance_label=QComboBox();self.instance_label.setMinimumWidth(110);self.instance_label.currentIndexChanged.connect(self.change_label);tools.addWidget(self.instance_label,1);rv.addLayout(tools)
        actions=QHBoxLayout()
        for name,callback in [('Undo',self.undo),('Redo',self.redo),('Delete',self.delete),('Merge',self.merge),('Fit',self.fit)]: button=QPushButton(name);button.clicked.connect(callback);actions.addWidget(button)
        actions.addStretch();rv.addLayout(actions)
        self.view=AnnotationView();self.view.set_mode('select');self.view.changed.connect(self.push_undo);self.view.selection_requested.connect(self.select);self.view.masks_changed.connect(self.masks_changed);self.view.status.connect(self.status.emit);self.view.tool_reset_requested.connect(self.reset_tool);self.size.valueChanged.connect(self.view.set_brush_size);rv.addWidget(self.view,1)
        footer=QHBoxLayout();self.mask_toggle=QCheckBox('Masks');self.mask_toggle.setChecked(True);self.mask_toggle.toggled.connect(self.view.set_labels_visible);footer.addWidget(self.mask_toggle)
        self.prev=QPushButton('◀');self.prev.clicked.connect(lambda:self.navigate(-1));footer.addWidget(self.prev);self.position=QLabel('0 / 0');footer.addWidget(self.position);self.next=QPushButton('▶');self.next.clicked.connect(lambda:self.navigate(1));footer.addWidget(self.next);footer.addStretch()
        self.assign_train=QPushButton('Training');self.assign_train.setCheckable(True);self.assign_train.clicked.connect(lambda:self.assign('train'));self.assign_val=QPushButton('Validation');self.assign_val.setCheckable(True);self.assign_val.clicked.connect(lambda:self.assign('val'));footer.addWidget(self.assign_train);footer.addWidget(self.assign_val);rv.addLayout(footer);split.addWidget(right);split.setSizes([360,1100])
        for key,fn in [('Ctrl+Z',self.undo),('Ctrl+Shift+Z',self.redo),('Delete',self.delete),('Ctrl+S',self.save_dataset)]:
            a=QAction(self);a.setShortcut(QKeySequence(key));a.setShortcutContext(Qt.WidgetWithChildrenShortcut);a.triggered.connect(fn);self.addAction(a)
        self.populate()

    def reset_tool(self): self.select_tool.setChecked(True);self.view.set_mode('select')
    def notify(self,text): self.status.emit(text)
    def error(self,e): QMessageBox.warning(self,'Data',str(e))
    def discard_ok(self):
        return not self.project.dirty or QMessageBox.question(self,'Unsaved dataset','Discard unsaved changes?',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)==QMessageBox.Yes
    def open_folder(self):
        p=QFileDialog.getExistingDirectory(self,'Open Dataset Folder')
        if p:self.load(p)
    def open_json(self):
        p,_=QFileDialog.getOpenFileName(self,'Open JSON File','','JSON files (*.json)')
        if p:self.load(p)
    def load(self,path):
        if not self.discard_ok(): return
        try: project=DataProject.load(path)
        except Exception as e:self.error(e);return
        self.project=project;self.current=-1;self.image_dirty=False;self.undo_stack=[];self.redo_stack=[];self.source_label.setText(str(path));self.refresh_categories();self.populate();self.show_image(0)
        if project.saved_path:self.saved.emit(str(project.saved_path))
    def commit(self):
        if self.current>=0 and self.image_dirty:self.project.commit(self.current,self.view.masks);self.image_dirty=False
    def refresh_categories(self):
        self._updating=True
        for combo in (self.global_label,self.instance_label):
            value=combo.currentData();combo.clear()
            for cat in self.project.categories:combo.addItem(cat['name'],cat['id'])
            if combo.findData(value)>=0:combo.setCurrentIndex(combo.findData(value))
        self._updating=False;self.update_metadata()
    def update_metadata(self):
        cid=self.instance_label.currentData()
        if cid is not None:self.view.new_mask_metadata={'category_id':cid,'noun_phrase':self.instance_label.currentText(),'annotation_source':'manual'}
    def visible(self):
        value=['all','train','val',''][self.filter.currentIndex()]
        return [i for i,r in enumerate(self.project.images) if value=='all' or r['split']==value]
    def populate(self):
        self.files.blockSignals(True);self.files.clear()
        for i in self.visible():
            rec=self.project.images[i];tag={'train':'Train','val':'Validate','':'Unassigned'}[rec['split']];item=QListWidgetItem(f"{rec['name']}  [{tag}]");item.setData(Qt.UserRole,i);self.files.addItem(item)
            if i==self.current:self.files.setCurrentItem(item)
        self.files.blockSignals(False)
        counts={s:sum(r['split']==s for r in self.project.images) for s in ('train','val','')};self.counts.setText(f"{len(self.project.images)} images · Train {counts['train']} · Validate {counts['val']} · Unassigned {counts['']}")
    def choose_item(self,item,previous):
        if item:self.show_image(item.data(Qt.UserRole))
    def show_image(self,index):
        if index==self.current or not 0<=index<len(self.project.images):return
        self.commit();rec=self.project.images[index]
        try:
            with Image.open(rec['path']) as im: array=np.asarray(im.convert('RGB')).copy()
            masks=self.project.masks(index)
        except Exception as e:self.error(e);return
        self.current=index;self.undo_stack=[];self.redo_stack=[];self.image_dirty=False;self.view.set_image(array,masks);self.reset_tool();self.select(-1);self.populate();self.update_footer()
    def update_footer(self):
        if self.current<0:return
        rec=self.project.images[self.current];self.image_name.setText(f"{rec['name']}  ·  {rec['width']} × {rec['height']}  ·  {len(self.view.masks)} instances");self.position.setText(f'{self.current+1} / {len(self.project.images)}');self.assign_train.setChecked(rec['split']=='train');self.assign_val.setChecked(rec['split']=='val')
    def navigate(self,step):
        indices=self.visible()
        if not indices:return
        pos=indices.index(self.current) if self.current in indices else (-1 if step>0 else len(indices))
        target=pos+step
        if 0<=target<len(indices):self.show_image(indices[target])
    def assign(self,split):
        if self.current<0:return
        self.project.images[self.current]['split']=split;self.project.dirty=True;self.populate();self.update_footer()
    def select(self,index):
        self.view.selected=index
        if 0<=index<len(self.view.masks):
            self._updating=True;self.instance_label.setCurrentIndex(self.instance_label.findData(self.view.masks[index]['category_id']));self._updating=False
        self.update_metadata();self.view.redraw()
    def push_undo(self):
        self.undo_stack.append((copy.deepcopy(self.view.masks),self.view.selected));self.undo_stack=self.undo_stack[-30:];self.redo_stack=[];self.image_dirty=True;self.project.dirty=True
    def restore(self,source,target):
        if not source or self.current<0:return
        target.append((copy.deepcopy(self.view.masks),self.view.selected));masks,selected=source.pop();transform=self.view.transform();self.view.set_image(self.view.array,masks);self.view.setTransform(transform);self.select(selected);self.image_dirty=True;self.project.dirty=True;self.update_footer()
    def undo(self):self.restore(self.undo_stack,self.redo_stack)
    def redo(self):self.restore(self.redo_stack,self.undo_stack)
    def masks_changed(self):self.image_dirty=True;self.project.dirty=True;self.update_footer()
    def change_label(self):
        if self._updating:return
        self.update_metadata();i=self.view.selected
        if 0<=i<len(self.view.masks):
            cid=self.instance_label.currentData()
            if cid==self.view.masks[i]['category_id']:return
            self.push_undo();self.view.masks[i]['category_id']=cid;self.view.masks[i]['noun_phrase']=self.instance_label.currentText();self.notify('Label changed to '+self.instance_label.currentText())
    def rename(self):
        cid=self.global_label.currentData()
        if cid is None:return
        name,ok=QInputDialog.getText(self,'Rename Label','Name',text=self.global_label.currentText())
        if not ok:return
        try:
            self.commit();self.project.rename(cid,name)
            for m in self.view.masks:
                if m['category_id']==cid:m['noun_phrase']=name.strip()
            self.undo_stack=[];self.redo_stack=[];self.refresh_categories();self.notify('Label renamed globally to '+name.strip())
        except Exception as e:self.error(e)
    def delete(self):
        i=self.view.selected
        if 0<=i<len(self.view.masks):self.push_undo();self.view.masks.pop(i);self.select(-1);self.update_footer()
    def merge(self):
        if self.current<0 or len(self.view.masks)<2:return
        first=self.view.selected
        if first<0:self.notify('Select an instance first');return
        values=[f"{i+1}: {next(c['name'] for c in self.project.categories if c['id']==m['category_id'])}" for i,m in enumerate(self.view.masks) if i!=first]
        name,ok=QInputDialog.getItem(self,'Merge Instances','Merge selected instance with',values,0,False)
        if not ok:return
        other=int(name.split(':')[0])-1;self.push_undo();self.view.masks[first]['mask']=self.view.masks[first]['mask']|self.view.masks[other]['mask'];self.view.masks.pop(other);self.select(first-(other<first));self.update_footer()
    def fit(self):
        if self.view.image_item:self.view.fitInView(self.view.image_item,Qt.KeepAspectRatio)
    def random_split(self,preview):
        try:plan=self.project.plan_split('all' if self.scope.currentIndex()==0 else 'unassigned',self.train_pct.value(),self.seed.value())
        except Exception as e:self.error(e);return
        if preview:
            dlg=QDialog(self);dlg.setWindowTitle('Preview Split');lay=QVBoxLayout(dlg);lay.addWidget(QLabel(f"Train {plan.count('train')} · Validate {plan.count('val')} · Unassigned {plan.count('')}"));listing=QListWidget();lay.addWidget(listing)
            for rec,value in zip(self.project.images,plan):listing.addItem(f"{rec['name']}  [{'Train' if value=='train' else 'Validate' if value=='val' else 'Unassigned'}]")
            buttons=QDialogButtonBox(QDialogButtonBox.Apply|QDialogButtonBox.Cancel);buttons.button(QDialogButtonBox.Apply).clicked.connect(dlg.accept);buttons.rejected.connect(dlg.reject);lay.addWidget(buttons);dlg.resize(550,440)
            if dlg.exec()!=QDialog.Accepted:return
        elif self.scope.currentIndex()==0 and any(r['split'] for r in self.project.images):
            if QMessageBox.question(self,'Random Split','Replace existing assignments for all images?',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:return
        self.project.apply_split(plan);self.populate();self.update_footer()
    def save_dataset(self):
        if self.export_thread or not self.project.images:return
        self.commit()
        n=sum(not r['split'] for r in self.project.images)
        if n and QMessageBox.question(self,'Unassigned Images',f'{n} images are unassigned. Save them in the unassigned folder?',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:return
        parent=QFileDialog.getExistingDirectory(self,'Select Save Location')
        if not parent:return
        name,ok=QInputDialog.getText(self,'Save Dataset','Dataset folder name',text='finetune_dataset')
        if not ok or not name.strip():return
        if Path(name).name!=name or name in ('.','..') or any(c in name for c in '<>:"/\\|?*'):self.error('Enter a folder name without path separators');return
        path=Path(parent)/name
        self.export_thread=QThread(self);self.export_worker=ExportWorker(self.project,path);self.export_worker.moveToThread(self.export_thread)
        self.export_dialog=QProgressDialog('Saving dataset…','',0,len(self.project.images),self);self.export_dialog.setWindowModality(Qt.WindowModal);self.export_dialog.setAutoClose(False);self.export_dialog.show();self.setEnabled(False)
        self.export_thread.started.connect(self.export_worker.run);self.export_worker.progress.connect(lambda done,total:self.export_dialog.setValue(done));self.export_worker.done.connect(self.export_done);self.export_worker.failed.connect(self.export_failed)
        self.export_worker.done.connect(self.export_worker.deleteLater);self.export_worker.failed.connect(self.export_worker.deleteLater);self.export_worker.done.connect(self.export_thread.quit);self.export_worker.failed.connect(self.export_thread.quit);self.export_thread.finished.connect(self.export_cleanup);self.export_thread.start()
    def export_done(self,path):self.project.saved_path=Path(path);self.project.dirty=False;self.saved.emit(path);self.notify('Dataset saved: '+path)
    def export_failed(self,message):self.error(message)
    def export_cleanup(self):self.export_dialog.close();self.setEnabled(True);self.export_thread.deleteLater();self.export_thread=None
