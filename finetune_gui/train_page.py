from __future__ import annotations
import json
from dataclasses import replace
import os
import shlex
import signal
import sys
import uuid
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Qt,QProcess,QTimer,QSettings
from PySide6.QtGui import QColor,QPainter,QPen
from PySide6.QtWidgets import (QWidget,QHBoxLayout,QVBoxLayout,QFormLayout,QGroupBox,QSplitter,QPushButton,QLabel,QComboBox,QLineEdit,QSpinBox,QDoubleSpinBox,QPlainTextEdit,QProgressBar,QFileDialog,QMessageBox,QScrollArea,QCheckBox,QDialog)
from .gpu_monitor import GpuStatusPanel
from . import convert_checkpoint
from .training import TrainingSettings,DEFAULT_CONFIG,SAM3_ROOT,TRAIN_PYTHON,host_path,wsl_string,dataset_summary,render_config,command,config_name_for,load_stats

class Curve(QWidget):
    def __init__(self):super().__init__();self.values=[];self.setMinimumHeight(200)
    def set_values(self,values):self.values=[float(v) for v in values if v is not None];self.update()
    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rect(),QColor('#0d1523'));p.setRenderHint(QPainter.Antialiasing);p.setPen(QPen(QColor('#35435b')));p.drawRect(45,15,self.width()-65,self.height()-45)
        if len(self.values)<2:p.setPen(QColor('#aebbd0'));p.drawText(self.rect(),Qt.AlignCenter,'Waiting for metrics');return
        lo,hi=min(self.values),max(self.values);span=hi-lo or 1;pts=[(45+i*(self.width()-65)/(len(self.values)-1),15+(hi-v)*(self.height()-45)/span) for i,v in enumerate(self.values)]
        p.setPen(QPen(QColor('#4de0bb'),2))
        for a,b in zip(pts,pts[1:]):p.drawLine(int(a[0]),int(a[1]),int(b[0]),int(b[1]))
        p.setPen(QColor('#aebbd0'));p.drawText(48,30,f'{hi:.4g}');p.drawText(48,self.height()-15,f'{lo:.4g}')

class ConversionDialog(QDialog):
    def __init__(self,parent):
        super().__init__(parent);self.setWindowTitle('Export Inference Model');self.setWindowModality(Qt.WindowModal);self.setMinimumWidth(480)
        self.setWindowFlag(Qt.WindowCloseButtonHint,False)
        layout=QVBoxLayout(self);self.message=QLabel('Training complete. Converting…');self.message.setWordWrap(True);layout.addWidget(self.message)
        self.progress=QProgressBar();self.progress.setRange(0,100);self.progress.setValue(0);layout.addWidget(self.progress)
        self.stage=QLabel('Preparing conversion…');layout.addWidget(self.stage)
    def reject(self):pass
    def closeEvent(self,event):event.ignore()

class TrainPage(QWidget):
    def __init__(self):
        super().__init__();self.active_settings=None;self.last_output=None;self.stopping=False;self.stop_attempts=0;self.running=False;self.converting=False;self.conversion_dialog=None;self.conversion_buffer="";self.conversion_error=""
        self.conversion_process=QProcess(self);self.conversion_process.setProcessChannelMode(QProcess.MergedChannels);self.conversion_process.readyReadStandardOutput.connect(self.read_conversion);self.conversion_process.finished.connect(self.conversion_finished);self.conversion_process.errorOccurred.connect(self.conversion_process_error)
        self.prefs=QSettings('BL62','FineTuneStudio');self.process=QProcess(self);self.process.setProcessChannelMode(QProcess.MergedChannels);self.process.readyReadStandardOutput.connect(self.read_output);self.process.finished.connect(self.finished);self.process.errorOccurred.connect(self.process_error)
        self.env_process=QProcess(self);self.env_process.setProcessChannelMode(QProcess.MergedChannels);self.env_process.readyReadStandardOutput.connect(self.read_env);self.env_process.finished.connect(self.env_finished)
        self.env_process.errorOccurred.connect(lambda e:self.env_error())
        self.helper=QProcess(self);self.helper.setProcessChannelMode(QProcess.MergedChannels);self.helper.finished.connect(self.stop_sent)
        root=QHBoxLayout(self);root.setContentsMargins(0,0,0,0);split=QSplitter();root.addWidget(split);self.setup=QWidget();lv=QVBoxLayout(self.setup)
        dataset=QGroupBox('Dataset');dl=QFormLayout(dataset);self.dataset,row=self.path_row('',True);self.dataset.setPlaceholderText('Save from Data or select a dataset');dl.addRow(row);self.dataset_info=QLabel('No dataset selected');self.dataset_info.setWordWrap(True);dl.addRow(self.dataset_info);self.dataset.editingFinished.connect(self.inspect_dataset);lv.addWidget(dataset)
        ckpt=QGroupBox('Checkpoint');cl=QFormLayout(ckpt);self.mode=QComboBox();self.mode.addItems(['Fine-tune','Resume Training']);cl.addRow('Mode',self.mode);self.checkpoint,row=self.path_row('/mnt/d/Chcekpoints/checkpoint_demo_05142026.pt');cl.addRow(row);lv.addWidget(ckpt)
        gpu=QGroupBox('GPU');self.gpu_layout=QVBoxLayout(gpu);self.gpu_ids=QLineEdit('0,1',gpu);self.gpu_ids.hide();self.gpu_choices={};self.update_gpu_choices([]);self.gpu_ids.textChanged.connect(self.sync_gpu_choices);lv.addWidget(gpu)
        params=QGroupBox('Parameters');pl=QFormLayout(params)
        def spin(lo,hi,value):w=QSpinBox();w.setRange(lo,hi);w.setValue(value);return w
        self.epochs=spin(1,100000,200);self.batch=spin(1,128,1);self.workers=spin(0,64,4);self.val_every=spin(1,10000,10);self.save_every=spin(0,10000,0);self.save_every.setSpecialValueText('Final only');self.lr=QDoubleSpinBox();self.lr.setDecimals(6);self.lr.setRange(.000001,10);self.lr.setValue(.05)
        for title,w in [('Epochs',self.epochs),('Batch / GPU',self.batch),('Workers',self.workers),('Validate every',self.val_every),('Save every',self.save_every),('LR scale',self.lr)]:pl.addRow(title,w)
        lv.addWidget(params);output=QGroupBox('Output');ol=QFormLayout(output);self.output,row=self.path_row('/home/subo/sam3/experiments/finetune_'+datetime.now().strftime('%Y%m%d_%H%M%S'),True);self.output_row=row;ol.addRow(row);self.open_btn=QPushButton('Open Output Folder');self.open_btn.clicked.connect(self.open_output);ol.addRow(self.open_btn);lv.addWidget(output)
        # Environment values stay in the saved configuration; only the shared header displays them.
        self._environment_config=QWidget(self);self._environment_config.hide()
        self.distro=QLineEdit(str(self.prefs.value('distro','Ubuntu-24.04')),self._environment_config)
        self.repository=QLineEdit(str(self.prefs.value('repository',str(SAM3_ROOT))),self._environment_config)
        self.python=QLineEdit(str(self.prefs.value('python',str(TRAIN_PYTHON))),self._environment_config)
        self.config=QLineEdit(str(self.prefs.value('config',str(DEFAULT_CONFIG))),self._environment_config)
        lv.addStretch();self._settings_groups=(dataset,ckpt,gpu,params)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(self.setup);scroll.setMinimumWidth(360);left=QWidget();ll=QVBoxLayout(left);ll.addWidget(scroll)
        controls=QHBoxLayout();self.validate_btn=QPushButton('Validate');self.validate_btn.clicked.connect(self.validate_clicked);self.start=QPushButton('Start Training');self.start.setObjectName('primary');self.start.clicked.connect(self.start_training);self.stop=QPushButton('Safe Stop');self.stop.setEnabled(False);self.stop.clicked.connect(self.stop_training)
        for b in (self.validate_btn,self.start,self.stop):controls.addWidget(b)
        ll.addLayout(controls);split.addWidget(left)
        right=QWidget();rv=QVBoxLayout(right);self.state=QLabel('Ready');self.state.setObjectName('state');rv.addWidget(self.state);self.progress=QProgressBar();rv.addWidget(self.progress);self.epoch=QLabel('Epoch — / —');rv.addWidget(self.epoch)
        self.gpu_status=GpuStatusPanel(self);self.gpu_status.set_selected_ids(self.gpu_ids.text());self.gpu_ids.textChanged.connect(self.gpu_status.set_selected_ids);rv.addWidget(self.gpu_status);self.gpu_status.monitor.updated.connect(self.update_gpu_choices)
        self.metric_selector=QComboBox();self.metric_selector.addItems(['Training Loss','Mask mIoU','Dice Loss','Validation AP','Validation AP50']);self.metric_selector.currentIndexChanged.connect(self.refresh);rv.addWidget(self.metric_selector);self.curve=Curve();rv.addWidget(self.curve,2);self.metrics=QLabel('No metrics yet');self.metrics.setWordWrap(True);rv.addWidget(self.metrics)
        self.artifacts=QLabel('No checkpoint yet');self.artifacts.setWordWrap(True);rv.addWidget(self.artifacts);self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumBlockCount(10000);rv.addWidget(self.log,2);split.addWidget(right);split.setSizes([430,1000])
        self.timer=QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(3000)
        self.stop_timer=QTimer(self);self.stop_timer.setInterval(500);self.stop_timer.timeout.connect(self.try_stop)

    def update_gpu_choices(self,readings):
        names={r.index:r.name for r in readings}
        ids=set(names)|{int(x) for x in self.gpu_ids.text().split(',') if x.strip().isdigit()}
        for index in sorted(ids):
            if index not in self.gpu_choices:
                box=QCheckBox();self.gpu_choices[index]=box;self.gpu_layout.addWidget(box)
                box.toggled.connect(self.gpu_selection_changed)
            self.gpu_choices[index].setText(f'GPU {index} · {names[index]}' if index in names else f'GPU {index}')
        self.sync_gpu_choices()

    def sync_gpu_choices(self):
        selected={int(x) for x in self.gpu_ids.text().split(',') if x.strip().isdigit()}
        for index,box in self.gpu_choices.items():
            box.blockSignals(True);box.setChecked(index in selected);box.blockSignals(False)

    def gpu_selection_changed(self):
        self.gpu_ids.setText(','.join(str(i) for i,box in sorted(self.gpu_choices.items()) if box.isChecked()))

    def set_setup_enabled(self,enabled):
        for group in self._settings_groups:group.setEnabled(enabled)
        self.output_row.setEnabled(enabled)

    def path_row(self,value,folder=False):
        edit=QLineEdit(str(value));row=QWidget();lay=QHBoxLayout(row);lay.setContentsMargins(0,0,0,0);lay.addWidget(edit,1);browse=QPushButton('Browse');browse.clicked.connect(lambda:self.browse(edit,folder));lay.addWidget(browse);return edit,row
    def browse(self,edit,folder):
        initial=str(self.local_path(edit.text())) if edit.text() else ''
        if folder:value=QFileDialog.getExistingDirectory(self,'Select Folder',initial)
        else:value,_=QFileDialog.getOpenFileName(self,'Select File',initial)
        if value:edit.setText(value)
        if edit is self.dataset:self.inspect_dataset()
    def local_path(self,value):
        raw=wsl_string(Path(value))
        if os.name=='nt' and raw.startswith('/') and not raw.startswith('/mnt/'):
            return Path('\\\\wsl.localhost\\'+self.distro.text().strip()+raw.replace('/','\\'))
        return host_path(Path(raw)) if os.name=='nt' else Path(raw)
    def set_dataset(self,path):self.dataset.setText(path);self.inspect_dataset()
    def inspect_dataset(self):
        try:
            summary=dataset_summary(self.local_path(self.dataset.text()));self.dataset_info.setText(f"Train {summary['train']['images']} · Validate {summary['val']['images']}\n"+' / '.join(summary['train']['categories']))
        except Exception as e:self.dataset_info.setText(str(e))
    def settings(self):
        if not self.gpu_ids.text().strip():raise ValueError('Select at least one GPU')
        ids=[x.strip() for x in self.gpu_ids.text().split(',')]
        if not ids or any(not x.isdigit() for x in ids) or len(set(ids))!=len(ids):raise ValueError('GPU IDs must be unique integers separated by commas')
        dataset=Path(wsl_string(Path(self.dataset.text().strip())))
        chosen=Path(wsl_string(Path(self.checkpoint.text().strip())))
        if not self.output.text().strip():raise ValueError('Select a new output folder')
        if not self.dataset.text().strip() or not self.checkpoint.text().strip():raise ValueError('Select a dataset and checkpoint')
        # Resume uses the same picker; the initializer must be a model-weights checkpoint.
        base=chosen
        if self.mode.currentIndex()==1:
            meta=self.local_path(str(chosen)).parent.parent/'gui_launch.json'
            if meta.is_file():
                previous=json.loads(meta.read_text(encoding='utf-8'));base=Path(previous['initial_checkpoint'])
        return TrainingSettings(config=Path(wsl_string(Path(self.config.text()))),dataset=dataset,output=Path(wsl_string(Path(self.output.text()))),checkpoint=base,category='finetune',epochs=self.epochs.value(),batch_size=self.batch.value(),gpus=len(ids),workers=self.workers.value(),val_every=self.val_every.value(),save_every=self.save_every.value(),lr_scale=self.lr.value(),resume_from=chosen if self.mode.currentIndex()==1 else None,repository=Path(wsl_string(Path(self.repository.text()))),python=Path(wsl_string(Path(self.python.text()))),distro=self.distro.text().strip(),export_inference=True,inference_base=base)
    def validate_settings(self):
        s=self.settings()
        for value,label in [(s.repository/'sam3/train/train.py','SAM3 repository'),(s.config,'config template'),(s.checkpoint,'checkpoint')]:
            if not self.local_path(str(value)).is_file():raise ValueError(f'Missing {label}: {value}')
        if not s.distro:raise ValueError('Enter a WSL distribution')
        data=dataset_summary(self.local_path(str(s.dataset)))
        if any(data[split]['images']<1 for split in ('train','val')):raise ValueError('Training and Validation must both contain images')
        if not data['train']['categories'] or data['train']['categories']!=data['val']['categories']:raise ValueError('Training and Validation label lists must match')
        out=self.local_path(str(s.output))
        if out.exists() and any(out.iterdir()):raise ValueError('Output folder must be new or empty')
        if s.resume_from and not self.local_path(str(s.resume_from)).is_file():raise ValueError('Resume checkpoint not found')
        if s.resume_from:
            meta=self.local_path(str(s.resume_from)).parent.parent/'gui_launch.json'
            if meta.is_file():
                previous=json.loads(meta.read_text())
                if previous.get('dataset')!=wsl_string(s.dataset) or previous.get('gpu_ids')!=self.gpu_ids.text().replace(' ',''):raise ValueError('Resume must use the original dataset and GPU IDs')
        return s
    def validate_clicked(self):
        try:s=self.validate_settings();self.log.appendPlainText(render_config(replace(s,config=self.local_path(str(s.config)))));self.state.setText('Configuration valid');self.inspect_dataset()
        except Exception as e:QMessageBox.warning(self,'Configuration',str(e))
    def shell_start(self,process,script):
        if os.name=='nt':process.setWorkingDirectory('C:\\Windows');process.start('C:\\Windows\\System32\\wsl.exe',['-d',self.distro.text().strip(),'--','bash','-lc',script])
        else:process.start('/bin/bash',['-lc',script])
    def env_script(self,s,checkpoint=False):
        code="import sys, torch, sam3, hydra, pycocotools; print('Python:',sys.executable); print('SAM3:',sam3.__file__); print('CUDA:',torch.version.cuda); print('GPUs:',torch.cuda.device_count()); assert torch.cuda.is_available(), 'CUDA unavailable'; "
        code+="[print(i,torch.cuda.get_device_name(i),str(round(torch.cuda.get_device_properties(i).total_memory/2**30,1))+' GiB') for i in range(torch.cuda.device_count())]; "
        code+=f"assert torch.cuda.device_count()>={s.gpus}, 'Requested GPUs unavailable'; "
        if checkpoint:
            ckpt=wsl_string(s.resume_from or s.checkpoint);code+=f"p=torch.load({ckpt!r},map_location='cpu',weights_only=False); assert isinstance(p,dict), 'Checkpoint must be a dictionary'; "
            code+=(f"assert 'optimizer' in p and 'epoch' in p, 'Not a full resume checkpoint'; assert int(p['epoch']) < {s.epochs}, 'Epochs must exceed the saved epoch'; " if s.resume_from else "assert 'model' in p or any(hasattr(v,'shape') for v in p.values()), 'Unsupported model checkpoint'; ")
        return 'cd '+shlex.quote(wsl_string(s.repository))+' && env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='+shlex.quote(self.gpu_ids.text().replace(' ',''))+' '+shlex.quote(wsl_string(s.python))+' -c '+shlex.quote(code)
    def check_environment(self):
        if self.env_process.state()!=QProcess.NotRunning:return
        try:s=self.settings() if self.dataset.text() and self.checkpoint.text() else TrainingSettings(repository=Path(wsl_string(Path(self.repository.text()))),python=Path(wsl_string(Path(self.python.text()))),gpus=len(self.gpu_ids.text().split(',')))
        except Exception as e:QMessageBox.warning(self,'Environment',str(e));return
        self.save_prefs();self.state.setText('Checking environment');self.shell_start(self.env_process,self.env_script(s))
    def read_env(self):self.log.appendPlainText(bytes(self.env_process.readAllStandardOutput()).decode('utf-8','replace').rstrip())
    def env_finished(self,code,status):self.state.setText('Environment ready' if code==0 else 'Environment check failed')
    def env_error(self):self.log.appendPlainText(self.env_process.errorString())
    def save_prefs(self):
        for name in ('distro','repository','python','config'):self.prefs.setValue(name,getattr(self,name).text())
    def start_training(self):
        if self.running or self.converting:return
        try:
            s=self.validate_settings();self.save_prefs()
            # Freeze settings for the whole run so metrics, stop and output keep their identity.
            self.active_settings=s;self.run_id=uuid.uuid4().hex;self.pid_file=s.output/'gui_training.pid';self.stop_file=s.output/'gui_stop_requested'
            cfg=s.repository/'sam3/train'/config_name_for(s.output)
            if self.local_path(str(cfg)).exists():cfg=cfg.with_name(cfg.stem+'_'+self.run_id[:8]+'.yaml')
            cfg_name='configs/gui_runs/'+cfg.name
            rendered=render_config(replace(s,config=self.local_path(str(s.config))));self.local_path(str(cfg)).parent.mkdir(parents=True,exist_ok=True)
            # Resolve host paths through the selected WSL distribution.
            self.local_path(str(cfg)).write_text(rendered,encoding='utf-8')
            out=self.local_path(str(s.output));out.mkdir(parents=True,exist_ok=True)
            (out/'gui_launch.json').write_text(json.dumps({'initial_checkpoint':wsl_string(s.checkpoint),'dataset':wsl_string(s.dataset),'gpu_ids':self.gpu_ids.text().replace(' ',''),'runtime_config':str(cfg),'run_id':self.run_id},indent=2),encoding='utf-8')
            args=command(s);args[args.index('-c')+1]=cfg_name
            bootstrap="import os,signal,sys; signal.signal(signal.SIGINT,signal.SIG_DFL); signal.signal(signal.SIGTERM,signal.SIG_DFL); sys.exit(130) if os.path.exists("+repr(wsl_string(self.stop_file))+") else None; os.execvpe(sys.argv[1],sys.argv[1:],os.environ)"
            train=shlex.join([wsl_string(s.python),'-c',bootstrap,*args]);gpu=shlex.quote(self.gpu_ids.text().replace(' ',''));pid=shlex.quote(wsl_string(self.pid_file));stop=shlex.quote(wsl_string(self.stop_file))
            wrapper=f"{self.env_script(s,True)} || exit $?\n[ -f {stop} ] && exit 130\ncd {shlex.quote(wsl_string(s.repository))} || exit 1\nexport CUDA_VISIBLE_DEVICES={gpu}\nsetsid {train} &\nchild=$!\nprintf '%s' \"$child\" > {pid}\nif [ -f {stop} ]; then kill -INT -- -\"$child\"; fi\nwait \"$child\"\nresult=$?\nrm -f {pid}\nexit \"$result\"\n"
            # A file avoids Windows/WSL quoting differences in nested bash commands.
            script=s.output/'launch.sh';self.local_path(str(script)).write_text(wrapper,encoding='utf-8',newline='\n')
            self.log.clear()
            self.running=True;self.stopping=False;self.last_output=s.output;self.set_setup_enabled(False);self.start.setEnabled(False);self.validate_btn.setEnabled(False);self.stop.setEnabled(True);self.state.setText('Starting');self.progress.setValue(0);self.curve.set_values([]);self.log.appendPlainText('Launching training. Environment and checkpoint are checked before GPU training.');self.shell_start(self.process,'bash '+shlex.quote(wsl_string(script)))
        except Exception as e:QMessageBox.warning(self,'Cannot Start',str(e))
    def read_output(self):self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode('utf-8','replace').rstrip())
    def process_error(self):
        self.log.appendPlainText(self.process.errorString())
        if self.process.state()==QProcess.NotRunning:self.finished(-1,None)
    def finished(self,code,status):
        was_running=self.running
        self.running=False;self.stop_timer.stop();self.stop.setEnabled(False);self.stop.setText('Safe Stop');self.refresh()
        if was_running and code==0 and status==QProcess.NormalExit and not self.stopping and self.active_settings and self.active_settings.export_inference:
            self.begin_conversion();return
        self.state.setText('Stopped' if self.stopping else 'Completed' if code==0 else f'Failed ({code})');self.unlock_after_run()

    def unlock_after_run(self):
        self.set_setup_enabled(True);self.start.setEnabled(True);self.validate_btn.setEnabled(True)
        if self.last_output:self.output.setText(str(self.last_output)+'_'+datetime.now().strftime('%H%M%S'))

    def begin_conversion(self):
        self.converting=True;self.conversion_buffer='';self.conversion_error='';self.state.setText('Training complete. Converting…')
        self.set_setup_enabled(False);self.start.setEnabled(False);self.validate_btn.setEnabled(False)
        self.conversion_dialog=ConversionDialog(self);self.conversion_dialog.show()
        self.log.appendPlainText('Training complete. Converting checkpoint to an inference model…')
        try:
            s=self.active_settings;out=self.local_path(str(s.output));self.inference_output=s.output/'checkpoints/checkpoint_sam3_format.pt'
            worker=s.output/'convert_checkpoint.py'
            # The converter is included in the packaged GUI; no torch import in the GUI process.
            source=Path(convert_checkpoint.__file__)
            if not source.is_file() or source.suffix!='.py':
                source=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent.parent))/'assets/convert_checkpoint.py'
            self.local_path(str(worker)).write_bytes(source.read_bytes())
            args=[wsl_string(s.python),'-u',wsl_string(worker),'--base',wsl_string(s.inference_base),'--training',wsl_string(s.output/'checkpoints/checkpoint.pt'),'--output',wsl_string(self.inference_output),'--fallback-base',wsl_string(s.repository/'sam3.pt')]
            self.shell_start(self.conversion_process,'env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= '+shlex.join(args))
        except Exception as e:self.conversion_error=str(e);self.conversion_finished(-1,None)

    def read_conversion(self):
        self.conversion_buffer+=bytes(self.conversion_process.readAllStandardOutput()).decode('utf-8','replace')
        while '\n' in self.conversion_buffer:
            line,self.conversion_buffer=self.conversion_buffer.split('\n',1);self.conversion_line(line)

    def conversion_line(self,line):
        if not line.strip():return
        if line.startswith(convert_checkpoint.PREFIX):
            try:
                message=json.loads(line[len(convert_checkpoint.PREFIX):])
                if 'error' in message:self.conversion_error=message['error'];self.log.appendPlainText('Conversion error: '+self.conversion_error)
                else:
                    self.conversion_dialog.progress.setValue(min(100,max(0,int(message['progress']))));self.conversion_dialog.stage.setText(message['message']);self.log.appendPlainText(message['message'])
            except (ValueError,KeyError,TypeError):self.log.appendPlainText(line)
        else:self.log.appendPlainText(line)

    def conversion_process_error(self,error):
        if error==QProcess.FailedToStart:
            self.conversion_error=self.conversion_process.errorString();self.conversion_finished(-1,None)

    def conversion_finished(self,code,status):
        if not self.converting:return
        self.read_conversion()
        if self.conversion_buffer:self.conversion_line(self.conversion_buffer);self.conversion_buffer=''
        success=code==0 and status==QProcess.NormalExit and self.local_path(str(self.inference_output)).is_file()
        self.converting=False
        if self.conversion_dialog:self.conversion_dialog.done(0);self.conversion_dialog.deleteLater();self.conversion_dialog=None
        self.unlock_after_run();self.state.setText('Training and conversion complete' if success else 'Training complete. Conversion failed');self.refresh()
        if success:
            display_path=str(self.local_path(wsl_string(self.inference_output)))
            self.log.appendPlainText('Inference model: '+display_path)
            QMessageBox.information(self,'Conversion Complete','Inference model saved:\n'+display_path)
        else:
            detail=self.conversion_error or 'See Training Log for details.'
            QMessageBox.warning(self,'Conversion Failed','Training completed. The full training checkpoint is preserved.\n'+detail)

    def stop_training(self):
        if not self.running:return
        self.stopping=True;self.stop_attempts=0;self.stop.setEnabled(False);self.stop.setText('Stopping…');self.state.setText('Stopping safely')
        self.local_path(str(self.stop_file)).touch();self.stop_timer.start();self.try_stop()
    def try_stop(self):
        if not self.running:self.stop_timer.stop();return
        if self.helper.state()!=QProcess.NotRunning:return
        self.stop_attempts+=1
        try:pid=self.local_path(str(self.pid_file)).read_text().strip()
        except OSError:pid=''
        if pid.isdigit():
            # Only signal the recorded session whose command belongs to this run.
            cfg=json.loads((self.local_path(str(self.active_settings.output))/'gui_launch.json').read_text())['runtime_config']
            match=Path(cfg).name
            script=f"if [ -r /proc/{pid}/cmdline ] && tr '\\0' ' ' < /proc/{pid}/cmdline | grep -F -- {shlex.quote(match)} >/dev/null; then kill -INT -- -{pid}; fi"
            self.shell_start(self.helper,script);self.stop_timer.stop();self.log.appendPlainText('Safe stop sent to this training process group.');return
        if self.stop_attempts>=40:
            self.stop_timer.stop();self.process.terminate();self.log.appendPlainText('Training had not started; launcher closed.');QTimer.singleShot(3000,self.kill_launcher)
    def stop_sent(self,code,status):
        if self.running and self.stopping:QTimer.singleShot(30000,self.offer_force_stop)
    def offer_force_stop(self):
        if not self.running or not self.stopping:return
        self.state.setText('Waiting for safe stop');self.log.appendPlainText('Training is still finishing; waiting for checkpoint cleanup.')
    def kill_launcher(self):
        if self.running and self.stopping:self.process.kill()
    def refresh(self):
        if not self.active_settings:return
        s=self.active_settings;out=self.local_path(str(s.output));rows=load_stats(out/'logs'/s.category/'train_stats.json');vals=load_stats(out/'logs'/s.category/'val_stats.json')
        mapping=[('Losses/train_all_loss',rows),('Losses/train_all_miou_semantic_seg',rows),('Losses/train_all_loss_semantic_dice',rows),('Meters_train/val_roboflow100/detection/coco_eval_bbox_AP',vals),('Meters_train/val_roboflow100/detection/coco_eval_bbox_AP_50',vals)]
        key,data=mapping[self.metric_selector.currentIndex()];self.curve.set_values([r.get(key) for r in data])
        if rows:
            row=rows[-1];epoch=int(row.get('Trainer/epoch',0))+1;self.epoch.setText(f'Epoch {epoch} / {s.epochs}');self.progress.setValue(min(100,round(100*epoch/s.epochs)));self.metrics.setText('  ·  '.join(f"{label} {row.get(k,0):.4g}" for label,k in [('Loss','Losses/train_all_loss'),('mIoU','Losses/train_all_miou_semantic_seg'),('Dice loss','Losses/train_all_loss_semantic_dice')]))
        ckpt=out/'checkpoints/checkpoint.pt';inference=out/'checkpoints/checkpoint_sam3_format.pt'
        text='Training Checkpoint: '+str(ckpt) if ckpt.exists() else 'No checkpoint yet'
        if inference.is_file():text+='\nInference Model: '+str(inference)
        self.artifacts.setText(text)
    def open_output(self):
        path=self.local_path(str(self.last_output)) if self.last_output else self.local_path(self.output.text())
        if not path.is_dir():return
        if os.name=='nt':os.startfile(str(path))
        else:QProcess.startDetached('xdg-open',[str(path)])
