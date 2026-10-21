"""Asynchronous device-wide NVIDIA telemetry; no model or CUDA context is loaded."""
from __future__ import annotations
import csv
from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
import shutil
from PySide6.QtCore import QObject,QProcess,QTimer,Signal
from PySide6.QtWidgets import QGroupBox,QGridLayout,QLabel,QProgressBar

QUERY_ARGS=['--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits']

@dataclass(frozen=True)
class GpuReading:
    index:int
    name:str
    utilization:float|None
    memory_used:float|None
    memory_total:float|None


def _number(value):
    value=value.strip()
    if value.casefold() in {'n/a','[n/a]','not supported','[not supported]','-',''}:return None
    number=float(value)
    if not math.isfinite(number) or number<0:raise ValueError('Invalid GPU metric')
    return number


def parse_gpu_csv(output):
    readings=[];seen=set()
    for row in csv.reader(io.StringIO(output),skipinitialspace=True):
        if not row or not any(x.strip() for x in row):continue
        if len(row)!=5:raise ValueError('Unexpected nvidia-smi output')
        index=int(row[0].strip());name=row[1].strip()
        if index<0 or index in seen or not name:raise ValueError('Invalid GPU identity')
        utilization,used,total=(_number(x) for x in row[2:])
        if utilization is not None and utilization>100:raise ValueError('Invalid GPU utilization')
        if total==0:total=None
        seen.add(index);readings.append(GpuReading(index,name,utilization,used,total))
    if not readings:raise ValueError('No GPU data returned')
    return sorted(readings,key=lambda r:r.index)


def query_command():
    executable=shutil.which('nvidia-smi')
    if os.name=='nt' and not executable:
        for candidate in (Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/nvidia-smi.exe',Path('C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe')):
            if candidate.is_file():executable=str(candidate);break
    if not executable and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():executable='/usr/lib/wsl/lib/nvidia-smi'
    if not executable:raise FileNotFoundError('nvidia-smi was not found')
    return [executable,*QUERY_ARGS]


class GpuMonitor(QObject):
    updated=Signal(object)
    unavailable=Signal(str)
    def __init__(self,parent=None,command_factory=None,interval_ms=2000,timeout_ms=5000):
        super().__init__(parent);self.command_factory=command_factory or query_command;self.timeout_ms=timeout_ms
        self.active=False;self.closed=False;self.timed_out=False;self.sample_count=0;self.request_count=0
        self.process=QProcess(self);self.process.finished.connect(self._finished);self.process.errorOccurred.connect(self._error)
        self.timer=QTimer(self);self.timer.setInterval(interval_ms);self.timer.timeout.connect(self.poll)
        self.deadline=QTimer(self);self.deadline.setSingleShot(True);self.deadline.timeout.connect(self._timeout)
    def start(self):
        if self.closed:return
        self.active=True;self.timer.start();self.poll()
    def pause(self):self.active=False;self.timer.stop()
    def poll(self):
        if self.closed or not self.active or self.process.state()!=QProcess.NotRunning:return
        try:command=self.command_factory()
        except Exception as e:self.unavailable.emit(str(e));return
        self.timed_out=False;self.request_count+=1;self.process.start(command[0],command[1:]);self.deadline.start(self.timeout_ms)
    def _finished(self,code,status):
        self.deadline.stop();output=bytes(self.process.readAllStandardOutput()).decode('utf-8','replace');error=bytes(self.process.readAllStandardError()).decode('utf-8','replace').strip()
        if self.closed:return
        if self.timed_out:self.unavailable.emit('GPU query timed out');return
        if code!=0:self.unavailable.emit(error or output.strip() or f'nvidia-smi exited with code {code}');return
        try:readings=parse_gpu_csv(output)
        except Exception as e:self.unavailable.emit(str(e));return
        self.sample_count+=1;self.updated.emit(readings)
    def _error(self,error):
        if error==QProcess.FailedToStart:
            self.deadline.stop()
            if not self.closed:self.unavailable.emit(self.process.errorString())
    def _timeout(self):
        if self.process.state()!=QProcess.NotRunning:self.timed_out=True;self.process.kill()
    def stop(self):
        self.closed=True;self.pause();self.deadline.stop()
        if self.process.state()!=QProcess.NotRunning:self.process.kill();self.process.waitForFinished(1000)


class GpuStatusPanel(QGroupBox):
    def __init__(self,parent=None,auto_start=True):
        super().__init__('GPU Status',parent);self.auto_start=auto_start;self.readings=[];self.cards={};self.selected=set()
        self.setToolTip('Device-wide readings from nvidia-smi. Refreshes every 2 seconds while Training is visible.')
        self.grid=QGridLayout(self);self.grid.setHorizontalSpacing(12)
        for column,title in enumerate(('GPU','Utilization','VRAM (Used / Total)')):self.grid.addWidget(QLabel(title),0,column)
        self.grid.setColumnStretch(0,2);self.grid.setColumnStretch(1,1);self.grid.setColumnStretch(2,2)
        self.status=QLabel('Detecting GPUs…');self.status.setWordWrap(True);self.grid.addWidget(self.status,1,0,1,3)
        self.monitor=GpuMonitor(self);self.monitor.updated.connect(self.apply_readings);self.monitor.unavailable.connect(self.show_unavailable)
    def set_selected_ids(self,value):
        self.selected={int(x.strip()) for x in value.split(',') if x.strip().isdigit()};self._names()
    def _names(self):
        for reading in self.readings:
            label=self.cards[reading.index]['name'];selected=reading.index in self.selected
            label.setText(f"GPU {reading.index} · {reading.name}"+(' · Selected' if selected else ''))
            label.setStyleSheet('color:#72e4c4' if selected else '')
    def _clear_cards(self):
        for card in self.cards.values():
            for widget in card.values():self.grid.removeWidget(widget);widget.deleteLater()
        self.cards={}
    def apply_readings(self,readings):
        identities=[(r.index,r.name) for r in readings]
        if identities!=[(r.index,r.name) for r in self.readings]:
            self._clear_cards()
            for row,r in enumerate(readings,1):
                name=QLabel();name.setWordWrap(True);util=QProgressBar();util.setRange(0,100);util.setMinimumWidth(100);memory=QProgressBar();memory.setRange(0,100);memory.setMinimumWidth(180)
                self.cards[r.index]={'name':name,'utilization':util,'memory':memory}
                for column,widget in enumerate((name,util,memory)):self.grid.addWidget(widget,row,column)
        self.readings=list(readings);self.status.hide()
        for r in readings:
            card=self.cards[r.index];util=card['utilization'];util.setValue(round(r.utilization) if r.utilization is not None else 0);util.setFormat(f'{r.utilization:.0f}%' if r.utilization is not None else 'N/A')
            memory=card['memory']
            if r.memory_used is None or r.memory_total is None:
                memory.setValue(0);memory.setFormat('N/A')
            else:
                memory.setValue(min(100,round(100*r.memory_used/r.memory_total)));memory.setFormat(f'{r.memory_used/1024:.1f} / {r.memory_total/1024:.1f} GiB')
        self._names()
    def show_unavailable(self,message):
        self._clear_cards();self.readings=[];self.status.setText('GPU status unavailable');self.status.setToolTip(message);self.status.show()
    def showEvent(self,event):
        super().showEvent(event)
        if self.auto_start:self.monitor.start()
    def hideEvent(self,event):self.monitor.pause();super().hideEvent(event)
    def stop(self):self.monitor.stop()
