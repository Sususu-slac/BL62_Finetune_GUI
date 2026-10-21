import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import sys,time,unittest
from PySide6.QtCore import QTimer,QProcess
from PySide6.QtWidgets import QApplication
from finetune_gui.gpu_monitor import GpuMonitor,GpuStatusPanel,GpuReading,parse_gpu_csv

app=QApplication.instance() or QApplication([])
CSV='0, NVIDIA RTX 6000 Ada Generation, 42, 4096, 49140\n1, NVIDIA RTX 6000 Ada Generation, N/A, 99, 49140\n'

class GpuTests(unittest.TestCase):
    def test_csv_handles_multiple_devices_and_unavailable_metrics(self):
        rows=parse_gpu_csv(CSV);self.assertEqual([r.index for r in rows],[0,1]);self.assertEqual(rows[0].utilization,42);self.assertEqual(rows[0].memory_used,4096);self.assertIsNone(rows[1].utilization)
        rows=parse_gpu_csv('2, "GPU, quoted name", [Not Supported], N/A, 0\n');self.assertEqual(rows[0].name,'GPU, quoted name');self.assertIsNone(rows[0].memory_used);self.assertIsNone(rows[0].memory_total)
    def test_malformed_or_empty_data_is_not_reported_as_zero_usage(self):
        for text in ('','NVIDIA-SMI has failed','0, GPU, 140, 0, 100','0, GPU, nan, 0, 100','0, GPU, 0, 1, 100\n0, GPU, 0, 1, 100'):
            with self.subTest(text=text):self.assertRaises(ValueError,parse_gpu_csv,text)
    def test_panel_selection_memory_and_failure_clear_stale_data(self):
        panel=GpuStatusPanel(auto_start=False);panel.set_selected_ids('1');panel.apply_readings(parse_gpu_csv(CSV));self.assertNotIn('Selected',panel.cards[0]['name'].text());self.assertIn('Selected',panel.cards[1]['name'].text());self.assertEqual(panel.cards[0]['utilization'].format(),'42%');self.assertEqual(panel.cards[1]['utilization'].format(),'N/A');self.assertIn('4.0 / 48.0 GiB',panel.cards[0]['memory'].format())
        panel.show_unavailable('Query failed');self.assertFalse(panel.readings);self.assertFalse(panel.cards);self.assertEqual(panel.status.text(),'GPU status unavailable');panel.apply_readings([GpuReading(0,'GPU',0,None,None)]);self.assertEqual(panel.cards[0]['memory'].format(),'N/A');panel.stop();panel.deleteLater();app.processEvents()
    def test_background_queries_do_not_overlap_or_block_event_loop(self):
        command=[sys.executable,'-c',"import time;time.sleep(.15);print("+repr(CSV)+")"]
        monitor=GpuMonitor(command_factory=lambda:command,interval_ms=50,timeout_ms=3000);samples=[];ticks=[]
        monitor.updated.connect(samples.append);pulse=QTimer();pulse.setInterval(10);pulse.timeout.connect(lambda:ticks.append(1));pulse.start()
        try:
            monitor.start();monitor.poll();monitor.poll();self.assertEqual(monitor.request_count,1);deadline=time.monotonic()+7
            while len(samples)<2 and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
            self.assertGreaterEqual(len(samples),2);self.assertGreaterEqual(len(ticks),10);monitor.stop();count=monitor.request_count
            end=time.monotonic()+.15
            while time.monotonic()<end:app.processEvents();time.sleep(.005)
            self.assertEqual(monitor.request_count,count);self.assertEqual(monitor.process.state(),QProcess.NotRunning)
        finally:monitor.stop();pulse.stop();monitor.deleteLater();app.processEvents()
    def test_timeout_recovers_on_later_successful_query(self):
        calls=[]
        def command():
            calls.append(1)
            code='import time;time.sleep(3)' if len(calls)==1 else 'print('+repr(CSV)+')'
            return [sys.executable,'-c',code]
        monitor=GpuMonitor(command_factory=command,interval_ms=80,timeout_ms=700);errors=[];samples=[];monitor.unavailable.connect(errors.append);monitor.updated.connect(samples.append)
        try:
            monitor.start();deadline=time.monotonic()+7
            while not samples and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
            self.assertTrue(any('timed out' in e for e in errors));self.assertTrue(samples);self.assertGreaterEqual(len(calls),2)
        finally:monitor.stop();monitor.deleteLater();app.processEvents()
if __name__=='__main__':unittest.main()
