import copy,json,tempfile,unittest,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from finetune_gui.data import DataProject,decode,encode
from finetune_gui.training import TrainingSettings,render_config,wsl_string

class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.source=self.root/'source';(self.source/'images').mkdir(parents=True)
        self.mask=np.zeros((32,40),bool);self.mask[2:25,3:30]=True;self.mask[7:15,10:19]=False;self.mask[29,38]=True
        data={'images':[],'annotations':[],'categories':[{'id':4,'name':'particle','supercategory':'particle'},{'id':8,'name':'fragment','supercategory':'fragment'}]}
        for i in range(4):
            Image.new('RGB',(40,32),'gray').save(self.source/'images'/f'{i}.png');data['images'].append({'id':i+1,'file_name':f'{i}.png','width':40,'height':32});data['annotations'].append({'id':i+1,'image_id':i+1,'category_id':4,'segmentation':encode(self.mask),'bbox':[0,0,40,32],'area':int(self.mask.sum())})
        self.json=self.source/'annotations.json';self.json.write_text(json.dumps(data));self.digest=hashlib.sha256(self.json.read_bytes()).hexdigest();self.p=DataProject.load(self.source)
    def tearDown(self):self.temp.cleanup()
    def test_exact_holes_single_pixels_overlap_and_source_preserved(self):
        masks=self.p.masks(0);np.testing.assert_array_equal(masks[0]['mask'],self.mask);masks.append(copy.deepcopy(masks[0]));masks[0]['mask'][0,0]=True;self.p.commit(0,masks);self.p.apply_split(['train','train','val','']);out=self.root/'out';self.p.export(out);loaded=DataProject.load(out);self.assertEqual(len(loaded.images),4);self.assertEqual(len(loaded.masks(0)),2);np.testing.assert_array_equal(loaded.masks(0)[0]['mask'],masks[0]['mask']);self.assertEqual(hashlib.sha256(self.json.read_bytes()).hexdigest(),self.digest);self.assertTrue((out/'unassigned/images/3.png').is_file())
    def test_global_rename_updates_all(self):
        self.p.rename(1,'grain');self.assertTrue(all(a['noun_phrase']=='grain' for r in self.p.images for a in r['annotations']));self.assertRaises(ValueError,self.p.rename,1,'fragment')
    def test_random_remaining_and_deterministic(self):
        self.p.apply_split(['train','val','','']);a=self.p.plan_split('unassigned',80,42);b=self.p.plan_split('unassigned',80,42);self.assertEqual(a,b);self.assertEqual(a[:2],['train','val']);self.assertEqual(a.count('train'),2);self.assertEqual(a.count('val'),2)
    def test_export_refuses_existing_data(self):
        self.p.apply_split(['train','val','','']);out=self.root/'output';out.mkdir();(out/'keep').write_text('keep');self.assertRaises(ValueError,self.p.export,out);self.assertEqual((out/'keep').read_text(),'keep')
    def test_export_failure_cleans_staging(self):
        self.p.apply_split(['train','val','','']);self.p.images[0]['path']=self.root/'missing.png';self.assertRaises(FileNotFoundError,self.p.export,self.root/'failed');self.assertFalse((self.root/'failed').exists());self.assertEqual(list(self.root.glob('.finetune-export-*')),[])
    def test_duplicate_filenames_are_disambiguated(self):
        self.p.images[1]['name']=self.p.images[0]['name'];self.p.apply_split(['train','train','val','val']);out=self.root/'dup';self.p.export(out);data=json.loads((out/'train/annotations.json').read_text());self.assertEqual(len({r['file_name'] for r in data['images']}),2)
    def test_export_requires_both_splits(self):self.assertRaises(ValueError,self.p.export,self.root/'invalid')
    def test_coco_polygons(self):self.assertGreater(decode([[1,1,12,1,12,12,1,12]],40,32).sum(),0)
    def test_windows_and_unc_conversion(self):
        self.assertEqual(wsl_string(Path('D:/Some Data/images')),'/mnt/d/Some Data/images');self.assertEqual(wsl_string(Path('//wsl.localhost/Ubuntu-24.04/home/subo')),'/home/subo')
    def test_config_render_uses_paths_and_resume(self):
        template=self.root/'template.yaml';template.write_text('roboflow_vl_100_root: old\nexperiment_log_dir: old\nsupercategory: old\nmax_epochs: 1\nnum_train_workers: 1\nlr_scale: 1\ntrain_batch_size: 1\nval_epoch_freq: 1\ncheckpoint_path: old\nsave_freq: 0\ngpus_per_node: 1\nuse_cluster: true\n')
        settings=TrainingSettings(config=template,dataset=Path('/data with spaces/颗粒'),resume_from=Path('/full checkpoint.pt'));text=render_config(settings);self.assertIn('resume_from: "/full checkpoint.pt"',text);self.assertIn('roboflow_vl_100_root: "/data with spaces/颗粒"',text);self.assertNotIn('resume_from',template.read_text())
if __name__=='__main__':unittest.main()
