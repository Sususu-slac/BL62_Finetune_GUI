"""COCO review and lossless, staged training-dataset export."""
from __future__ import annotations
import copy
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

EXTENSIONS={'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}


def decode(segmentation, width, height):
    if isinstance(segmentation, list):
        if not segmentation: return np.zeros((height,width),bool)
        rle=mask_utils.merge(mask_utils.frPyObjects(segmentation,height,width))
    elif isinstance(segmentation, dict):
        rle=segmentation
        if isinstance(rle.get('counts'),list): rle=mask_utils.frPyObjects(rle,height,width)
    else: raise ValueError('Unsupported COCO segmentation')
    arr=mask_utils.decode(rle)
    if arr.ndim==3: arr=np.any(arr,axis=2)
    if arr.shape!=(height,width): raise ValueError('Mask dimensions do not match image')
    return arr.astype(bool)


def encode(mask):
    rle=mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts']=rle['counts'].decode('ascii')
    return rle


class DataProject:
    def __init__(self):
        self.images=[]; self.categories=[]; self.dirty=False; self.saved_path=None

    @classmethod
    def load(cls, selected):
        selected=Path(selected).resolve(); project=cls()
        if selected.is_dir() and (selected/'train/annotations.json').is_file():
            sources=[(selected/'train/annotations.json','train'),(selected/'val/annotations.json','val')]
            if (selected/'unassigned/annotations.json').is_file(): sources.append((selected/'unassigned/annotations.json',''))
            project.saved_path=selected
        elif selected.is_file(): sources=[(selected,'')]
        else:
            annotation=selected/'annotations.json'
            if not annotation.is_file() and selected.name=='images': annotation=selected.parent/'annotations.json'
            if annotation.is_file(): sources=[(annotation,'')]
            else:
                paths=sorted(p for p in selected.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())
                if not paths: raise ValueError('No images or COCO annotations found')
                project.categories=[{'id':1,'name':'particle','supercategory':'particle'}]
                for p in paths:
                    with Image.open(p) as im: w,h=im.size
                    project.images.append({'path':p,'name':p.name,'width':w,'height':h,'annotations':[],'split':''})
                return project
        names={}; counter=1
        for source,split in sources:
            data=json.loads(source.read_text(encoding='utf-8'))
            if not isinstance(data.get('images'),list) or not isinstance(data.get('annotations'),list): raise ValueError('COCO images and annotations are required')
            category_map={}
            for cat in data.get('categories',[]):
                name=cat['name'].strip()
                if name not in names:
                    names[name]=counter; project.categories.append({'id':counter,'name':name,'supercategory':cat.get('supercategory') or name}); counter+=1
                category_map[int(cat['id'])]=names[name]
            by_image={}
            for ann in data['annotations']:
                ann=copy.deepcopy(ann)
                if int(ann['category_id']) not in category_map: raise ValueError('Annotation references unknown category')
                ann['category_id']=category_map[int(ann['category_id'])]
                by_image.setdefault(int(ann['image_id']),[]).append(ann)
            for rec in data['images']:
                name=Path(str(rec['file_name']).replace('\\','/'))
                candidates=[source.parent/'images'/name,source.parent/name,source.parent/'images'/name.name,source.parent/name.name]
                path=next((p.resolve() for p in candidates if p.is_file()),None)
                if path is None: raise ValueError(f'Missing image: {name}')
                with Image.open(path) as im: w,h=im.size
                if (w,h)!=(int(rec['width']),int(rec['height'])): raise ValueError(f'Incorrect image dimensions: {name}')
                project.images.append({'path':path,'name':name.name,'width':w,'height':h,'annotations':by_image.get(int(rec['id']),[]),'split':split})
        if not project.categories: project.categories=[{'id':1,'name':'particle','supercategory':'particle'}]
        if not project.images: raise ValueError('Dataset has no images')
        return project

    def masks(self,index):
        rec=self.images[index]
        return [dict(copy.deepcopy(a),mask=decode(a.get('segmentation',[]),rec['width'],rec['height'])) for a in rec['annotations']]

    def commit(self,index,masks):
        annotations=[]
        for item in masks:
            mask=item['mask']
            if not np.any(mask): continue
            ann={k:copy.deepcopy(v) for k,v in item.items() if k!='mask' and not k.startswith('_')}
            rle=encode(mask); binary=dict(rle,counts=rle['counts'].encode('ascii'))
            ann.update(segmentation=rle,area=float(mask_utils.area(binary)),bbox=mask_utils.toBbox(binary).tolist(),iscrowd=0)
            annotations.append(ann)
        self.images[index]['annotations']=annotations

    def rename(self,cat_id,name):
        name=name.strip()
        if not name: raise ValueError('Enter a label name')
        if any(c['id']!=cat_id and c['name'].casefold()==name.casefold() for c in self.categories): raise ValueError('Label name already exists')
        cat=next(c for c in self.categories if c['id']==cat_id)
        cat['name']=name;cat['supercategory']=name
        for rec in self.images:
            for ann in rec['annotations']:
                if ann['category_id']==cat_id: ann['noun_phrase']=name
        self.dirty=True

    def merge_categories(self,source_id,target_id):
        if source_id==target_id: raise ValueError('Choose two different labels')
        known={c['id']:c for c in self.categories}
        if source_id not in known or target_id not in known: raise ValueError('Unknown label')
        target_name=known[target_id]['name']
        for rec in self.images:
            for ann in rec['annotations']:
                if ann['category_id']==source_id:
                    ann['category_id']=target_id;ann['noun_phrase']=target_name
        self.categories=[c for c in self.categories if c['id']!=source_id]
        self.dirty=True

    def plan_split(self,scope,percent,seed):
        if not 0<percent<100: raise ValueError('Training percentage must be between 1 and 99')
        plan=[r['split'] for r in self.images]
        indices=[i for i,r in enumerate(self.images) if scope=='all' or not r['split']]
        if not indices: raise ValueError('No unassigned images remain')
        random.Random(seed).shuffle(indices)
        n=round(len(indices)*percent/100)
        if len(indices)>=2: n=max(1,min(len(indices)-1,n))
        for j,i in enumerate(indices): plan[i]='train' if j<n else 'val'
        return plan

    def apply_split(self,plan):
        if len(plan)!=len(self.images) or any(x not in ('train','val','') for x in plan): raise ValueError('Invalid split plan')
        for rec,value in zip(self.images,plan): rec['split']=value
        self.dirty=True

    def export(self,destination,progress=None):
        destination=Path(destination).resolve()
        if destination.exists() and any(destination.iterdir()): raise ValueError('Choose a new or empty output folder to preserve existing data')
        if not any(r['split']=='train' for r in self.images) or not any(r['split']=='val' for r in self.images): raise ValueError('Assign at least one Training and one Validation image')
        for rec in self.images:
            if rec['path']==destination or destination in rec['path'].parents: raise ValueError('Save into a separate output folder')
        destination.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.finetune-export-',dir=destination.parent))
        try:
            done=0
            for split in ('train','val',''):
                records=[r for r in self.images if r['split']==split]
                if not records and not split: continue
                folder=staging/(split or 'unassigned');(folder/'images').mkdir(parents=True)
                payload={'images':[],'annotations':[],'categories':copy.deepcopy(self.categories)}; used=set();ann_id=1
                for image_id,rec in enumerate(records,1):
                    name=rec['name'];stem=Path(name).stem;suffix=Path(name).suffix;serial=1
                    while name.casefold() in used: name=f'{stem}_{serial}{suffix}';serial+=1
                    used.add(name.casefold());shutil.copy2(rec['path'],folder/'images'/name)
                    payload['images'].append({'id':image_id,'file_name':name,'width':rec['width'],'height':rec['height']})
                    for annotation in rec['annotations']:
                        ann=copy.deepcopy(annotation);ann.update(id=ann_id,image_id=image_id);ann_id+=1;payload['annotations'].append(ann)
                    done+=1
                    if progress: progress(done,len(self.images))
                (folder/'annotations.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
            (staging/'dataset.json').write_text(json.dumps({'format':'BL62 Fine-tune Dataset','version':1,'unassigned':sum(not r['split'] for r in self.images)},indent=2),encoding='utf-8')
            if destination.exists(): destination.rmdir()
            os.replace(staging,destination)
        except BaseException:
            shutil.rmtree(staging,ignore_errors=True);raise
        self.saved_path=destination;self.dirty=False
        return destination
