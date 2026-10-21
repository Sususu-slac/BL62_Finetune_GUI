"""Standalone CPU conversion worker, also copied beside each training launch."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import uuid

PREFIX='BL62_CONVERSION '

def progress(value,message):
    print(PREFIX+json.dumps({'progress':value,'message':message}),flush=True)

def convert(base_path,training_path,output_path,fallback_base=None):
    import torch
    base_path,training_path,output_path=map(Path,(base_path,training_path,output_path))
    if output_path.resolve() in {training_path.resolve(),base_path.resolve()}:raise ValueError('Output must differ from input checkpoints')
    if output_path.exists():raise ValueError('Inference output already exists')
    # Memory mapping avoids loading optimizer tensors into RAM. All tensors remain on CPU.
    def load(path):
        data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if not isinstance(data,dict):raise ValueError(f'Checkpoint is not a dictionary: {path}')
        return data
    progress(5,'Loading training checkpoint…');training=load(training_path);weights=training.get('model')
    if not isinstance(weights,dict) or not weights:raise ValueError('Training checkpoint has no model weights')
    progress(20,'Loading inference template…');base=load(base_path);base=base.get('model',base)
    if not isinstance(base,dict):raise ValueError('Invalid inference template')
    if not any(k.startswith('detector.') for k in base):
        if not fallback_base:raise ValueError('Inference template requires detector-prefixed SAM3 weights')
        base_path=Path(fallback_base);base=load(base_path);base=base.get('model',base)
        if not isinstance(base,dict) or not any(k.startswith('detector.') for k in base):raise ValueError('No compatible SAM3 inference template found')
    if output_path.resolve()==base_path.resolve():raise ValueError('Output must differ from inference template')
    progress(35,'Checking weight names and shapes…');converted=OrderedDict(base);matched=[];errors=[]
    for key,value in weights.items():
        if not torch.is_tensor(value):raise ValueError('Non-tensor model weight: '+key)
        target_key=key if key.startswith(('detector.','tracker.')) else 'detector.'+key
        target=base.get(target_key)
        if not torch.is_tensor(target) or tuple(target.shape)!=tuple(value.shape):errors.append(key);continue
        converted[target_key]=value;matched.append(target_key)
    # SAM3 text training omits the optional SAM2 interactive neck; retain it from the template.
    retained=[k for k in base if k.startswith('detector.') and k not in matched]
    missing=[k for k in retained if not k.startswith('detector.backbone.vision_backbone.sam2_convs.')]
    if errors or missing:raise ValueError(f'Incompatible checkpoints: {len(errors)} unmatched training weights, {len(missing)} missing detector weights. '+', '.join((errors+missing)[:5]))
    if not matched:raise ValueError('No matching trained weights')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    temp=output_path.with_name(output_path.name+'.'+uuid.uuid4().hex+'.tmp')
    report_path=output_path.with_suffix('.report.json');report_temp=temp.with_suffix('.json')
    try:
        progress(50,'Writing inference model…');torch.save(converted,temp)
        progress(85,'Verifying saved model…');saved=load(temp)
        if set(saved)!=set(converted):raise ValueError('Saved checkpoint keys did not match')
        for key,value in converted.items():
            if not torch.is_tensor(saved[key]) or saved[key].shape!=value.shape or saved[key].dtype!=value.dtype:raise ValueError('Saved weight mismatch: '+key)
        report={'base':str(base_path),'training_checkpoint':str(training_path),'output':str(output_path),'matched':len(matched),'retained_template_weights':len(base)-len(matched),'retained_detector_keys':retained,'epoch':training.get('epoch'),'output_bytes':temp.stat().st_size}
        report_temp.write_text(json.dumps(report,indent=2),encoding='utf-8')
        temp.replace(output_path);report_temp.replace(report_path)
        progress(100,'Conversion complete');return report
    finally:
        temp.unlink(missing_ok=True);report_temp.unlink(missing_ok=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--training',required=True);parser.add_argument('--output',required=True);parser.add_argument('--fallback-base')
    args=parser.parse_args()
    try:convert(args.base,args.training,args.output,args.fallback_base)
    except Exception as error:
        print(PREFIX+json.dumps({'error':str(error)}),flush=True);raise SystemExit(1)
