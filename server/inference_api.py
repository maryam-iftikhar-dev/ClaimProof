"""Local SegFormer-B2 inference API. Keeps the original model preprocessing and thresholds."""
import io, json, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import SegformerForSemanticSegmentation, SegformerConfig

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / 'model' / 'segformer_aug_best.pt'
METRICS = ROOT / 'model' / 'segformer_aug_metrics.json'
M = json.loads(METRICS.read_text())
THR = float(M['photo_threshold_5pct_false_alarms'])
PIX = float(M['pixel_threshold_chosen_on_val'])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
GRID = [['upper-left','upper-centre','upper-right'],['centre-left','centre','centre-right'],['lower-left','lower-centre','lower-right']]


def pick_stages(hs, sizes):
    out, j = [], len(hs)-1
    for c in reversed(sizes):
        while j >= 0 and hs[j].shape[1] != c: j -= 1
        if j < 0: raise RuntimeError(f'Cannot find SegFormer feature map with {c} channels')
        out.append(hs[j]); j -= 1
    return tuple(reversed(out))


class Net(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m.segformer(
            x,
            output_hidden_states=True
        )

        if out.hidden_states is None:
            return self.m(pixel_values=x).logits

        return self.m.decode_head(
            pick_stages(
                out.hidden_states,
                self.m.config.hidden_sizes
            )
        )

def rename_new_to_old(sd):
    pairs = [('attention.sequence_reduction.sequence_reduction','attention.self.sr'),('attention.sequence_reduction.layer_norm','attention.self.layer_norm'),('attention.q_proj','attention.self.query'),('attention.k_proj','attention.self.key'),('attention.v_proj','attention.self.value'),('attention.o_proj','attention.output.dense'),('layernorm_before','layer_norm_1'),('layernorm_after','layer_norm_2'),('mlp.fc1','mlp.dense1'),('mlp.fc2','mlp.dense2'),('decode_head.linear_projections','decode_head.linear_c')]
    out = {}
    for k,v in sd.items():
        k = re.sub(r'\.segformer\.stages\.(\d+)\.patch_embeddings\.',r'.segformer.encoder.patch_embeddings.\1.',k)
        k = re.sub(r'\.segformer\.stages\.(\d+)\.layer_norm\.',r'.segformer.encoder.layer_norm.\1.',k)
        k = re.sub(r'\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.',r'.segformer.encoder.block.\1.\2.',k)
        for a,b in pairs: k = k.replace(a,b)
        out[k] = v
    return out

_net = None
def get_net():
    global _net
    if _net is None:
        if not CKPT.exists(): raise RuntimeError(f'Model checkpoint missing: {CKPT}')
        cfg = SegformerConfig(num_labels=1, hidden_sizes=[64,128,320,512], depths=[3,4,6,3], num_attention_heads=[1,2,5,8], sr_ratios=[8,4,2,1], decoder_hidden_size=768)
        net = Net(SegformerForSemanticSegmentation(cfg))
        sd = torch.load(CKPT, map_location='cpu', weights_only=True)
        try: net.load_state_dict(sd)
        except RuntimeError: net.load_state_dict(rename_new_to_old(sd))
        _net = net.to(DEVICE).eval()
    return _net

def prepare(image):
    im = image.convert('RGB'); w,h = im.size; scale = 1024/max(w,h)
    if scale < 1: im = im.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BOX)
    buf=io.BytesIO(); im.save(buf,'JPEG',quality=90); buf.seek(0)
    return np.asarray(Image.open(buf).convert('RGB'))

@torch.no_grad()
def detect(im):
    net=get_net(); x=torch.from_numpy(np.ascontiguousarray(((im.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1)))[None]
    h,w=x.shape[2:]; H=((h+31)//32)*32; W=((w+31)//32)*32
    xp=torch.zeros(1,3,H,W); xp[:,:,:h,:w]=x; valid=torch.zeros(1,1,H,W); valid[:,:,:h,:w]=1
    with torch.autocast(DEVICE.type, enabled=DEVICE.type=='cuda'): logits=net(xp.to(DEVICE))
    logits=logits.float(); vq=F.interpolate(valid.to(DEVICE),size=logits.shape[-2:],mode='nearest')
    probq=torch.sigmoid(logits)*vq
    score=float((F.avg_pool2d(probq,9,stride=1,padding=4,count_include_pad=False)*vq).max())
    pm=torch.sigmoid(F.interpolate(logits,size=(H,W),mode='bilinear',align_corners=False))[0,0,:h,:w].cpu().numpy()
    return score,pm

def regions_of(pm):
    from scipy import ndimage
    labels,n=ndimage.label(pm>PIX); h,w=pm.shape; regions=[]
    for k in range(1,n+1):
        mask=labels==k; pct=100*mask.sum()/mask.size
        if pct < .1: continue
        ys,xs=np.nonzero(mask)
        regions.append({'location':GRID[min(2,int(3*ys.mean()/h))][min(2,int(3*xs.mean()/w))], 'area_percent':round(float(pct),1), 'confidence':round(float(pm[mask].mean()),2)})
    return sorted(regions,key=lambda r:-r['area_percent'])[:3]

app=FastAPI(title='ClaimProof inference API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:3000','http://127.0.0.1:3000'],allow_methods=['*'],allow_headers=['*'])
@app.get('/health')
def health(): return {'status':'ok','device':str(DEVICE),'model_loaded':_net is not None}
@app.post('/analyze')
async def analyze(file: UploadFile=File(...)):
    if file.content_type not in ('image/jpeg','image/png','image/webp','image/bmp'): raise HTTPException(415,'Upload a JPG, PNG, WEBP, or BMP image.')
    raw=await file.read()
    if len(raw)>15*1024*1024: raise HTTPException(413,'Image must be 15 MB or smaller.')
    try: im=prepare(Image.open(io.BytesIO(raw)))
    except Exception: raise HTTPException(400,'The uploaded file is not a readable image.')
    score,pm=detect(im); flagged=score>THR; regions=regions_of(pm)
    overlay=np.zeros_like(im,dtype=np.float32); overlay[...,0]=255
    alpha=(np.clip(pm,0,1)*.6)[...,None]
    heat=np.clip(im*(1-alpha)+overlay*alpha,0,255).astype(np.uint8)
    buf=io.BytesIO(); Image.fromarray(heat).save(buf,format='JPEG',quality=90)
    original_buf=io.BytesIO(); Image.fromarray(im).save(original_buf,format='JPEG',quality=90)
    if flagged:
        where=f"The strongest region is in the {regions[0]['location']} and covers about {regions[0]['area_percent']}% of the photo." if regions else 'No single region stood out clearly.'
        reason=f'The detector flagged this photo as likely AI-edited: its manipulation score is {score:.2f}, above the {THR:.2f} threshold. {where} This is an automated finding, so a human analyst should review the highlighted area and make the final call.'
    else: reason=f'The detector did not flag this photo: its manipulation score is {score:.2f}, below the {THR:.2f} threshold. No clear edited region was identified. This is an automated result, so a human analyst can still review the photo.'
    return {'score':round(score,4),'threshold':THR,'flagged':flagged,'verdict':'FLAGGED: likely AI-edited' if flagged else 'NOT FLAGGED','regions':regions,'reason':reason,'heatmap':'data:image/jpeg;base64,'+__import__('base64').b64encode(buf.getvalue()).decode(),'processed_image':'data:image/jpeg;base64,'+__import__('base64').b64encode(original_buf.getvalue()).decode(),'metrics':{'test_auc':M['test_auc'],'test_detection_rate':M['test_detection_rate'],'test_false_alarm_rate':M['test_false_alarm_rate'],'test_pixel_f1_mean_per_edited_photo':M['test_pixel_F1_mean_per_edited_photo']}}

if __name__=='__main__':
    import uvicorn
    get_net()
    uvicorn.run(app,host='127.0.0.1',port=8001)
