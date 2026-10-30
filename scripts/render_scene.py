"""Illustrate the frozen city meshes and archived rays, without re-running RT."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
import drjit as dr
dr.set_thread_count(8)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
from matplotlib.transforms import Bbox
from PIL import Image

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output/scene')
parser.add_argument('--reuse-render', action='store_true')
args=parser.parse_args()
EXP=args.experiment.resolve()
OUT=args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(EXP),str(EXP/'support'),str(Path(__file__).resolve().parents[1]/'support')]
from public_multibounce_geometry import PublicGeometry
rec=json.loads((EXP/'cache/reference_parts/confirm_00000_rt.json').read_text())[28]
pos=np.array(rec['position'])
tx,rx=pos[:3],pos[3:]
cfg=json.loads((EXP/'configs/effective_config.json').read_text())
candidates,candidate_stats=PublicGeometry(cfg).candidates(pos)
cached=np.load(EXP/'cache/features_confirm.npz')['G3'][28,6:].reshape(64,40)
assert np.allclose(np.array([c['row'] for c in candidates]),cached[:len(candidates)],atol=1e-7)
valid_candidates=[c for c in candidates if c['valid']]

def split_mesh(material):
    """Local display derivatives only; retain exact source vertices and triangles."""
    lines=(EXP/f'assets/{material}.ply').read_text().splitlines()
    end=lines.index('end_header')+1
    nv=int(next(s for s in lines[:end] if s.startswith('element vertex ')).split()[-1])
    verts=lines[end:end+nv]; faces=lines[end+nv:]
    metadata=json.loads((EXP/f'assets/{material}_triangles.json').read_text())
    assert len(faces)==len(metadata)
    result={}
    for name,occluder in [('context',False),('occluder',True)]:
        fs=[f for f,m in zip(faces,metadata) if (m['building_id']==0)==occluder]
        header=[f'element face {len(fs)}' if s.startswith('element face ') else s for s in lines[:end]]
        path=OUT/f'{material}_{name}.ply'
        path.write_text('\n'.join(header+verts+fs)+'\n')
        result[name]=str(path)
    return result
mesh_parts={m:split_mesh(m) for m in ['concrete','glass']}
selected=[]
for order in range(3):
    matches=[(i,p) for i,p in enumerate(rec['paths']) if p['order']==order and p['z'][0]>=-110 and p['z'][1]<=1.8e-6]
    i,p=max(matches,key=lambda x:x[1]['z'][0])
    selected.append({'index':i,'order':order,'gain_db':p['z'][0],
                     'points':[tx.tolist()]+[h['vertex'] for h in p['hits']]+[rx.tolist()]})

T=mi.ScalarTransform4f
origin=np.array([-350.65822208,-194.96958264,440.])
target=np.array([120.,120.,26.])
halfwidth=205.
width,height=1800,1180
camera=T.look_at(origin=origin,target=target,up=[0,0,1]) @ T.scale([halfwidth,halfwidth,1])
scene={
    'type':'scene','integrator':{'type':'path','max_depth':12,'hide_emitters':True},
    'sensor':{'type':'orthographic','to_world':camera,
              'film':{'type':'hdrfilm','width':width,'height':height,'pixel_format':'rgba','rfilter':{'type':'tent'}},
              'sampler':{'type':'independent','sample_count':96}},
    'sky':{'type':'constant','radiance':{'type':'rgb','value':[.55,.57,.61]}},
    'sun':{'type':'directional','direction':[.4,.55,-1.0],
           'irradiance':{'type':'rgb','value':[1.6,1.55,1.5]}},
    'concrete':{'type':'ply','filename':mesh_parts['concrete']['context'],
                'bsdf':{'type':'diffuse','reflectance':{'type':'rgb','value':[.52,.54,.56]}}},
    'glass':{'type':'ply','filename':mesh_parts['glass']['context'],
             'bsdf':{'type':'roughplastic','alpha':.16,
                     'diffuse_reflectance':{'type':'rgb','value':[.16,.31,.41]}}},
    'road':{'type':'ply','filename':str(EXP/'assets/asphalt.ply'),
            'bsdf':{'type':'diffuse','reflectance':{'type':'rgb','value':[.19,.21,.23]}}},
    'display_ground':{'type':'rectangle','to_world':T.translate([120,120,-.08])@T.scale([125,125,1]),
                      'bsdf':{'type':'diffuse','reflectance':{'type':'rgb','value':[.73,.74,.71]}}},
}
for material in ['concrete','glass']:
    scene[f'{material}_occluder']={
        'type':'ply','filename':mesh_parts[material]['occluder'],
        'bsdf':{'type':'blendbsdf','weight':.18,
                'bsdf_0':{'type':'null'},'bsdf_1':scene[material]['bsdf']}}
colors=['#0072BD','#D95319','#7E2F8E']
rgb=[np.array(matplotlib.colors.to_rgb(c)) for c in colors]
for ray in selected:
    pts=np.array(ray['points']);o=ray['order']
    for j,(a,b) in enumerate(zip(pts[:-1],pts[1:])):
        scene[f'ray{o}_{j}']={'type':'cylinder','p0':a,'p1':b,'radius':.78,
            'bsdf':{'type':'diffuse','reflectance':{'type':'rgb','value':rgb[o]}}}
    for j,p in enumerate(pts[1:-1]):
        scene[f'hit{o}_{j}']={'type':'sphere','center':p,'radius':1.25,
            'bsdf':{'type':'diffuse','reflectance':{'type':'rgb','value':rgb[o]}}}
for name,p in [('tx',tx),('rx',rx)]:
    scene[name]={'type':'sphere','center':p,'radius':2.0,
                  'bsdf':{'type':'diffuse','reflectance':{'type':'rgb','value':[.05,.06,.07]}}}

if '--reuse-render' not in sys.argv:
    print('Rendering archived city and query 28 on CPU.',flush=True)
    loaded=mi.load_dict(scene)
    img=mi.render(loaded,spp=192,seed=28)
    bitmap=mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGBA,mi.Struct.Type.UInt8,True)
    bitmap.write(str(OUT/'scene_render.png'))
    print('Render saved.',flush=True)

# Reproduce the orthographic projection for typography and endpoint callouts.
forward=(target-origin);forward/=np.linalg.norm(forward)
right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
up=np.cross(right,forward)
def project(p):
    d=np.asarray(p)-target
    # Mitsuba's camera local +x points left in image coordinates.
    return (width/2+np.dot(d,right)*width/(2*halfwidth),
            height/2-np.dot(d,up)*width/(2*halfwidth))

font_manager.findfont('Times New Roman',fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,
                      'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False})
image=plt.imread(OUT/'scene_render.png')
fig,ax=plt.subplots(figsize=(7.2,4.72))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.imshow(image);ax.set_xlim(0,width);ax.set_ylim(height,0);ax.axis('off')
for candidate in sorted(candidates,key=lambda c:c['valid']):
    xy=np.array([project(p) for p in candidate['vertices']])
    ax.plot(xy[:,0],xy[:,1],color='#343D45',alpha=.65,lw=1.05,
            ls='-' if candidate['valid'] else '--',zorder=3 if candidate['valid'] else 2)
# Put the archived representative rays above coincident candidate overlays.
for ray in selected:
    xy=np.array([project(p) for p in ray['points']])
    ax.plot(xy[:,0],xy[:,1],color=colors[ray['order']],lw=1.6,zorder=5)
for p,label,offset in [(tx,'TX UAV\nz = 90.3 m',(18,-26)),(rx,'RX UGV\nz = 1.5 m',(-82,40))]:
    xy=project(p)
    ax.scatter(*xy,s=27,facecolor='white',edgecolor='black',linewidth=1,zorder=8)
    ax.annotate(label,xy=xy,xytext=offset,textcoords='offset points',fontsize=10.5,
                fontweight='bold',ha='left',va='center',
                bbox=dict(boxstyle='round,pad=.25',fc='white',ec='#444444',lw=.6),
                arrowprops=dict(arrowstyle='-',lw=.8,color='#222222'),zorder=9)

# A geometric scale dimension on the front edge of the actual urban footprint.
a,b=project([0,-9,0]),project([240,-9,0])
ax.annotate('',xy=a,xytext=b,arrowprops=dict(arrowstyle='<->',color='#333333',lw=.8))
mid=(np.array(a)+b)/2
ax.annotate('240 m',xy=mid,xytext=(0,12),textcoords='offset points',ha='center',fontsize=10,
            bbox=dict(fc='white',ec='none',pad=1,alpha=.9))
handles=[Line2D([0],[0],color=c,lw=2.0,label=l) for c,l in zip(colors,['Direct','Single reflection','Double reflection'])]
handles.extend([
    Line2D([0],[0],color='#343D45',alpha=.65,lw=1.05,ls='-',label='Valid candidates'),
    Line2D([0],[0],color='#343D45',alpha=.65,lw=1.05,ls='--',label='Invalid candidates')])
ax.legend(handles=handles,loc='upper right',bbox_to_anchor=(.99,.90),ncol=1,frameon=False,
          fontsize=9.5,handlelength=2.2,labelspacing=.4)
fig.savefig(OUT/'scene_preview.png',dpi=240,transparent=True)
# Trim transparent margins consistently in PNG and PDF, keeping upright text.
with Image.open(OUT/'scene_preview.png') as preview:
    w,h=preview.size
    left,top,right_edge,bottom=preview.getchannel('A').getbbox()
pad=12
left=max(0,left-pad);top=max(0,top-pad)
right_edge=min(w,right_edge+pad);bottom=min(h,bottom+pad)
fw,fh=fig.get_size_inches()
crop=Bbox.from_extents(left/w*fw,(h-bottom)/h*fh,right_edge/w*fw,(h-top)/h*fh)
fig.savefig(OUT/'scene_preview.png',dpi=240,transparent=True,bbox_inches=crop)
fig.savefig(OUT/'scene_preview.pdf',transparent=True,bbox_inches=crop)
plt.close(fig)
manifest={'query_id':28,'selection':'Lowest test query ID containing all three reflection orders above -110 dB; strongest ray per order.',
          'position_m':rec['position'],'selected_rays':selected,'camera_origin':origin.tolist(),
          'camera_target':target.tolist(),'rendering':'Mitsuba CPU rendering of exact frozen mesh vertices; visualization colors assigned for clarity.',
          'framing':'20 degree clockwise model rotation relative to previous view (camera azimuth +20 degrees); reduced camera elevation; transparent margins cropped with 12 pixel padding. Legend sits within the scene vertical extent.',
          'ground_and_roads':'Context geometry only: active RF mesh manifest includes concrete and glass. Road mesh shows the known road network; display_ground is a visualization plane.',
          'marker_sizes':'Enlarged for visibility, not physical antenna dimensions.',
          'transparent_building':{'building_id':0,'surface_opacity':.18,'reason':'Occludes the selected rays from the illustration camera.'},
          'candidate_stats':candidate_stats,
          'candidate_display':'All 11 retained candidates: 4 valid in solid dark gray, 7 invalid in dashed dark gray, alpha=0.65. Projected overlays expose complete hypotheses, including obstructed segments. Three selected archived rays overlaid in color above coincident candidates.',
          'all_candidates':[{'valid':c['valid'],'order':c['order'],'face_ids':[int(i) for i in c['ids']],'vertices':c['vertices'].tolist()} for c in candidates],
          'valid_candidates':[{'order':c['order'],'face_ids':[int(i) for i in c['ids']],'vertices':c['vertices'].tolist()} for c in valid_candidates],
          'source_sha256':{f:hashlib.sha256((EXP/f).read_bytes()).hexdigest() for f in ['assets/concrete.ply','assets/glass.ply','assets/asphalt.ply','configs/mesh_M2.json','cache/reference_parts/confirm_00000_rt.json']}}
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2))
print('Preview and provenance saved.',flush=True)
