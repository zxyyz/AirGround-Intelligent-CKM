import os
from common import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
FONTDIR=Path(os.environ.get('TIMES_NEW_ROMAN_DIR', 'local-fonts/TimesNewRoman'))
COLORS=['#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F']

def setup():
    from fontTools.ttLib import TTFont
    fontinfo=[]
    for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        p=FONTDIR/name;assert p.exists();font_manager.fontManager.addfont(str(p));f=TTFont(p)
        names={i:sorted({n.toUnicode() for n in f['name'].names if n.nameID==i}) for i in [0,1,2,4,6,8,13]}
        assert any('Times New Roman' in v for v in names[1]);assert any('Microsoft supplied font' in v for v in names[13])
        fontinfo.append(dict(file_name=name,sha256=sha(p),metadata=names));f.close()
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.labelsize':12,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight','axes.prop_cycle':matplotlib.cycler(color=COLORS)})
    save(EXP/'results/font_provenance.json',dict(family='Microsoft Times New Roman',files=fontinfo,distribution='font files excluded from all outputs; reference installed local fonts only'))
    return plt

def savefig(fig,name):
    fig.savefig(EXP/f'figures/{name}.pdf',metadata={'Creator':'Python Matplotlib; Microsoft Times New Roman; frozen GOSPA scale analysis'})
    fig.savefig(EXP/f'figures/{name}.png',dpi=180);plt.close(fig)
