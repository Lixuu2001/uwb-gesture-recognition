from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.font_manager import FontProperties, fontManager

FONT = 'C:/Windows/Fonts/msyh.ttc'
BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
fontManager.addfont(FONT)
fontManager.addfont(BOLD)
matplotlib.rcParams.update({'svg.fonttype':'none', 'pdf.fonttype':42,
    'font.family':'Microsoft YaHei', 'axes.unicode_minus':False})
W, H = 180, 189
fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor='white')
ax = fig.add_axes([0,0,1,1])
ax.set(xlim=(0,W), ylim=(H,0))
ax.axis('off')

C = {'ink':'#223649', 'blue':'#305B80', 'line':'#6C8295',
     'border':'#A1B4C3', 'pale':'#F5F8FA', 'header':'#E7EFF5',
     'mid':'#4D6579', 'white':'#FFFFFF'}
checks = []

def rect(x,y,w,h,fill='white',edge=None,lw=.65,z=2):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge or C['border'],lw=lw,zorder=z))

def text(x,y,s,size=8.5,bold=False,color=None,align='center',region=None):
    t=ax.text(x,y,s,ha=align,va='center',fontsize=size,color=color or C['ink'],
        fontproperties=FontProperties(fname=BOLD if bold else FONT),
        linespacing=1.55,zorder=5)
    if region: checks.append((s,t,region))
    return t

def line(points,color=None,lw=.75,style='-'):
    ax.plot(*zip(*points),color=color or C['line'],lw=lw,linestyle=style,zorder=1)

def arrow(a,b,color=None,lw=.8,style='-'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,
        linewidth=lw,color=color or C['line'],linestyle=style,
        shrinkA=0,shrinkB=0,zorder=3))

def box(x,y,w,h,s,size=8.5,bold=False,fill=None,edge=None,color=None):
    rect(x,y,w,h,fill or C['white'],edge)
    text(x+w/2,y+h/2,s,size,bold,color,region=(x+1,y+1,x+w-1,y+h-1))

# Application need, followed by the two directions used in manuscript section 5.2.
box(4,4,172,12,'金属加工车间无人搬运车的雷达手势导航指令交互',10.2,True,
    fill=C['blue'],edge=C['blue'],color='white')
line([(90,16),(90,20),(31,20),(31,24)])
line([(90,20),(119.5,20),(119.5,24)])
arrow((31,21),(31,24))
arrow((119.5,21),(119.5,24))
box(4,24,54,9,'三视角方向',9.4,True,fill=C['header'])
box(63,24,113,9,'单视角方向',9.4,True,fill=C['header'])

# The two radar modalities are independent inputs, not a shared physical encoder.
arrow((31,33),(31,39))
line([(119.5,33),(119.5,36),(90,36),(90,39)])
line([(119.5,36),(149,36),(149,39)])
arrow((90,36),(90,39))
arrow((149,36),(149,39))
box(4,39,54,11,'三视角 UWB 离散样本',8.5,fill=C['pale'])
box(63,39,54,11,'训练：多视角 UWB 样本\n部署：单视角输入',7.8,fill=C['pale'])
box(122,39,54,11,'单雷达 FMCW 连续流',8.5,fill=C['pale'])

cols=[
    (4,'研究内容一','开集指令识别\n与新指令注册',[
        ('逐视角独立编码','双路径表征 · 随机丢视角'),
        ('原型距离与决策聚合','决策级证据聚合'),
        ('已知识别与未知拒识','已知识别 · 未知拒识'),
        ('少样本注册与校准','原型标定 · 共形校准')]),
    (63,'研究内容二','跨视角重建\n与观测可靠性评估',[
        ('三级掩蔽重建','时域块 · 距离块 · 整视角'),
        ('多视角向单视角蒸馏','一致性约束 · 教师–学生蒸馏'),
        ('多证据观测可靠性评估','残差 · 表征分歧 · 原型距离'),
        ('异常告警与观测降级','故障告警 · 合成观测验证')]),
    (122,'研究内容三','连续流在线检测\n与承诺控制',[
        ('因果特征提取','历史窗口 · 递推归一化'),
        ('事件检测与指令识别','事件存在性 · 起点 · 类别'),
        ('事件级拒识与确认注册','未知事件缓存与确认注册'),
        ('证据累积与稳定承诺','等待 → 早判 → 承诺 → 冷却')])]

for x,number,title,steps in cols:
    arrow((x+27,50),(x+27,56))
    rect(x,56,54,94,fill='white',edge=C['border'],lw=.85)
    rect(x,56,54,19,fill=C['header'],edge=C['border'],lw=.65)
    text(x+27,60,number,7.2,color=C['mid'])
    text(x+27,68.3,title,9,True,region=(x+1,63,x+53,74))
    for i,(main,detail) in enumerate(steps):
        y=78+18*i
        rect(x+3,y,48,12,fill=C['pale'],edge=C['border'],lw=.55)
        text(x+27,y+6,main,8.5,False,region=(x+4,y+1,x+50,y+11))
        if i<3:
            # The final registration step in content I is conditional on confirmed support samples.
            arrow((x+27,y+12),(x+27,y+18),style='--' if x==4 and i==2 else '-')
            if i==2 and x in (4,122):
                text(x+30,y+15,'人工确认' if x==4 else '已知事件',6.2,color=C['mid'],align='left')

# Equal-width research lanes converge into a single evaluation and output chain.
for x in (31,90,149):
    line([(x,150),(x,155),(90,155)])
arrow((90,155),(90,161))
box(4,161,172,9,'跨受试者验证：开集识别性能 · 观测退化鲁棒性 · 事件风险与时延',8.6,
    fill=C['pale'],edge=C['border'])
arrow((90,170),(90,176))
box(4,176,172,9,'导航指令交互输出：指令执行  /  未知拒识  /  证据不足等待',9,True,
    fill=C['header'],edge=C['blue'])

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
failures=[]
for s,t,(x0,y0,x1,y1) in checks:
    bb=t.get_window_extent(renderer).transformed(ax.transData.inverted())
    # Inverted y-axis makes the transformed display box's y bounds reversed.
    tx0,tx1=sorted([bb.x0,bb.x1]); ty0,ty1=sorted([bb.y0,bb.y1])
    if tx0<x0-.15 or tx1>x1+.15 or ty0<y0-.15 or ty1>y1+.15:
        failures.append({'text':s,'bounds':[tx0,ty0,tx1,ty1],'allowed':[x0,y0,x1,y1]})

name='技术路线图_学术重建版'
fig.savefig(ROOT/f'{name}.png',dpi=600,facecolor='white')
fig.savefig(ROOT/f'{name}.svg',facecolor='white')
fig.savefig(ROOT/f'{name}.pdf',facecolor='white',metadata={'Title':'雷达手势导航指令交互技术路线','Author':''})
fig.savefig(ROOT/'preview.png',dpi=180,facecolor='white')
(ROOT/'layout_qa.json').write_text(json.dumps({'size_mm':[W,H],'text_checked':len(checks),'overflow':failures},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'text_checked':len(checks),'overflow':failures},ensure_ascii=False))
if failures: sys.exit(2)
