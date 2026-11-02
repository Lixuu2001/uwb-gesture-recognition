from pathlib import Path
root=Path(__file__).resolve().parent
src=Path(r'D:\uwb_gestures\outputs\route_template_pptx_20260906\.build\build.mjs').read_text(encoding='utf-8-sig')
src=src.replace('D:/uwb_gestures/outputs/route_template_pptx_20260906','D:/uwb_gestures/outputs/proposal_unified_20260906')
start=src.index('// Preserve the two-direction topology')
end=src.index('const cards=[',start)
src=src[:start]+'''// Dataset-specific inputs and the revised research order.
line(527.5,732,527.5,748,'#126B9A');
line(355,748,870,748,'#126B9A');
down(355,748,23,C.teal,15);down(870,748,23,C.purple,15);
box(19,772,672,74,grad('#E0F2F0','#F6FAF9'),C.teal,9,2);
box(702,772,335,74,grad('#F1E7F6','#FCF9FD'),C.purple,9,2);
asset(42,780,87,62,42,780,87,62,'多视角UWB训练数据示意');
text(140,780,537,29,'UWB 离散手势样本',24,C.teal,true,'center');
text(140,812,537,26,'多视角训练与参照，单雷达表征及开集评测',20,'#1E2931',false,'center');
text(710,780,319,29,'FMCW 连续记录',24,C.purple,true,'center');
text(710,812,319,26,'逐帧因果推理与事件级验证',19,'#1E2931',false,'center');
line(355,846,355,859,C.teal);line(184,859,527,859,C.teal);
down(184,859,17,C.teal,14);down(527,859,17,C.orange,14);
down(870,846,30,C.purple,15);

'''+src[end:]
a=src.index('const cards=[');b=src.index('for(const c of cards)',a)
src=src[:a]+'''const cards=[
 {x:18,w:335,c:C.teal,light:'#E1F2EF',label:'研究内容一',title:'面向单雷达部署的手势表征学习\\n与多视角知识迁移',sub:'',icon:[79,884,30,32],
 bullets:['单雷达幅度与运动表征学习','重建补全与知识蒸馏对照','迁移收益与观测退化评估'],foot:'跨用户性能 · 最差用户 · 适用边界'},
 {x:363,w:328,c:C.orange,light:'#FFF0DE',label:'研究内容二',title:'雷达手势指令的开集识别\\n与少样本新指令注册',sub:'',icon:[445,884,29,32],
 bullets:['标准化原型距离与未知拒识','新指令注册与已有类尺度校准','单雷达校准与多视角参照'],foot:'已知识别 · 未知拒识 · 旧类保持'},
 {x:702,w:335,c:C.purple,light:'#F1E7F6',label:'研究内容三',title:'雷达手势连续流的在线检测\\n与指令承诺控制',sub:'',icon:[783,884,31,32],
 bullets:['因果事件检测与指令识别','事件级未知拒识与确认注册','等待 → 早判 → 稳定承诺 → 冷却'],foot:'错误承诺率—决策延迟权衡曲线'}
];
'''+src[b:]
src=src.replace("text(c.x+8,929,c.w-16,30,c.title,21.3,c.c,true,'center');", "text(c.x+8,927,c.w-16,59,c.title,18.3,c.c,true,'center');")
src=src.replace("text(c.x+12,963,c.w-24,25,c.sub,18.5,'#6B7480',false,'center');",'')
src=src.replace("c.x===363?16.3:18",'16.5')
src=src.replace('训练期借三视角学表征 → 部署期单雷达连续运行 → 证据不足时等待或拒识','单雷达表征基础 → 开集识别与注册 → 连续流检测与指令确认')
src=src.replace('跨用户可靠 · 传感器失效可降级 · 指令词表可扩充 · 时延与误触发可权衡','UWB 与 FMCW 分别训练和校准，衔接拒识、注册与确认机制')
src=src.replace('技术路线图_模板版_可编辑.pptx','技术路线图_研究内容统一版.pptx')
src=src.replace("text(405,699,423,23,'UWB 距离—慢时间矩阵 / FMCW 距离—多普勒图',16.8", "text(405,699,423,23,'表征学习 → 开集指令判别 → 连续流在线决策',17.2")
# The Word document provides its own caption beneath the figure.
src=src.replace("text(65,1453,925,27,'图1  面向金属加工车间无人搬运车的雷达手势导航指令交互技术路线',20,'#183454',true,'center');",'')
(root/'.build'/'build.mjs').write_text(src,encoding='utf-8')
