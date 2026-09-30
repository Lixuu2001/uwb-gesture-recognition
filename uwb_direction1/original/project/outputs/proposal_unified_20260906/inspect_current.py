from pathlib import Path
from zipfile import ZipFile
from lxml import etree
from PIL import Image
p=Path(r'C:\Users\19252\Desktop\2-萧子愚-2024202212_修改版.docx')
out=Path(r'D:\uwb_gestures\outputs\proposal_unified_20260906')
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with ZipFile(p) as z:
 root=etree.fromstring(z.read('word/document.xml'))
 lines=[]
 for i,para in enumerate(root.xpath('//w:body//w:p',namespaces=ns)):
  s=''.join(para.xpath('.//w:t/text()',namespaces=ns))
  imgs=para.xpath('.//a:blip/@r:embed',namespaces={**ns,'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'})
  if s or imgs: lines.append(f'{i}: {s}' + (f' [IMAGES {imgs}]' if imgs else ''))
 (out/'source_text.txt').write_text('\n'.join(lines),encoding='utf-8')
 print('paragraphs', len(lines))
 print('MEDIA:')
 for n in z.namelist():
  if n.startswith('word/media/'):
   dest=out/Path(n).name
   dest.write_bytes(z.read(n))
   try:
    im=Image.open(dest); print(dest.name,im.size,im.format)
   except: print(dest.name)
 (out/'source_rels.xml').write_bytes(z.read('word/_rels/document.xml.rels'))

