from pathlib import Path
from pypdf import PdfReader
import pypdfium2 as pdfium
p=Path(r'D:\uwb_gestures\outputs\technical_route_20260906')
f=p/'技术路线图_学术重建版.pdf'
r=PdfReader(f)
print('PDF pages',len(r.pages))
print('PDF size pts',r.pages[0].mediabox)
s=r.pages[0].extract_text()
for required in ['三视角方向','观测可靠性评估','少样本注册与校准','证据累积与稳定承诺','未知拒识']:
 print(required,required in s)
 assert required in s
pdf=pdfium.PdfDocument(str(f))
pdf[0].render(scale=1.9).to_pil().save(p/'pdf_qa.png')
print('PDF render complete')
