from common import B,config,write,sha
import sys
import engine as e
def main():
    assert len(sys.argv)==2 and sys.argv[1] in ['timemil','shuffle'];arm=sys.argv[1]
    m,o,g=e.make(arm);s=e.capture(m,o,g,0,0);path=B/'seed'/(arm+'.pt')
    assert not path.exists();e.atomic_save(path,s)
    write(B/'seed'/(arm+'.json'),dict(path=str(path.relative_to(B)),sha256=sha(path),state_sha256=e.semantic(s),new_UWB_fit=False))
if __name__=='__main__':main()
