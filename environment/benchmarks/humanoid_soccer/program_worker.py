"""Private JSON-lines worker for the non-exec AST control language."""
import json
import resource
import sys
from pathlib import Path

# -I -S excludes user packages and inherited Python paths. Only this trusted
# module is added; model text cannot import it or any filesystem/network API.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from control_program import Program

resource.setrlimit(resource.RLIMIT_AS,(192*1024**2,192*1024**2))
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
program=None;memory={}
for line in sys.stdin:
    try:
        if len(line)>262144:raise ValueError('Input too large')
        request=json.loads(line)
        if 'code' in request:
            program=Program(request['code']);memory={};result={'ready':True}
        else:
            result={'action':program.control(request['obs'],memory),'memory':memory}
        encoded=json.dumps(result,allow_nan=False)
        if len(encoded)>65536:raise ValueError('Result and memory must fit in 64 KiB')
    except Exception as e:
        encoded=json.dumps({'error':type(e).__name__+': '+str(e)[:1000]})
    print(encoded,flush=True)
