"""
torch.compile / CUDA-graph playground on a toy transformer-block stack.
Compares: eager | compile(default) | compile(reduce-overhead) | compile(max-autotune)
Reports per mode: ms/iter, GPU-kernel count, cudaLaunchKernel count, cudaGraphLaunch count,
and compile+warmup time. Exports a Chrome trace per mode.

Run:
  ENV=/home/cicichen/.conda/envs/nanovllm
  LD_LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/lib $ENV/bin/python h100_setup/torch_compile_playground.py
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, copy, json
from torch.profiler import profile, ProfilerActivity
torch.manual_seed(0)
dev="cuda"
D, NB, B = 512, 4, 16        # small dims + small batch -> launch-bound (so graphs help)

class Block(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.ln1=nn.LayerNorm(d); s.fc1=nn.Linear(d,4*d); s.fc2=nn.Linear(4*d,d); s.ln2=nn.LayerNorm(d)
    def forward(s,x):
        h=s.fc1(s.ln1(x)); h=F.gelu(h); h=s.fc2(h); x=x+h     # pointwise (LN,gelu,add) + 2 matmuls
        h=F.silu(s.ln2(x)); return x+0.5*h                    # more pointwise

base=nn.Sequential(*[Block(D) for _ in range(NB)]).to(dev).eval()
x=torch.randn(B,D,device=dev)

def make(mode):
    m=copy.deepcopy(base).to(dev).eval()
    return {"eager":lambda:m,
            "default":lambda:torch.compile(m),
            "reduce-overhead":lambda:torch.compile(m,mode="reduce-overhead"),
            "max-autotune":lambda:torch.compile(m,mode="max-autotune")}[mode]()

@torch.inference_mode()
def loop(fn,n):
    for _ in range(n): y=fn(x)
    return y

def cuda_time(fn,it=50,warm=25):
    loop(fn,warm); torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); loop(fn,it); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/it

def trace_counts(fn,path,it=5):
    loop(fn,3)
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
        loop(fn,it); torch.cuda.synchronize()
    p.export_chrome_trace(path)
    ev=json.load(open(path))["traceEvents"]
    gk=sum(1 for e in ev if e.get("cat")=="kernel")           # GPU kernel executions
    lk=sum(1 for e in ev if e.get("name")=="cudaLaunchKernel") # per-kernel CPU launches
    gl=sum(1 for e in ev if e.get("name")=="cudaGraphLaunch")  # graph replays
    return gk/it, lk/it, gl/it

print(f"toy: {NB} blocks, d={D}, batch={B}  (small → launch-bound)\n")
print(f"{'mode':<16}{'ms/iter':>9}{'GPU kern':>10}{'cudaLaunch':>12}{'cudaGraphLaunch':>17}{'compile+warm s':>16}")
print("-"*82)
for mode in ["eager","default","reduce-overhead","max-autotune"]:
    fn=make(mode)
    t0=time.time(); ms=cuda_time(fn); cw=time.time()-t0
    gk,lk,gl=trace_counts(fn, f"/home/cicichen/nano-vllm/h100_setup/trace_{mode}.json")
    print(f"{mode:<16}{ms:>9.3f}{gk:>10.0f}{lk:>12.0f}{gl:>17.0f}{cw:>16.1f}")
print("\ntraces: h100_setup/trace_{eager,default,reduce-overhead,max-autotune}.json")
print("open in https://ui.perfetto.dev — compare CPU launch calls + GPU gaps across modes")
