"""Trace WITH graph breaks (a host-sync op per layer, standing in for an untraceable
op like flash_attn) vs WITHOUT — to see compilation get fragmented."""
import torch, json
from torch.profiler import profile, ProfilerActivity
dev="cuda"; D=1024; L=4
W1=torch.randn(D,4*D,device=dev); W2=torch.randn(4*D,D,device=dev)
def make(brk):
    def f(x):
        for _ in range(L):
            h=torch.nn.functional.gelu(x@W1)
            if brk: _=h.sum().item()      # <- host sync => NATURAL graph break per layer
            x=x+(h@W2)
        return x
    return f
x=torch.randn(64,D,device=dev)
for tag,brk in [("with_breaks",True),("no_breaks",False)]:
    torch._dynamo.reset()
    try:
        exp=torch._dynamo.explain(make(brk))(x); gb,gc=exp.graph_break_count,exp.graph_count
    except Exception as e: gb,gc="?","?"
    torch._dynamo.reset()
    cf=torch.compile(make(brk))
    for _ in range(5): cf(x)
    torch.cuda.synchronize()
    path=f"/home/cicichen/nano-vllm/h100_setup/graphbreak_{tag}.json"
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
        for _ in range(5): cf(x)
        torch.cuda.synchronize()
    p.export_chrome_trace(path)
    ev=json.load(open(path))["traceEvents"]
    regions=sum(1 for e in ev if "Torch-Compiled Region" in str(e.get("name","")))
    print(f"{tag:<12} graph_breaks={gb} graph_count={gc}  compiled-regions/call={regions/5:.0f}  trace->h100_setup/graphbreak_{tag}.json")
