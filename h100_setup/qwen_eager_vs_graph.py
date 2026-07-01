import os, time, json, torch
from nanovllm import LLM, SamplingParams
from torch.profiler import profile, ProfilerActivity
MODE=os.environ.get("MODE","graph")
path=os.path.expanduser(os.environ.get("MODEL","~/huggingface/Qwen3-0.6B"))
llm=LLM(path, enforce_eager=(MODE=="eager"), max_model_len=1024)
llm.add_request(list(range(32)), SamplingParams(temperature=0.6, max_tokens=300, ignore_eos=True))
def step(): return llm.step()[1]
for _ in range(25): step()                 # warm: prefill + some decode
K=20; n=0
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
    t0=time.perf_counter()
    while n<K and not llm.is_finished():
        if step()<0: n+=1
    torch.cuda.synchronize()
ms=(time.perf_counter()-t0)/max(n,1)*1e3
tr=f"/home/cicichen/nano-vllm/h100_setup/qwen_{MODE}.json"; p.export_chrome_trace(tr)
ev=json.load(open(tr))["traceEvents"]
gk=sum(1 for e in ev if e.get("cat")=="kernel")
lk=sum(1 for e in ev if e.get("name")=="cudaLaunchKernel")
gl=sum(1 for e in ev if e.get("name")=="cudaGraphLaunch")
print(f"MODE={MODE:<6} ms/decode={ms:6.3f}  over {n} steps -> per step: GPU_kernels~{gk//max(n,1)} cudaLaunchKernel~{lk//max(n,1)} cudaGraphLaunch~{gl//max(n,1)}")
