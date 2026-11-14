"""Measure inference overhead on the actual device, excluding loading/preprocessing."""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
from reid.model import OcclusionOSNet


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=100)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    results=[]
    for variant in ['plain','full']:
        model=OcclusionOSNet(751,variant).cuda().eval()
        for batch in [1,32]:
            x=torch.randn(batch,3,256,128,device='cuda')
            with torch.autocast('cuda'):
                for _ in range(30):model(x)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                elapsed=[]
                for _ in range(args.iterations):
                    t=time.perf_counter()
                    model(x)
                    torch.cuda.synchronize()
                    elapsed.append((time.perf_counter()-t)*1000)
            results.append({'variant':variant,'batch':batch,'inference_parameters':sum(p.numel() for name,p in model.named_parameters() if not name.startswith('classifier.')),'median_batch_ms':statistics.median(elapsed),'mean_batch_ms':statistics.mean(elapsed),'images_per_second':batch*1000/statistics.mean(elapsed),'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20})
        del model,x
        torch.cuda.empty_cache()
    report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'shape':[3,256,128],'precision':'AMP FP16','warmup':30,'iterations':args.iterations,'timing':'Python wall time with CUDA synchronization per iteration, model only; excludes preprocessing and retrieval. Run without concurrent training. Random initialized models, architecture benchmark, not accuracy evaluation.','results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
