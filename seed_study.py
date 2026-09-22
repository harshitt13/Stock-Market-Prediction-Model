"""5 seeds on AAPL for the two neural models, 12-fold config."""
import sys, os, time
sys.path.insert(0, "src")
from concurrent.futures import ProcessPoolExecutor, as_completed

SEEDS = [0, 1, 2, 3, 4]

def run_seed(seed):
    import torch
    torch.set_num_threads(3)
    from experiments import run_single, save_run, run_path, cached_loader
    t0 = time.time()
    frame = run_single(
        "AAPL", "full", seed,
        load_raw=cached_loader(os.path.join("results", "raw")),
        min_train_size=1008, test_size=252, step_size=252,
        epochs=100, include_models=["lstm", "transformer"],
    )
    path = save_run(frame, run_path("AAPL", "seedstudy", seed,
                                    os.path.join("results", "seeds")))
    return {"seed": seed, "path": path, "rows": len(frame),
            "minutes": (time.time() - t0) / 60}

if __name__ == "__main__":
    started = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(run_seed, s): s for s in SEEDS}
        for f in as_completed(futures):
            r = f.result(); out.append(r)
            print(f"  seed {r['seed']} done: {r['rows']} rows, {r['minutes']:.1f} min",
                  flush=True)
    print(f"\nwall {(time.time()-started)/60:.1f} min on 5 workers x 3 threads")
