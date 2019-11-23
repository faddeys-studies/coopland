import argparse
import os
import numpy as np


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("model_dir_or_results_txt")
    cli.add_argument("--stuck-thr", type=int)
    opts = cli.parse_args()

    pth = opts.model_dir_or_results_txt
    if os.path.isdir(pth):
        pth = os.path.join(pth, "test-results.txt")

    results = np.loadtxt(pth, int)
    if results.ndim == 1:
        results = np.expand_dims(results, 1)

    res_max = results.max(axis=1)
    print("N:", results.shape[0])
    print("Max-Mean:", res_max.mean())
    print("Min-Mean:", results.min(axis=1).mean())
    print("Mean:", results.mean())

    stuck_thr = opts.stuck_thr if opts.stuck_thr is not None else results.max()
    print("Stuck rate:", np.sum(res_max >= stuck_thr) / results.shape[0])

    res_exited = results[res_max < stuck_thr]
    print("Exited Max-Mean:", res_exited.max(axis=1).mean())
    print("Exited Min-Mean:", res_exited.min(axis=1).mean())
    print("Exited Mean:", res_exited.mean())


if __name__ == "__main__":
    main()
