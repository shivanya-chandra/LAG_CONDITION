
import sys, os, argparse, pandas as pd

from build_schedule import build_schedule

def schedule_signature(df: pd.DataFrame):
    
    df2 = df.sort_values("position")
    
    return tuple(zip(df2["pair"].tolist(), df2["encodingTask"].tolist(), df2["position"].tolist()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--name1", default="output_schedule_1.csv")
    ap.add_argument("--name2", default="output_schedule_2.csv")
    ap.add_argument("--seed1", type=int, default=101)
    ap.add_argument("--seed2", type=int, default=202)
    ap.add_argument("--max-restarts", type=int, default=1000)
    ap.add_argument("--drop-60-if-needed", action="store_true",
                    help="Allow fallback by removing lag-60 rows if needed.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out1 = os.path.join(args.outdir, args.name1)
    out2 = os.path.join(args.outdir, args.name2)


    df1 = build_schedule(args.input_csv, out1, seed=args.seed1,
                         max_restarts=args.max_restarts,
                         drop60_if_needed=args.drop_60_if_needed)
    sig1 = schedule_signature(df1)
    print(f"Wrote {out1}")

 
    seed2 = args.seed2
    attempts = 0
    while True:
        attempts += 1
        df2 = build_schedule(args.input_csv, out2, seed=seed2,
                             max_restarts=args.max_restarts,
                             drop60_if_needed=args.drop_60_if_needed)
        sig2 = schedule_signature(df2)
        if sig2 != sig1:
            break
        if attempts > 100:
            raise RuntimeError("Could not produce a second distinct schedule after 100 nearby seeds.")
        seed2 += 1
        print(f"Second schedule duplicated first; retrying with seed={seed2}...")

    print(f"Wrote {out2}")
    print(f"Seeds used: schedule_1={args.seed1}, schedule_2={seed2}")

if __name__ == "__main__":
    main()
