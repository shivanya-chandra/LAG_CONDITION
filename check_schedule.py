import sys
import pandas as pd
from collections import defaultdict

WINDOWS = {
    "0":  (0,0),
    "5":  (4,6),
    "10": (9,11),
    "20": (18,22),
    "30": (28,32),
    "60": (55,65),
}

def window_for(lc):
    s = str(lc).strip().lower()
    if s == "single":
        return None
    base = int(float(s))
    return WINDOWS.get(str(base), (max(0, base-1), base+1))

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 check_schedule.py <schedule_csv>")
        sys.exit(1)
    path = sys.argv[1]
    df = pd.read_csv(path)


    by_pair = defaultdict(list)
    for _, r in df.iterrows():
        by_pair[r["pair"]].append(r)

    
    computed = []
    lag_ok = []
    window_ok = []


    pos_map = {(r["pair"], r["encodingTask"]): r["position"] for _, r in df.iterrows()}

    for _, r in df.iterrows():
        lc = str(r["lag_condition"]).lower()
        if lc == "single":
            computed.append("")
            lag_ok.append(True)     
            window_ok.append(True)  
            continue

    
        rows = sorted(by_pair[r["pair"]], key=lambda x: x["position"])
        gen_pos = [x["position"] for x in rows if x["encodingTask"] == "generate"]
        fb_pos  = [x["position"] for x in rows if x["encodingTask"] == "feedback"]

        if len(gen_pos) == 1 and len(fb_pos) == 1:
            comp = fb_pos[0] - gen_pos[0] - 1
            computed.append(comp)
            try:
                actual = float(r["actual_lag"])
                lag_ok.append(actual == float(comp))
            except:
                lag_ok.append(False)
            w = window_for(lc)
            window_ok.append(True if (w is None) else (w[0] <= comp <= w[1]))
        else:
    
            computed.append("")
            lag_ok.append(False)
            window_ok.append(False)

    out = df.copy()
    out["computed_lag"] = computed
    out["lag_ok"] = lag_ok
    out["window_ok"] = window_ok

    out_path = path.replace(".csv", "") + "-checked.csv"
    out.to_csv(out_path, index=False)

    print(f"Wrote checked file: {out_path}")
    bad = out[~out["lag_ok"] | ~out["window_ok"]]
    if len(bad) == 0:
        print("All rows validate (lag_ok & window_ok are True).")
    else:
        print("Rows with issues:")
        print(bad[["pair","lag_condition","actual_lag","computed_lag","lag_ok","window_ok","position"]].head(20))
        print(f"... total problematic rows: {len(bad)}")

if __name__ == "__main__":
    main()
