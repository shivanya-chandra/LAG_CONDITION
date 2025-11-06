import sys, argparse, math, random
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

def lag_window(cond_str: str):
    s = str(cond_str).strip().lower()
    if s == "single":
        return None

    base = int(float(s))
    if str(base) in WINDOWS:
        return WINDOWS[str(base)]

    return (max(0, base-1), base+1)

def read_input(path):
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    req = {"cue","target","pair","lag_condition"}
    if not req.issubset(df.columns):
        raise RuntimeError(f"Input missing required columns: {req} (found: {list(df.columns)})")
    df["lag_condition"] = df["lag_condition"].astype(str).str.strip().str.lower()
    return df

def describe_counts(df):
    counts = df["lag_condition"].value_counts(dropna=False).to_dict()

    lines = []
    for k in sorted(counts.keys(), key=lambda x: str(x)):
        lines.append(f"  {k}: {counts[k]}")
    return "\n".join(lines)

def build_items(df):
    singles_df = df[df["lag_condition"]=="single"].copy()
    dupes_df   = df[df["lag_condition"]!="single"].copy()

    singles = []
    for idx, r in singles_df.iterrows():
        singles.append({
            "idx": idx, "cue": r["cue"], "target": r["target"], "pair": r["pair"],
            "lag_condition": "single"
        })

    dupes = []
    for idx, r in dupes_df.iterrows():
        w = lag_window(r["lag_condition"])
        if w is None:
            raise RuntimeError(f"Unexpected None window for non-single: {r['lag_condition']}")
        dupes.append({
            "idx": idx, "cue": r["cue"], "target": r["target"], "pair": r["pair"],
            "lag_condition": r["lag_condition"], "lag_min": w[0], "lag_max": w[1]
        })

    return singles, dupes

def try_place_all_pairs(N, dupes, rng, max_backtrack_steps=8000):
    """
    Backtracking placer with randomized restarts flavor:
    - Always place hardest (longer max lag, rarer window) first, but pick their lag & start at random among feasible.
    - If stuck, backtrack a few pairs.
    """
    
    dupes_sorted = sorted(dupes, key=lambda d: (d["lag_max"], d["lag_min"]), reverse=True)

    feasible = []
    for d in dupes_sorted:
        flist = {}
        for s in range(0, N-1):
            lags = []
            lmin, lmax = d["lag_min"], d["lag_max"]
            for L in range(lmin, lmax+1):
                p2 = s + L + 1
                if p2 < N:
                    lags.append(L)
            if lags:
                flist[s] = lags
        feasible.append(flist)

    slots = [None]*N  
    placed = []       

    i = 0
    steps = 0
    while i < len(dupes_sorted):
        steps += 1
        if steps > max_backtrack_steps:
            return None  

        d = dupes_sorted[i]
        f = feasible[i]
        
        cand = []
       
        start_positions = list(f.keys())
        rng.shuffle(start_positions)
        for s in start_positions:
            p1 = s
   
            Ls = list(f[s])
            mid = (d["lag_min"] + d["lag_max"])/2.0
            Ls.sort(key=lambda L: abs(L - mid))
            
            if len(Ls) > 2:
              
                extra = rng.choice(Ls)
                candidate_lags = list(dict.fromkeys([Ls[0], Ls[1] if len(Ls)>1 else Ls[0], extra]))
            else:
                candidate_lags = Ls
            rng.shuffle(candidate_lags)
            for L in candidate_lags:
                p2 = p1 + L + 1
                if p2 < N and slots[p1] is None and slots[p2] is None:
                    cand.append((p1, L, p2))
        if cand:
          
            p1, L, p2 = rng.choice(cand)
            slots[p1] = ("gen", i)
            slots[p2] = ("fb",  i)
            placed.append((i, p1, L, p2))
            i += 1
            continue

   
        back = 1 if len(placed) < 5 else rng.randint(1, min(5, len(placed)))
        for _ in range(back):
            if not placed: break
            j, pp1, LL, pp2 = placed.pop()
            slots[pp1] = None
            slots[pp2] = None
            if j < i:
                i = j  


    out = [{"encodingTask": None, "actual_lag": None, "item": dupes_sorted[slots[p][1]]} if slots[p] else None for p in range(N)]

    for p in range(N):
        if slots[p] is None: continue
        role, idx_in_sorted = slots[p]
        it = dupes_sorted[idx_in_sorted]
        if role == "gen":
            out[p] = {"encodingTask":"generate","actual_lag":None,"item":it}
        else:
           
            for q in range(p-1, -1, -1):
                if slots[q] and slots[q][0]=="gen" and dupes_sorted[slots[q][1]]["idx"] == it["idx"]:
                    out[q]["actual_lag"] = p - q - 1
                    out[p] = {"encodingTask":"feedback","actual_lag": p - q - 1, "item": it}
                    break
    return out

def build_schedule(in_path, out_path, seed=None, max_restarts=300, drop60_if_needed=False):
    if seed is not None:
        random.seed(seed)
    df = read_input(in_path)
    singles, dupes = build_items(df)


    N = 2*len(dupes) + len(singles)


    print("Input lag_condition counts:\n" + describe_counts(df))


    attempt = 0
    rng = random.Random(seed)
    order_shuffler = list(range(len(dupes)))

    while attempt < max_restarts:
        attempt += 1
        
        rng.shuffle(order_shuffler)
        dupes_shuffled = [dupes[i] for i in order_shuffler]

        placed_pairs = try_place_all_pairs(N, dupes_shuffled, rng, max_backtrack_steps=12000)
        if placed_pairs is None:
            continue 


        empty_positions = [i for i, v in enumerate(placed_pairs) if v is None]
        if len(empty_positions) < len(singles):
 
            continue

        if len(singles) > 0:
            step = len(empty_positions) / len(singles)
            chosen_positions = [empty_positions[int(round(i*step + step/2 - 0.5))] for i in range(len(singles))]
            used = set()
            
            for k, pos in enumerate(chosen_positions):
                if pos in used or pos < 0 or pos >= N:
                    found = None
                    for p in empty_positions:
                        if p not in used:
                            found = p; break
                    if found is None:
                        break
                    chosen_positions[k] = found
                    used.add(found)
                else:
                    used.add(pos)
        else:
            chosen_positions = []

        schedule = list(placed_pairs)
        for item, pos in zip(singles, chosen_positions):
            schedule[pos] = {"encodingTask":"single","actual_lag":"", "item": item}


        if any(v is None for v in schedule):
            
            continue


        problems = validate_schedule(schedule)
        if not problems:
        
            out_df = to_dataframe(schedule)
            out_df.to_csv(out_path, index=False)
            print(f"Wrote {len(out_df)} trials to {out_path} after {attempt} attempt(s).")
            return out_df


    if drop60_if_needed:
        print("Placement failed after restarts; retrying with lag 60 rows removed (as allowed).")
        df_reduced = df[df["lag_condition"].str.lower() != "60"].copy()
        out_df2 = build_schedule_reduced(df_reduced, out_path, seed=seed, max_restarts=max_restarts)
        return out_df2

    raise RuntimeError("Failed to place all duplicate items under lag constraints after multiple restarts. "
                       "Try increasing --max-restarts, changing --seed, adding fillers, or using --drop-60-if-needed.")

def build_schedule_reduced(df, out_path, seed=None, max_restarts=300):
    if seed is not None:
        random.seed(seed)
    print("Reduced input counts (lag 60 removed):\n" + describe_counts(df))
    singles, dupes = build_items(df)
    N = 2*len(dupes) + len(singles)
    rng = random.Random(seed)

    for attempt in range(1, max_restarts+1):
        placed_pairs = try_place_all_pairs(N, dupes, rng, max_backtrack_steps=12000)
        if placed_pairs is None:
            continue
    
        empty_positions = [i for i, v in enumerate(placed_pairs) if v is None]
        schedule = list(placed_pairs)
        if len(empty_positions) < len(singles):
            continue
        if len(singles) > 0:
            step = len(empty_positions) / len(singles)
            chosen_positions = [empty_positions[int(round(i*step + step/2 - 0.5))] for i in range(len(singles))]
        else:
            chosen_positions = []
        for item, pos in zip(singles, chosen_positions):
            schedule[pos] = {"encodingTask":"single","actual_lag":"", "item": item}

        problems = validate_schedule(schedule)
        if not problems:
            out_df = to_dataframe(schedule)
            out_df.to_csv(out_path, index=False)
            print(f"Wrote {len(out_df)} trials to {out_path} after {attempt} attempt(s) (lag 60 removed).")
            return out_df

    raise RuntimeError("Even without lag 60, failed to build schedule. Consider adding fillers.")

def validate_schedule(schedule):
    out_df = to_dataframe(schedule)
    problems = []
    occ = defaultdict(list)
    for _, r in out_df.iterrows():
        occ[r["pair"]].append(r)
    def window_for(lc):
        if str(lc).lower() == "single":
            return None
        try:
            base = int(float(str(lc)))
        except:
            return None
        return WINDOWS.get(str(base), (max(0, base-1), base+1))
    for k, items in occ.items():
        items = sorted(items, key=lambda r: r["position"])
        lc = str(items[0]["lag_condition"]).lower()
        if lc == "single":
            if len(items) != 1:
                problems.append(f"{k}: expected 1 single, found {len(items)}")
        else:
            if len(items) != 2:
                problems.append(f"{k}: expected 2 occurrences, found {len(items)}")
            else:
                pos1 = [r for r in items if r["encodingTask"]=="generate"]
                pos2 = [r for r in items if r["encodingTask"]=="feedback"]
                if not pos1 or not pos2:
                    problems.append(f"{k}: missing generate/feedback")
                else:
                    a_lag = pos2[0]["position"] - pos1[0]["position"] - 1
                    w = window_for(lc)
                    if w and not (w[0] <= a_lag <= w[1]):
                        problems.append(f"{k}: actual lag {a_lag} not in {w} for {lc}")
    return problems

def to_dataframe(schedule):
    rows = []
    for i, slot in enumerate(schedule, start=1):
        it = slot["item"]
        rows.append({
            "cue": it["cue"],
            "target": it["target"],
            "pair": it["pair"],
            "lag_condition": str(it.get("lag_condition")),
            "actual_lag": slot["actual_lag"] if slot["encodingTask"]!="single" else "",
            "encodingTask": slot["encodingTask"],
            "position": i
        })
    return pd.DataFrame(rows, columns=["cue","target","pair","lag_condition","actual_lag","encodingTask","position"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-restarts", type=int, default=300,
                    help="Randomized restart attempts before giving up (increase if you see placement failures).")
    ap.add_argument("--drop-60-if-needed", action="store_true",
                    help="If placement fails, automatically retry with lag 60 rows removed (allowed by spec).")
    args = ap.parse_args()
    build_schedule(args.input_csv, args.output_csv, seed=args.seed,
                   max_restarts=args.max_restarts, drop60_if_needed=args.drop_60_if_needed)

if __name__ == "__main__":
    main()
