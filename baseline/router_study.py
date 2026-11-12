#!/usr/bin/env python3
"""Spend 6H compute where it changes the answer: one rule for the exit gate and the token count.

Rule: an image takes the action a that maximises  P(correct | what 3H knows, a) - lam * cost(a).  lam (accuracy per GMAC) is set by the budget.
Actions: 0 = exit at 3H, then 6H keeping each cached share of the patches (cache_logits.py; eighths of 196 by default).
Budget for every policy: the calibration-half cost of the baseline (gate 1.0, keep 50%), as in target80_gate_search.

A waste  Where the 6H compute of an entropy-gated policy goes: already-right / fixed / hopeless / harmed. Oracle = cheapest correct action per image.
B rule with entropy as the only information. The exact best policy "more tokens for higher entropy" is found by dynamic programming,
         without a threshold grid. B4 uses the team's actions (exit, 25, 50, 75%); B8 uses the full eighths grid.
         Check: B4 should land near the team's thresholds 0.8 / 1.25. Marginal table: accuracy gained per extra GMAC for each step, by entropy decile.
C same rule, better information. One default HistGradientBoosting regressor estimates the expected gain of running 6H,
         E[correct(6H at keep ratio r) - correct(3H) | 3H outputs, r], pooled over the keep ratios (5-fold out-of-fold on val, seed 20261112).
         An image exits when no ratio has estimated gain above lam * cost. C_ent sees the entropy only (control: should match B8).
         C_full sees the top-10 probabilities, both entropies, logit margin, a CLS-attention summary, the 192-d CLS feature and the
         predicted class's track record. Diagnostic: out-of-fold AUC for "3H wrong and 6H at 50% right", entropy alone vs all 3H outputs.
         (First attempt, val only, before any V2 run: one classifier per action for P(correct). It failed the control, C_ent 0.36 points
         below B8 on the holdout half, because taking the best of nine separately estimated numbers favours overestimates. Replaced.)
Everything is fitted on the calibration half; holdout and full-val numbers are exploratory (this validation set has been searched before).
D V2     Only with --v2. Thresholds and lam are carried over unchanged; C models are refit on all 50,000 val images. One run, no retuning.
Planned comparisons on V2 (paired bootstrap, 5000 resamples, seed 0; void if the V2 cost exceeds the comparator's by more than 1%):
         grid = B8 - B4 (does a finer token grid matter?)        info = C_full - B8 (does knowing more about the gain matter?)
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from analyze_adaptive_drop import save_json
from plot_h1_diagnosis import NARROW_MACS, wide_macs
from target80_gate_search import RATIOS as TEAM_RATIOS, codes_for

LAMS = np.r_[0.0, np.logspace(-3, 0.5, 141)]  # search resolution for lam in part C
BASELINE = {"kind": "fixed", "gate": 1.0, "ratio": 0.5}
OLD_KEYS = {0.25: "drop_25_correct", 0.5: "drop_50_correct", 0.75: "drop_75_correct", 1.0: "wide_correct"}


def load(path):
    c = torch.load(path, weights_only=True)
    ratios = [float(r) for r in c["ratios"]]
    y, z, att = c["target"].numpy(), c["narrow_logits"].float(), c["attention"].float()
    p3 = z.softmax(1)
    correct = np.stack([z.argmax(1).numpy() == y] + [c[f"wide_{round(1000 * r)}_logits"].argmax(1).numpy() == y for r in ratios], 1)
    top, zt = p3.topk(10, 1).values, z.topk(2, 1).values
    patch = att[:, 1:] / att[:, 1:].sum(1, keepdim=True)
    ent = lambda q: -(q * q.clamp_min(1e-12).log()).sum(1, keepdim=True)  # noqa: E731
    scalar = torch.cat([c["entropy"][:, None], top, ent(p3), zt[:, :1] - zt[:, 1:], zt[:, :1], z.std(1, keepdim=True), ent(patch),
                        patch.max(1, keepdim=True).values, att[:, :1]], 1).numpy()
    cost = np.r_[0, [wide_macs({"ratio": r, "mode": "drop", "k": 0}) for r in ratios]].astype(np.int64)
    return dict(y=y, correct=correct, entropy=c["entropy"].double().numpy(), scalar=scalar, cls=c["narrow_cls"].float().numpy(),
                pred=z.argmax(1).numpy(), paths=c["image_path"], ratios=ratios, cost=cost)


def team_options(entropy, policy, ratios):
    """Option index per image for a policy written in the team's format (codes index target80_gate_search.RATIOS)."""
    column = {round(r, 6): j + 1 for j, r in enumerate(ratios)}
    lut = np.array([column.get(round(r, 6), -1) for r in TEAM_RATIOS])
    codes = codes_for(entropy, policy)
    opt = np.where(codes < 0, 0, lut[np.maximum(codes, 0)])
    assert (opt >= 0).all(), "policy uses a keep ratio that is not in the cache"
    return opt


def tally(opt, c, cost):
    return c[np.arange(len(opt)), opt], int(NARROW_MACS) * len(opt) + int(cost[opt].sum())


def best_monotone(score, c, cost_g, budget_g):
    """Exact best policy 'action index never decreases as score rises' within a total-cost budget: Lagrangian + dynamic programming.
    c: (n, A) correctness for the allowed actions in cost order; cost_g: (A,) extra GMACs. Returns thresholds on score and lam."""
    order = np.argsort(score, kind="stable")
    s, v0 = score[order], c[order].astype(float)
    actions = np.arange(v0.shape[1])

    def run(lam):
        v, f, back = v0 - lam * cost_g[None], np.zeros(len(actions)), np.zeros((len(s), len(actions)), np.int8)
        for i in range(len(s)):
            best = np.maximum.accumulate(f)
            back[i] = np.maximum.accumulate(np.where(f == best, actions, 0))  # best previous action that is <= the current one
            f = best + v[i]
        a, act = int(f.argmax()), np.empty(len(s), np.int8)
        for i in range(len(s) - 1, -1, -1):
            act[i] = a
            a = back[i, a]
        return act

    lo, hi = 0.0, 2.0
    if cost_g[run(0.0)].sum() <= budget_g:
        hi = 0.0
    else:
        for _ in range(30):  # smallest lam whose cost fits the budget
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if cost_g[run(mid)].sum() <= budget_g else (mid, hi)
    act = run(hi)
    return np.array([s[np.argmax(act >= j)] if (act >= j).any() else np.inf for j in actions[1:]]), hi


def design(dtr, itr, dte, ite, which, ctr):
    """Inputs of the per-action classifiers for training rows and test rows (possibly another dataset). 3H outputs only."""
    if which == "ent":
        return dtr["scalar"][itr, :1], dte["scalar"][ite, :1]
    g, ptr, pte = ctr.mean(0), dtr["pred"][itr], dte["pred"][ite]  # track record of the predicted class: per-action accuracy, one pseudo-count prior
    s, n = np.zeros((1000, ctr.shape[1])), np.zeros(1000)
    np.add.at(s, ptr, ctr.astype(float))
    np.add.at(n, ptr, 1)
    etr = (s[ptr] - ctr + g) / (n[ptr, None] - 1 + 1)  # leave-one-out: a training row never sees its own label
    ete = (s[pte] + g) / (n[pte, None] + 1)
    return np.hstack([dtr["scalar"][itr], dtr["cls"][itr], etr]), np.hstack([dte["scalar"][ite], dte["cls"][ite], ete])


def fit_predict(xtr, ctr, xte, ratios):
    """Estimated gain of each action over exiting: column 0 (exit) is 0 by definition, the rest come from one pooled regressor."""
    r = np.asarray(ratios, np.float32)
    pool = lambda x: np.hstack([np.repeat(x.astype(np.float32), len(r), 0), np.tile(r, len(x))[:, None]])  # noqa: E731
    gain = (ctr[:, 1:].astype(np.float32) - ctr[:, :1]).reshape(-1)
    model = HistGradientBoostingRegressor(random_state=0).fit(pool(xtr), gain)
    return np.hstack([np.zeros((len(xte), 1), np.float32), model.predict(pool(xte)).reshape(len(xte), len(r)).astype(np.float32)])


def route(phat, lam, cost):
    return (phat - lam * cost[None] / 1e9).argmax(1)


def pick_lam(phat, c, cost, budget):
    rows = [(float(lam),) + (lambda t: (int(t[0].sum()), t[1]))(tally(route(phat, lam, cost), c, cost)) for lam in LAMS]
    return min((r for r in rows if r[2] <= budget), key=lambda r: (-r[1], r[2]))[0]


def waste(opt, c):
    sent, n3, got = opt > 0, c[:, 0], c[np.arange(len(opt)), opt]
    return dict(sent=int(sent.sum()), already_right=int((sent & n3 & got).sum()), fixed=int((sent & ~n3 & got).sum()),
                hopeless=int((sent & ~n3 & ~got).sum()), harmed=int((sent & n3 & ~got).sum()), wrong_but_not_sent=int((~sent & ~n3).sum()))


def oracle(c, cost, budget):
    fixable = ~c[:, 0] & c[:, 1:].any(1)
    cum = np.sort(np.where(c[:, 1:], cost[None, 1:], np.iinfo(np.int64).max).min(1)[fixable]).cumsum()
    used = int((cum <= budget - int(NARROW_MACS) * len(c)).sum())
    return dict(correct=int(c[:, 0].sum()) + used, top1=100 * (int(c[:, 0].sum()) + used) / len(c), fixable=int(fixable.sum()), fixes_used=used,
                gmacs_to_fix_all=(int(NARROW_MACS) * len(c) + int(cum[-1])) / len(c) / 1e9)


def boot(a, b):
    d, rng, means = a.astype(float) - b, np.random.default_rng(0), []
    for _ in range(50):  # 5000 paired resamples in chunks
        means.append(d[rng.integers(0, len(d), size=(100, len(d)))].mean(1))
    m = np.concatenate(means)
    return dict(diff_pp=100 * float(d.mean()), ci95_pp=[float(v) for v in np.percentile(m * 100, [2.5, 97.5])], p_le0=float((m <= 0).mean()))


def table(methods, ratios, c, cost, index, tag):
    rows, kept, share = [], {}, np.r_[0.0, ratios]
    for name, opt in methods.items():  # "baseline" comes first
        o = opt[index]
        corr, total = tally(o, c[index], cost)
        kept[name] = (corr, total)
        rows.append(dict(split=tag, method=name, n=len(index), correct=int(corr.sum()), top1=100 * float(corr.mean()), gmacs=total / len(index) / 1e9,
                         cost_change_percent=100 * (total / kept["baseline"][1] - 1), stage2_fraction=float((o > 0).mean()),
                         mean_keep_ratio_when_sent=float(share[o][o > 0].mean()) if (o > 0).any() else 0.0,
                         vs_baseline=boot(corr, kept["baseline"][0]), vs_team=boot(corr, kept["team_same_budget"][0]) if "team_same_budget" in kept else None))
    contrasts = {}
    for name, (x, y) in {"grid_B8_minus_B4": ("B8", "B4"), "info_Cfull_minus_B8": ("C_full", "B8")}.items():
        contrasts[name] = dict(boot(kept[x][0], kept[y][0]), cost_ratio=kept[x][1] / kept[y][1], void=bool(kept[x][1] > 1.01 * kept[y][1]))
    return rows, contrasts, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True, help="split_ids.json of the adaptive-drop experiment")
    ap.add_argument("--policies", type=Path, required=True, help="locked_policies.json")
    ap.add_argument("--old-cache", type=Path, help="adaptive-drop per_image.pt, for the known-answer check")
    ap.add_argument("--v2", type=Path, help="ImageNet-V2 logit cache; run once, after the val part is settled")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    V = load(args.val)
    n, C, e, cost, ratios = len(V["y"]), V["correct"], V["entropy"], V["cost"], V["ratios"]
    report = {"protocol": __doc__, "ratios": ratios, "gmacs_per_action": (cost / 1e9).tolist(), "checks": {}}
    if args.old_cache:  # hardware noise: 3H and full-6H paths differ by a handful of images, token-drop paths by up to ~30 (ties in the Top-K boundary)
        old = torch.load(args.old_cache, weights_only=True)
        order = old["image_id"].argsort()
        assert [old["image_path"][i] for i in order.tolist()] == V["paths"], "image order differs from the old cache"
        mism = {"3H": int((old["narrow_correct"][order].numpy() != C[:, 0]).sum())}
        mism.update({f"keep_{r}": int((old[OLD_KEYS[r]][order].numpy() != C[:, j + 1]).sum()) for j, r in enumerate(ratios) if r in OLD_KEYS})
        report["checks"]["correctness_mismatches_vs_old_cache"] = mism
        assert mism["3H"] <= 10 and mism["keep_1.0"] <= 10 and max(mism.values()) <= 60, mism
    split, policies = json.loads(args.split.read_text()), json.loads(args.policies.read_text())
    cal, hold, everything = np.array(sorted(split["calibration"])), np.array(sorted(split["holdout"])), np.arange(n)
    base, team = team_options(e, BASELINE, ratios), team_options(e, policies["same_budget"], ratios)
    report["checks"]["val_correct"] = dict(baseline=int(tally(base, C, cost)[0].sum()), team_same_budget=int(tally(team, C, cost)[0].sum()), expected=[39826, 40038])
    assert abs(report["checks"]["val_correct"]["baseline"] - 39826) <= 15 and abs(report["checks"]["val_correct"]["team_same_budget"] - 40038) <= 15, report["checks"]
    print("checks", report["checks"], flush=True)
    budget = tally(base[cal], C[cal], cost)[1]
    budget_g = (budget - int(NARROW_MACS) * len(cal)) / 1e9

    # B: the rule with entropy only
    sets = {"B4": [0] + [j + 1 for j, r in enumerate(ratios) if r in (0.25, 0.5, 0.75)], "B8": list(range(len(cost)))}
    locked = {}
    for name, allowed in sets.items():
        thr, lam = best_monotone(e[cal], C[cal][:, allowed], cost[allowed] / 1e9, budget_g)
        locked[name] = dict(allowed=allowed, thresholds=[float(t) if np.isfinite(t) else "+inf" for t in thr], lam_correct_per_gmac=lam,
                            keep_ratios=[ratios[a - 1] for a in allowed[1:]])
        print(name, "thresholds", dict(zip(locked[name]["keep_ratios"], locked[name]["thresholds"])), "lam", round(lam, 4), flush=True)
    apply_b = lambda ent, name: np.array(locked[name]["allowed"])[(ent[:, None] >= np.array([np.inf if t == "+inf" else t for t in locked[name]["thresholds"]])[None]).sum(1)]  # noqa: E731
    edges = np.quantile(e, np.linspace(0, 1, 11))
    band = np.clip(np.searchsorted(edges, e, side="right") - 1, 0, 9)
    acc = np.stack([C[band == b].mean(0) for b in range(10)])
    report["marginal_pp_per_gmac_by_entropy_decile"] = dict(
        decile_upper_edges=edges[1:].tolist(), accuracy_percent=(100 * acc).round(2).tolist(),
        steps=["exit->" + str(ratios[0])] + [f"{ratios[j - 1]}->{ratios[j]}" for j in range(1, len(ratios))],
        values=(100 * np.diff(acc, axis=1) / (np.diff(cost) / 1e9)[None]).round(2).tolist(),
        team_steps={"exit->0.25": (100 * (acc[:, ratios.index(0.25) + 1] - acc[:, 0]) / (cost[ratios.index(0.25) + 1] / 1e9)).round(2).tolist(),
                    "0.25->0.5": (100 * (acc[:, ratios.index(0.5) + 1] - acc[:, ratios.index(0.25) + 1]) / ((cost[ratios.index(0.5) + 1] - cost[ratios.index(0.25) + 1]) / 1e9)).round(2).tolist()})

    # C: the same rule with a learned estimate of P(correct | 3H outputs, action)
    phat = {s: np.zeros((n, len(cost)), np.float32) for s in ["ent", "full"]}
    auc = {s: np.zeros(n) for s in phat}
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=20261112).split(everything)):
        for s in phat:
            xtr, xte = design(V, tr, V, te, s, C[tr])
            phat[s][te] = fit_predict(xtr, C[tr], xte, ratios)
            fix50 = ~C[:, 0] & C[:, ratios.index(0.5) + 1]
            auc[s][te] = HistGradientBoostingClassifier(random_state=0).fit(xtr, fix50[tr]).predict_proba(xte)[:, 1]
        print("fold", fold, "done", flush=True)
    for s in phat:
        locked[f"C_{s}"] = dict(lam_correct_per_gmac=pick_lam(phat[s][cal], C[cal], cost, budget))
    fix50 = ~C[:, 0] & C[:, ratios.index(0.5) + 1]
    report["fixable_auc_out_of_fold"] = dict(raw_entropy=float(roc_auc_score(fix50, e)), model_entropy_only=float(roc_auc_score(fix50, auc["ent"])),
                                             model_all_3h_outputs=float(roc_auc_score(fix50, auc["full"])), positives=int(fix50.sum()))
    print("fixable AUC", report["fixable_auc_out_of_fold"], flush=True)
    report["locked"], report["calibration_budget_macs"] = locked, budget
    print("locked lam", {k: round(v["lam_correct_per_gmac"], 4) for k, v in locked.items()}, flush=True)

    def methods(d, pred):
        return {"baseline": team_options(d["entropy"], BASELINE, d["ratios"]), "team_same_budget": team_options(d["entropy"], policies["same_budget"], d["ratios"]),
                "B4": apply_b(d["entropy"], "B4"), "B8": apply_b(d["entropy"], "B8"),
                "C_ent": route(pred["ent"], locked["C_ent"]["lam_correct_per_gmac"], cost), "C_full": route(pred["full"], locked["C_full"]["lam_correct_per_gmac"], cost)}

    mv = methods(V, phat)
    report["val"], report["val_contrasts"] = [], {}
    for tag, idx in [("calibration", cal), ("holdout", hold), ("all", everything)]:
        rows, contrasts, _ = table(mv, ratios, C, cost, idx, tag)
        report["val"] += rows
        report["val_contrasts"][tag] = contrasts
    report["val_waste"] = {k: waste(mv[k], C) for k in ["baseline", "team_same_budget", "B8", "C_full"]}
    report["val_oracle_same_budget"] = oracle(C, cost, tally(base, C, cost)[1])
    choices = {f"val_{k}": v.astype(np.int8) for k, v in mv.items()}

    if args.v2:
        T = load(args.v2)
        assert T["ratios"] == ratios
        rows_t = np.arange(len(T["y"]))
        pred = {}
        for s in phat:
            xtr, xte = design(V, everything, T, rows_t, s, C)
            pred[s] = fit_predict(xtr, C, xte, ratios)
        mt = methods(T, pred)
        report["v2"], report["v2_contrasts"], kept = table(mt, ratios, T["correct"], cost, rows_t, "v2")
        report["checks"]["v2_correct"] = dict(baseline=int(kept["baseline"][0].sum()), team_same_budget=int(kept["team_same_budget"][0].sum()), expected=[6867, 6896])
        assert abs(report["checks"]["v2_correct"]["baseline"] - 6867) <= 15 and abs(report["checks"]["v2_correct"]["team_same_budget"] - 6896) <= 15, report["checks"]
        report["v2_waste"] = {k: waste(mt[k], T["correct"]) for k in ["baseline", "team_same_budget", "B8", "C_full"]}
        report["v2_oracle_same_budget"] = oracle(T["correct"], cost, kept["baseline"][1])
        say = lambda x: "void: cost differs" if x["void"] else "gain confirmed" if x["ci95_pp"][0] > 0 else "loss confirmed" if x["ci95_pp"][1] < 0 else "no difference confirmed"  # noqa: E731
        report["verdict"] = {k: dict(v, verdict=say(v)) for k, v in report["v2_contrasts"].items()}
        choices.update({f"v2_{k}": v.astype(np.int8) for k, v in mt.items()})
        print("VERDICT", json.dumps(report["verdict"]), flush=True)
    save_json(args.output / "report.json", report)
    np.savez(args.output / "choices.npz", **choices)
    for x in report["val"] + report.get("v2", []):
        if x["split"] != "calibration":
            t = x["vs_team"]
            print(f"{x['split']:8s} {x['method']:17s} top1 {x['top1']:.3f} gmacs {x['gmacs']:.4f} ({x['cost_change_percent']:+.2f}%) 6H {100 * x['stage2_fraction']:.1f}% keep {100 * x['mean_keep_ratio_when_sent']:.0f}%"
                  + (f" | vs team {t['diff_pp']:+.3f} [{t['ci95_pp'][0]:+.3f}, {t['ci95_pp'][1]:+.3f}]" if t else ""), flush=True)
    print("contrasts", json.dumps({k: report[k] for k in report if "contrasts" in k}), flush=True)
    print("waste/oracle", json.dumps({k: report[k] for k in report if "waste" in k or "oracle" in k}), flush=True)


if __name__ == "__main__":
    main()
