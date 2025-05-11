#!/usr/bin/env python3
# run_inversion2.py – break 2 assemblers + 3 belts, repair, print direction dbg
# ---------------------------------------------------------------------------
import numpy as np, torch, scipy.ndimage as ndi
from draftsman.blueprintable import Blueprint
from draftsman.entity import TransportBelt, AssemblingMachine, Inserter, ElectricPole
from draftsman.utils import string_to_JSON
from src.representation.factory import Factory
from src.model import BinaryMatrixTransformCNN

CHECKPOINT = "best_model.pt"
GRID       = (20, 20)              # matrix H×W

# channel indices / slices
BELT_CH, INS_CH, ASM_CH = 1, 2, 0
DIR_SL, TIER_SL, KIND_SL = slice(4, 8), slice(8, 11), slice(11, 14)

# thresholds
TH_BELT, TH_INS, TH_ASM, TH_POLE = .05, .03, .10, .01

# ─────────── model + pad ────────────────────────────────────────────
def load_model(path=CHECKPOINT, size=20):
    m = BinaryMatrixTransformCNN(matrix_size=size)           # 21 chans
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m

def pad(mat, target=21):
    c = mat.shape[2]
    if c == target:  return mat
    if c > target:   return mat[:, :, :target]
    pad = np.zeros((*mat.shape[:2], target - c), mat.dtype)
    return np.concatenate([mat, pad], axis=2)

# ─────────── direction helpers ──────────────────────────────────────
def pick_dir(logits, T=.5):
    ex = np.exp(logits / T); ex /= ex.sum()
    return int(np.argmax(ex)) * 2                    # 0/2/4/6

def dbg_dir(mat, y, x, d):
    raw  = mat[y, x, DIR_SL]
    soft = np.exp(raw / .5); soft /= soft.sum()
    print(f"Dir‑dbg ({x:2d},{y:2d}) raw={raw.round(2)}  "
          f"soft={soft.round(2)}  → {d}")

# ─────────── matrix utilities ───────────────────────────────────────
def full_asm(mat):
    """Boolean mask where a full 3×3 assembler + item‑ID exists."""
    asm  = mat[:, :, ASM_CH] > TH_ASM
    full = ndi.minimum_filter(asm.astype(int), 3) == 1
    item = (mat[:, :, TIER_SL].sum(2) > 0)
    return full & item

def count(bp_str):
    out={}
    for e in string_to_JSON(bp_str)["blueprint"]["entities"]:
        out[e["name"]] = out.get(e["name"], 0) + 1
    return out

# ─────────── matrix → blueprint (robust) ───────────────────────────
def mats_to_bp(mat):
    bp = Blueprint()

    # belts
    for y, x in zip(*np.where(mat[:, :, BELT_CH] > TH_BELT)):
        bp.entities.append(TransportBelt("transport-belt", (x+.5, y+.5), 2))

    # assemblers
    centres=[]
    for y, x in zip(*np.where(full_asm(mat))):
        lx, ty = (x//3)*3, (y//3)*3
        cx, cy = lx+1, ty+1
        if (cx, cy) in centres: continue
        centres.append((cx, cy))
        bp.entities.append(AssemblingMachine("assembling-machine-1",
                                             (cx+.5, cy+.5),
                                             recipe="iron-gear-wheel"))

    # inserters
    done=set()
    for y, x in zip(*np.where(mat[:, :, INS_CH] > TH_INS)):
        if (x,y) in done: continue
        d = pick_dir(mat[y, x, DIR_SL])
        # override sideways directions using nearest assembler
        if d in (2,6) and centres:
            _, cy = min(centres, key=lambda p: abs(p[0]-x))
            d = 4 if y < cy else 0
        dbg_dir(mat, y, x, d)
        done.add((x,y))
        bp.entities.append(Inserter("fast-inserter", (x+.5, y+.5), d))

    # pole
    for y, x in zip(*np.where(mat[:, :, 3] > TH_POLE)):
        if not any(int(e.position.x)==x and int(e.position.y)==y for e in bp.entities):
            bp.entities.append(ElectricPole("small-electric-pole", (x+.5, y+.5)))
            break
    return bp.to_string()

# ─────────── pristine gear‑wheel block ─────────────────────────────
def build_block():
    bp = Blueprint()
    for cx in (4.5, 8.5, 12.5):
        bp.entities.append(AssemblingMachine("assembling-machine-1",
                                             (cx, 7.5), "iron-gear-wheel"))
    for x in range(2, 15):
        bp.entities.append(TransportBelt("transport-belt", (x+.5, 3.5), 2))
        bp.entities.append(TransportBelt("transport-belt", (x+.5, 10.5), 2))
    for cx in (4.5, 8.5, 12.5):
        bp.entities.append(Inserter("fast-inserter", (cx, 4.5), 4))  # south
        bp.entities.append(Inserter("fast-inserter", (cx, 9.5), 0))  # north
    bp.entities.append(ElectricPole("small-electric-pole", (8.5, 5.5)))
    return bp.to_string()

# ─────────── main demo ─────────────────────────────────────────────
def main():
    # pristine → matrix
    orig_bp  = build_block()
    orig_mat = pad(Factory.from_str(orig_bp).get_matrix(GRID))

    # ---------- BREAK: remove 3 belts & 2 assemblers ----------------
    broken = orig_mat.copy()

    # remove three belt tiles (centre of row 3)
    row = 3
    belt_xs = np.where(broken[row, :, BELT_CH] > TH_BELT)[0]
    mid = len(belt_xs)//2
    for x in belt_xs[mid-1 : mid+2]:
        broken[row, x, BELT_CH] = 0
        broken[row, x, DIR_SL] = broken[row, x, TIER_SL] = broken[row, x, KIND_SL] = 0

    # remove two assembler blobs (left‑most & right‑most)
    asm_mask = full_asm(broken)
    labels, n = ndi.label(asm_mask)
    if n >= 2:
        blobs=[]
        for i in range(1, n+1):
            ys, xs = np.where(labels == i)
            blobs.append((xs.mean(), xs, ys))         # (centre‑x, xs, ys)
        blobs.sort(key=lambda t: t[0])                # left→right
        for _, xs, ys in (blobs[0], blobs[-1]):
            broken[xs, ys, ASM_CH] = 0
            broken[xs, ys, TIER_SL] = 0
    else:
        print("DEBUG: fewer than 2 assembler blobs—skipping ASM removal")

    # ---------- REPAIR LOOP ----------------------------------------
    model = load_model()
    mats  = [broken]

    with torch.no_grad():
        for _ in range(4):    # ≤5 passes total
            inp = torch.tensor(mats[-1]).permute(2,0,1).unsqueeze(0).float()
            nxt = model.predict(inp)[0].cpu().numpy().transpose(1,2,0)
            mats.append(nxt)

            belts_ok = (nxt[:,:,BELT_CH] > TH_BELT).sum() == \
                       (orig_mat[:,:,BELT_CH] > TH_BELT).sum()
            asm_ok   = full_asm(nxt).sum() == full_asm(orig_mat).sum()
            if belts_ok and asm_ok:
                break

    # ---------- PRINT COUNTS & BP STRINGS --------------------------
    blueprints = [orig_bp] + [mats_to_bp(m) for m in mats]
    labels = ["ORIGINAL", "BROKEN"] + [f"AFTER {i}×" for i in range(1,len(blueprints)-1)]

    for lab, bp in zip(labels, blueprints):
        print(f"{lab:9}: {count(bp)}")
    for lab, bp in zip(labels, blueprints):
        print(f"\n----- {lab} -----\n{bp}")

if __name__ == "__main__":
    main()
