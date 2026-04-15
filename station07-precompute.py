"""
precompute.py — Station 07 QML Showdown
Generates the 150-point result grid (5 sample_sizes × 6 noise_levels × 5 circuit_depths)
for both classical and quantum classifiers.

Since this wraps a *simulated* model (real model files slot in via models/),
all accuracy values are derived from realistic statistical formulas that
reproduce the observed 70% quantum vs 90% classical gap.

Run once:  python precompute.py
Output:    results_grid.json
"""
    
import json
import math
import random
import itertools

random.seed(42)

# ── Grid parameters ──────────────────────────────────────────────────────────
SAMPLE_SIZES   = [0.10, 0.25, 0.50, 0.75, 1.00]
NOISE_LEVELS   = [0, 1, 2, 3, 4, 5]
CIRCUIT_DEPTHS = [1, 2, 4, 6, 8]

# ── Realistic accuracy model ──────────────────────────────────────────────────
#
# Classical: starts ~78% at 10% data, saturates to ~90% at 100% data.
#            Unaffected by noise_level or circuit_depth.
#
# Quantum:   starts ~58% at 10% data / depth=1, peaks ~70% at 100% data / depth=4,
#            degrades with noise (p = noise_level * 0.02 depolarizing probability).
#            Also degrades at very shallow (depth=1) or very deep (depth=8) circuits.

def classical_accuracy(sample_size: float) -> float:
    """Sigmoid-saturating curve: 78% → 90% as sample_size 0.1 → 1.0"""
    base = 0.78 + 0.12 * (1 - math.exp(-4.0 * (sample_size - 0.10)))
    noise = random.gauss(0, 0.005)
    return round(min(0.96, max(0.60, base + noise)), 4)

def quantum_accuracy(sample_size: float, noise_level: int, circuit_depth: int) -> float:
    """
    Base ~70% at ideal (noise=0, depth=4, sample=1.0).
    Degrades with noise, sample scarcity, and suboptimal depth.
    """
    # Sample size effect: steeper curve than classical (quantum needs more data)
    sample_gain = 0.58 + 0.14 * (1 - math.exp(-5.0 * (sample_size - 0.10)))

    # Circuit depth effect: peak at depth=4, worse at 1 (underfit) and 8 (overfit+noise)
    depth_factor = 1.0 - 0.08 * abs(math.log2(max(circuit_depth, 1)) - 2.0)
    depth_factor = max(0.7, depth_factor)

    # Depolarizing noise degradation: p = noise_level * 0.02
    noise_prob = noise_level * 0.02
    # Noise compounds with circuit depth: deeper = more gate errors
    noise_penalty = noise_prob * (1 + 0.15 * circuit_depth)
    noise_factor = max(0.0, 1.0 - 2.5 * noise_penalty)

    acc = sample_gain * depth_factor * noise_factor
    jitter = random.gauss(0, 0.008)
    return round(min(0.88, max(0.40, acc + jitter)), 4)

# ── Confusion matrix ──────────────────────────────────────────────────────────
def make_confusion_matrix(accuracy: float, n_samples: int = 200) -> list:
    """2×2 confusion matrix for binary classification."""
    total = n_samples
    half = total // 2
    # True positives
    tp = int(half * accuracy * random.uniform(0.95, 1.05))
    tp = min(tp, half)
    fp = half - tp
    # True negatives
    tn = int(half * accuracy * random.uniform(0.95, 1.05))
    tn = min(tn, half)
    fn = half - tn
    return [[tp, fn], [fp, tn]]

# ── Decision boundary points ──────────────────────────────────────────────────
def make_decision_boundary(accuracy: float, model: str, noise_level: int = 0) -> list:
    """
    Returns 300 points: {x, y, true_class, predicted_class}.
    Classical: crisp linear-ish boundary.
    Quantum: noisier, more irregular boundary — visually distinct.
    """
    points = []
    n = 300

    for i in range(n):
        # Two overlapping Gaussian blobs
        if i < n // 2:
            true_class = 0
            cx, cy = -0.5, -0.5
        else:
            true_class = 1
            cx, cy = 0.5, 0.5

        x = random.gauss(cx, 0.6)
        y = random.gauss(cy, 0.6)

        # Classical: clean linear decision (x + y > 0 → class 1)
        if model == 'classical':
            boundary_val = x + y + random.gauss(0, 0.1)
            predicted = 1 if boundary_val > 0 else 0
            # Add misclassifications according to accuracy
            if random.random() > accuracy:
                predicted = 1 - predicted

        else:  # quantum
            # Noisier, curved-ish boundary
            noise_perturbation = noise_level * 0.08
            boundary_val = (x + y
                            + 0.3 * math.sin(x * 2)
                            + random.gauss(0, 0.25 + noise_perturbation))
            predicted = 1 if boundary_val > 0 else 0
            if random.random() > accuracy:
                predicted = 1 - predicted

        points.append({
            "x": round(x, 3),
            "y": round(y, 3),
            "true_class": true_class,
            "predicted": predicted
        })

    return points

# ── Misclassified sample analysis ─────────────────────────────────────────────
def compute_misclassified(classical_points: list, quantum_points: list) -> dict:
    classical_errors = {i for i, p in enumerate(classical_points)
                        if p["true_class"] != p["predicted"]}
    quantum_errors   = {i for i, p in enumerate(quantum_points)
                        if p["true_class"] != p["predicted"]}

    both             = classical_errors & quantum_errors
    classical_only   = classical_errors - quantum_errors
    quantum_only     = quantum_errors   - classical_errors

    def pts(indices, source):
        return [{"x": source[i]["x"], "y": source[i]["y"],
                 "true_class": source[i]["true_class"]} for i in list(indices)[:40]]

    return {
        "classical_only": pts(classical_only, classical_points),
        "quantum_only":   pts(quantum_only,   quantum_points),
        "both":           pts(both,           classical_points),
    }

# ── Main grid generation ──────────────────────────────────────────────────────
def generate_grid():
    grid = {}
    total = len(SAMPLE_SIZES) * len(NOISE_LEVELS) * len(CIRCUIT_DEPTHS)
    done = 0

    for ss, nl, cd in itertools.product(SAMPLE_SIZES, NOISE_LEVELS, CIRCUIT_DEPTHS):
        key = f"{ss}_{nl}_{cd}"

        c_acc = classical_accuracy(ss)
        q_acc = quantum_accuracy(ss, nl, cd)

        n_samples = max(40, int(ss * 200))
        c_cm = make_confusion_matrix(c_acc, n_samples)
        q_cm = make_confusion_matrix(q_acc, n_samples)

        c_pts = make_decision_boundary(c_acc, 'classical', nl)
        q_pts = make_decision_boundary(q_acc, 'quantum',   nl)

        misclassified = compute_misclassified(c_pts, q_pts)

        grid[key] = {
            "sample_size":    ss,
            "noise_level":    nl,
            "circuit_depth":  cd,
            "classical": {
                "accuracy":               c_acc,
                "confusion_matrix":       c_cm,
                "decision_boundary":      c_pts,
            },
            "quantum": {
                "accuracy":               q_acc,
                "confusion_matrix":       q_cm,
                "decision_boundary":      q_pts,
            },
            "misclassified": misclassified,
        }

        done += 1
        if done % 25 == 0:
            print(f"  {done}/{total} configurations computed...")

    return grid

if __name__ == "__main__":
    print("Generating 150-point QML result grid...")
    grid = generate_grid()
    out_path = "results_grid.json"
    with open(out_path, "w") as f:
        json.dump(grid, f, separators=(',', ':'))
    size_kb = len(json.dumps(grid)) // 1024
    print(f"Done — {len(grid)} configurations written to {out_path} (~{size_kb} KB)")
