"""Accuracy metrics and HTML rendering for the model accuracy report.

Pure numpy: no torch, no webview, no project state, so every function here is
directly testable. api.py owns the inference loop and hands the results in.

Scoring follows the label space used everywhere (0 target, 1 background,
255 unlabeled): the target class is the positive one, and only pixels the user
actually drew *and* that carry valid image data are scored. Every metric is
None when its denominator is zero — the same contract as Api._target_iou —
because rendering a missing score as 0.000 is indistinguishable from a model
that got everything wrong.
"""

from data import UNLABELED

TARGET = 0  # positive class

# Per-image outcomes. "background only" still counts toward a set's pooled
# score: its false positives are real evidence of over-prediction, and dropping
# those images would quietly inflate the result.
STATUS_SCORED = "scored"
STATUS_BACKGROUND_ONLY = "background only"
STATUS_UNLABELED = "unlabeled"


def confusion(labels, class_map, valid_mask):
    """TP/FP/FN/TN counts of the target class over the drawn, valid pixels.

    labels / class_map: (H, W) uint8 in label space. valid_mask: (H, W) bool.
    """
    scored = (labels != UNLABELED) & valid_mask
    gt = scored & (labels == TARGET)
    pred = scored & (class_map == TARGET)
    return {
        "tp": int((gt & pred).sum()),
        "fp": int((pred & ~gt).sum()),  # pred is already inside scored
        "fn": int((gt & ~pred).sum()),
        "tn": int((scored & ~gt & ~pred).sum()),
    }


def _ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def derive(counts):
    """IoU / precision / recall / F1 / accuracy from a confusion dict."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    return {
        "iou": _ratio(tp, tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
    }


def status_of(counts):
    if counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"] == 0:
        return STATUS_UNLABELED
    if counts["tp"] + counts["fn"] == 0:
        return STATUS_BACKGROUND_ONLY  # drawn, but no target ground truth
    return STATUS_SCORED


def score_image(name, role, labels, class_map, valid_mask):
    """One per-image row: counts, metrics, status and the pixel tallies."""
    counts = confusion(labels, class_map, valid_mask)
    row = {"name": name, "role": role, "status": status_of(counts), **counts}
    row.update(derive(counts))
    row["labeled_px"] = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
    row["gt_target_px"] = counts["tp"] + counts["fn"]
    row["pred_target_px"] = counts["tp"] + counts["fp"]
    return row


def aggregate(rows):
    """Set-level summary over per-image rows.

    'micro' pools the counts and then derives, which is the headline number and
    matches how train.py accumulates its epoch IoU. 'macro' averages the
    per-image scores instead and carries its own n, since images with no score
    drop out of it. The two answer different questions and are never blended.
    """
    scored = [r for r in rows if r["status"] != STATUS_UNLABELED]
    pooled = {k: sum(r[k] for r in scored) for k in ("tp", "fp", "fn", "tn")}
    macro = {}
    for key in ("iou", "precision", "recall", "f1", "accuracy"):
        values = [r[key] for r in scored if r[key] is not None]
        macro[key] = round(sum(values) / len(values), 4) if values else None
    return {
        "images": len(rows),
        "scored_images": len(scored),
        "macro_images": sum(1 for r in scored if r["iou"] is not None),
        "counts": pooled,
        "micro": derive(pooled),
        "macro": macro,
        # False when the set pools to no target pixels at all: the metrics are
        # not merely low, they are unanswerable.
        "scoreable": (pooled["tp"] + pooled["fp"] + pooled["fn"]) > 0,
    }


def build_report(rows, total_epochs=0, project_name="", generated_at=""):
    """Full report payload: both sets plus the per-image rows."""
    validation = [r for r in rows if r["role"] == "validation"]
    training = [r for r in rows if r["role"] == "training"]
    return {
        "project_name": project_name,
        "generated_at": generated_at,
        "total_epochs": total_epochs,
        "validation": aggregate(validation),
        "training": aggregate(training),
        "rows": rows,
    }


# ---------- HTML ----------

def _fmt(value):
    return "—" if value is None else f"{value:.4f}"


def _escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


_CSS = """
body { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
       margin: 0; padding: 32px; background: #14121a; color: #e9e5f1; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: #a49cb8; margin-bottom: 24px; font-size: 13px; }
.sets { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
.set { flex: 1 1 260px; background: #211f2d; border: 1px solid #3a3750;
       border-radius: 10px; padding: 16px 18px; }
.set h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em;
          color: #a49cb8; margin: 0 0 12px; }
.headline { font-size: 30px; font-family: ui-monospace, Menlo, monospace; }
.headline .label { font-size: 12px; color: #a49cb8; font-family: inherit; }
.metrics { margin-top: 12px; font-size: 13px; color: #a49cb8; line-height: 1.7; }
.metrics b { color: #e9e5f1; font-family: ui-monospace, Menlo, monospace;
             font-weight: 500; }
.note { background: #2e2d3e; border-left: 3px solid #f26fb5; padding: 10px 14px;
        border-radius: 5px; margin-bottom: 20px; font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid #2a2838; }
th:first-child, td:first-child { text-align: left; font-family: ui-monospace,
                                 Menlo, monospace; }
th { color: #a49cb8; font-weight: 500; text-align: right; }
tr.validation td:nth-child(2) { color: #5fd08a; }
td.dim { color: #6f6886; }
"""


def _set_html(title, summary):
    if not summary["images"]:
        body = '<div class="headline"><span class="label">no images</span></div>'
    elif not summary["scoreable"]:
        body = ('<div class="headline"><span class="label">not scoreable — '
                'no target labels in this set</span></div>')
    else:
        micro, macro = summary["micro"], summary["macro"]
        body = (
            f'<div class="headline">{_fmt(micro["iou"])}'
            f'<span class="label"> target IoU (micro)</span></div>'
            f'<div class="metrics">'
            f'F1 <b>{_fmt(micro["f1"])}</b> · '
            f'precision <b>{_fmt(micro["precision"])}</b> · '
            f'recall <b>{_fmt(micro["recall"])}</b><br>'
            f'accuracy <b>{_fmt(micro["accuracy"])}</b> · '
            f'macro IoU <b>{_fmt(macro["iou"])}</b> '
            f'(mean over {summary["macro_images"]} images)<br>'
            f'{summary["scored_images"]} of {summary["images"]} images scored'
            f'</div>')
    return f'<div class="set"><h2>{_escape(title)}</h2>{body}</div>'


def render_html(report):
    """The report as a self-contained HTML page (no external assets)."""
    rows_html = []
    for row in report["rows"]:
        dim = ' class="dim"' if row["status"] != STATUS_SCORED else ""
        rows_html.append(
            f'<tr class="{row["role"]}"><td>{_escape(row["name"])}</td>'
            f'<td>{row["role"]}</td>'
            f'<td{dim}>{row["status"]}</td>'
            f'<td>{row["labeled_px"]:,}</td>'
            f'<td>{row["tp"]:,}</td><td>{row["fp"]:,}</td><td>{row["fn"]:,}</td>'
            f'<td>{_fmt(row["iou"])}</td><td>{_fmt(row["precision"])}</td>'
            f'<td>{_fmt(row["recall"])}</td><td>{_fmt(row["f1"])}</td></tr>')

    gap = ""
    v, t = report["validation"]["micro"]["iou"], report["training"]["micro"]["iou"]
    if v is not None and t is not None:
        gap = (f'<div class="note">Training IoU exceeds validation IoU by '
               f'<b>{t - v:.4f}</b>. A large gap means the model fits the '
               f'images it trained on better than unseen ones.</div>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Accuracy report — {_escape(report["project_name"])}</title>
<style>{_CSS}</style></head><body>
<h1>Accuracy report — {_escape(report["project_name"])}</h1>
<div class="sub">{_escape(report["generated_at"])} · model trained for
{report["total_epochs"]} epochs · scored over user-labeled, valid pixels only
(target = positive class)</div>
<div class="sets">{_set_html("Validation (held out)", report["validation"])}
{_set_html("Training", report["training"])}</div>
{gap}
<table><thead><tr><th>image</th><th>set</th><th>status</th><th>labeled px</th>
<th>TP</th><th>FP</th><th>FN</th><th>IoU</th><th>precision</th><th>recall</th>
<th>F1</th></tr></thead><tbody>{"".join(rows_html)}</tbody></table>
</body></html>
"""
