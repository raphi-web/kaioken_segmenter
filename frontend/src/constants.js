export const UNLABELED = 255

// Fallback palette for the no-project (standalone image) case; an open
// project's own classes (from get_project) take precedence everywhere else.
export const CLASSES = [
  { id: 0, name: 'Target', color: '#ff5028' },
  { id: 1, name: 'Background', color: '#3c8cff' },
]

export function hexToRgb(hex) {
  const h = hex.replace('#', '')
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16))
}

export const LABEL_ALPHA = 180

// ---------- magic wand ----------

// Slider position -> backend tolerance for source="image": slider / 200.
export const WAND_TOL_SCALE = 200

// source="sam" spans [0.4, 1.2] instead of dividing by 100, because the useful
// cosine distances do not start at zero. At `fine` on a 512px tile, three
// clicks each selected under 0.15% of the image below 0.7, 0.6-2.3% at 0.8, and
// 7-30% by 1.0: a region at 0.7-0.95, a speck below it, the whole scene above.
// A plain /100 left that entire band inside the right-hand fifth of the travel.
export const WAND_SAM_TOL_MIN = 0.4
export const WAND_SAM_TOL_SPAN = 0.8

// Per-class starting tolerance, in slider units. What counts as "the same
// material" is a property of the material: the measured per-class optima span
// 0.07 to 0.19, and one global number is a compromise weighted toward whichever
// class the project has most of. Measured over 388 regions across 20 images by
// clicking the most interior pixel of each connected labeled region: seeded per
// class scores 0.397 mean IoU against 0.353 for a flat 0.13 (+12.4%), the share
// of regions reaching IoU>=0.5 rises 30.9% -> 36.3%, and median over-selection
// halves, 10.6% -> 4.7%.
//
// Keyed by class NAME, not id -- these are properties of the material and ids
// are per project. Matching is case-insensitive and falls back to a prefix, so
// both a short name and this dataset's fuller ones ("Raw-Earth/Agriculture",
// "Agriculture-Green") resolve to the same seed.
export const WAND_TOLERANCE_SEEDS = {
  urban: 38,        // 0.19
  'raw-earth': 28,  // 0.14
  road: 24,         // 0.12
  forest: 22,       // 0.11
  wetland: 22,      // 0.11
  shrub: 20,        // 0.10
  meadow: 18,       // 0.09
  agri: 18,         // 0.09
  water: 14,        // 0.07
}

// The fallback cannot be improved on as a single number: over those same 388
// regions the best fixed setting is 0.14 at 0.349 mean IoU against 0.347 for
// 0.13 -- noise -- and the curve is flat from 0.11 to 0.15. What pins it low is
// the cliff to the right: median over-selection runs 6% at 0.13, 25% at 0.15,
// 118% at 0.17 and 244% at 0.19 while IoU falls away only slowly. The wand is
// additive, so selecting too little costs another click while selecting too
// much costs an undo and hand correction.
export const WAND_TOLERANCE_DEFAULT = 26 // 0.13

export function wandToleranceSeed(className) {
  const name = String(className || '').toLowerCase()
  if (name in WAND_TOLERANCE_SEEDS) return WAND_TOLERANCE_SEEDS[name]
  for (const [key, value] of Object.entries(WAND_TOLERANCE_SEEDS)) {
    if (name.startsWith(key)) return value
  }
  return WAND_TOLERANCE_DEFAULT
}

// Budget slider (1..100) -> pixels, log-scaled over 100..100,000 and rounded to
// two significant figures; the top of the travel means "no cap". Log because
// useful budgets span three orders of magnitude -- a road edge is a few hundred
// pixels, a field tens of thousands -- and a linear slider would spend most of
// its travel where you never want to be.
export const WAND_BUDGET_MIN = 100
export const WAND_BUDGET_MAX = 100000

export function wandBudget(slider) {
  if (slider >= 100) return null
  const t = (slider - 1) / 98
  const raw = WAND_BUDGET_MIN * (WAND_BUDGET_MAX / WAND_BUDGET_MIN) ** t
  const digits = Math.max(0, Math.floor(Math.log10(raw)) - 1)
  const step = 10 ** digits
  return Math.max(WAND_BUDGET_MIN, Math.round(raw / step) * step)
}

// Preview: the class colour at 40%, with a two-tone boundary on top. One tone
// always loses -- a light outline vanishes on Snow (#eef2f7), a dark one
// vanishes in shadow -- so selected edge pixels get the light tone and the
// unselected pixels touching them get the dark one. The pair reads on both.
export const WAND_PREVIEW_ALPHA = 102
export const WAND_EDGE_ALPHA = 102
export const WAND_EDGE_LIGHT = 255
export const WAND_EDGE_DARK = 20

// Share of the image past which the readout calls the selection a runaway.
export const WAND_WARN_FRACTION = 0.4
