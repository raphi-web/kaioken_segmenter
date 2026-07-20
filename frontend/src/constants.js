export const UNLABELED = 255

export const CLASSES = [
  { id: 0, name: 'Target', color: [255, 80, 40] },
  { id: 1, name: 'Background', color: [60, 140, 255] },
]

export const CLASS_COLORS = Object.fromEntries(CLASSES.map((c) => [c.id, c.color]))

export const LABEL_ALPHA = 180
