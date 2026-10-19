# Cricket video annotation guidelines (initial draft)

This guide defines a minimal consistent annotation contract. It is not a substitute for source-rights review.

## Person boxes

- Use original decoded frame indices and source timestamps, not the sampled-frame counter.
- Label each clearly visible person with a tight axis-aligned box in pixel coordinates: `[x1, y1, x2, y2]`, where the top-left is inclusive and bottom-right is exclusive by convention.
- Keep truncated people when a meaningful visible extent exists; annotate the visible portion and mark truncation/occlusion in a richer annotation source if supported.
- Do not infer hidden limbs or body extents behind another player.
- Exclude people that are only speculative blobs. Maintain an ignore-region policy for ambiguous crowd/stadium backgrounds.
- Label umpires, players, coaches and spectators consistently in separate metadata if the downstream task needs those distinctions; do not silently mix them in a player-only benchmark.

## Ball annotations

- Annotate a ball only when the ball is visually identifiable at the source resolution.
- Store a centre point and/or tight box plus visibility state: visible, blurred, partially_occluded, fully_occluded, out_of_frame, or uncertain.
- Never interpolate a ground-truth ball position through frames where the ball is not visible.
- Record the source frame and reviewer confidence; use independent review for difficult labels.

## Tracking identities

- Assign local track IDs within a continuous shot.
- Do not assume the same visual person across a broadcast cut without separate evidence.
- Preserve an explicit unknown identity when occlusion or resolution prevents confident association.
- Record track birth, last visible frame, and reason for termination where possible.

## Shot boundaries and views

- Record the first frame of the new shot and the boundary type (hard cut, dissolve, replay transition, uncertain).
- Distinguish camera cuts from rapid camera motion or flashes.
- Suggested view labels: wide_field, pitch_view, bowler_runup, batter_closeup, replay, crowd_other, unknown.

## Dataset split and review

- Split by match or broadcast source, never randomly by neighbouring frame.
- Keep real footage and synthetic game footage separate in reports.
- Have a second reviewer adjudicate ambiguous ball positions, player identities and event timestamps.
- Store source rights, clip timestamps, annotation version, reviewer, and content hash in a manifest.
- Do not commit restricted footage or frames to Git; publish only metadata and download instructions where appropriate.
