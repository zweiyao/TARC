"""Synthetic speckle frames, so the real preprocessing can run with no sensor.

PLACEHOLDER. The sensor is not wired up yet, and the recorded speckle frames
live on a host that is currently unreachable, so the frames here are generated.
Everything downstream of them is the real thing: these go through flow.py's
Farneback exactly as camera frames would.

Generating rather than replaying buys one thing a recording cannot — the
displacement is applied here, so its magnitude is known and the flow can be
checked against it.
"""
import cv2
import numpy as np

from tactile.flow import TactileFlow, _colorize

# What flow.py's viewer/record actually use; the TactileFlow default is
# (960, 540), which is 2.4x slower for no accuracy gain (README perf table).
FLOW_SIZE = (640, 360)
# Fixed colour scale, matching viewer.py. Per-frame normalisation would make
# heatmaps from different moments incomparable.
VMAX = 3.0


def speckle(shape=(1080, 1920), seed=0, grain=6):
    """Random RGB speckle, the pattern the gel surface carries.

    Farneback needs every local window to be uniquely matchable. Per-pixel white
    noise is too fine to survive the downsample to 640x360, so the grain is
    built coarse and upsampled — which is also what the real speckle looks like.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, (h // grain + 1, w // grain + 1, 3), dtype=np.uint8)
    return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)


def warp(img, peak_px=2.5, sigma_frac=0.15, center=(0.5, 0.5)):
    """Push the image through a Gaussian displacement bump, as a press would.

    Returns (warped, peak_px). The peak is in *flow.py's* 640x360 working
    resolution, not the input resolution, because that is where the flow is
    measured — the caller's assertion has to compare against the same units.
    """
    h, w = img.shape[:2]
    scale = w / FLOW_SIZE[0]  # input px per working px
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = center[1] * h, center[0] * w
    sigma = sigma_frac * min(h, w)
    bump = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))

    # Displace along +x only: mag = sqrt(u^2+v^2) then reduces to |u|, so the
    # peak of mag is exactly the peak of the bump. A diagonal push would make
    # the expected value depend on the direction too.
    d = peak_px * scale * bump
    return (
        cv2.remap(img, xs - d, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT),
        peak_px,
    )


def magnitude(frame, ref_frames):
    """Raw frame -> displacement magnitude field, in working-resolution px."""
    # ema=1.0 drops flow.py's cross-frame smoothing, making this a pure function
    # of (frame, ref_frames). With the default 0.5 the same frame fed twice
    # gives two different answers.
    tf = TactileFlow(size=FLOW_SIZE, ema=1.0)
    tf.set_reference(ref_frames)
    return tf.process(frame)["mag"]


def heatmap(frame, ref_frames, size=(320, 240), vmax=VMAX):
    """Raw frame -> the uint8 BGR heatmap the `tactile` observation key wants.

    `size` is OpenCV's (width, height), so the default returns (240, 320, 3).
    """
    heat = _colorize(magnitude(frame, ref_frames), vmax, cv2.COLORMAP_TURBO)
    return cv2.resize(heat, size, interpolation=cv2.INTER_AREA)


def readout_noise(img, seed, sigma=2.0):
    """Per-frame sensor noise on top of a fixed pattern.

    set_reference averages N frames of the *same* scene to suppress exactly
    this. Handing it N independent patterns instead makes the average a blur
    that matches nothing, and the rest-frame noise floor comes out ~6x the real
    sensor's (measured: 0.41 px vs the 0.065 px in the source README).
    """
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def press(seed=0, peak_px=2.5, n_ref=3):
    """The canned press the tests use: reference frames, pressed frame, peak."""
    base = speckle(seed=seed)
    ref = [readout_noise(base, seed=seed + 100 + i) for i in range(n_ref)]
    cur, peak = warp(readout_noise(base, seed=seed + 200), peak_px=peak_px)
    return ref, cur, peak


if __name__ == "__main__":
    import sys

    ref, cur, peak = press()
    mag = magnitude(cur, ref)
    rest = magnitude(ref[0], ref)
    print(f"applied peak      {peak:.3f} px")
    print(f"recovered peak    {mag.max():.3f} px   (err {abs(mag.max()-peak):.3f})")
    print(f"rest frame  mean  {rest.mean():.4f}  p99 {np.percentile(rest, 99):.4f}")
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tactile_heatmap.png"
    cv2.imwrite(out, heatmap(cur, ref))
    print(f"heatmap {heatmap(cur, ref).shape} -> {out}")
