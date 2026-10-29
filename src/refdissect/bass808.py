"""The 808: where each note starts, its pitch to the cent, and its slides.

An 808 line is one sample (or one synth patch) retriggered at different
transpositions, cut by the next note. So the notes are read as
transpositions of the song's own 808, learnt from its long notes, the
operation Simpler performs:

1. reattacks(): where the 808 band's envelope on the bass stem starts again
   (a notch: the old note cut; or a rise), with its evidence; an attack of
   the drum stem's low band at the same moment marks a note that starts
   with a kick, and where the re-attacks are too few to learn from, the
   drum attacks are onsets too;
2. learn(): the template, the long notes of the most common waveform lined
   up and averaged (what the sample sounds like at that pitch), its settled
   pitch by QIFFT on the sustain once the drop has settled, and its own
   pitch curve (the drop);
3. candidates(): where a note may start and at which transposition (a
   matched filter of the template's head at every semitone, refined by the
   energy each explains over 300 ms);
4. segment(): the notes, by dynamic programming over the candidates: the
   template copies that explain the bass stem's 808 band best, one penalty
   per note. A kick inside a note, or the ripple of a distorted low tone,
   would have to explain the note's sustain with the template's attack and
   drop, and mostly loses to the note going on;
5. retriggers(): a note split where the 808 starts again inside it (two
   16th notes at one pitch at 160 BPM explain less than their penalty);
6. refine(): each note's transposition to the cent over up to 300 ms;
7. kick_splits(): a very short note on a kick at a transposition past the
   search's whole semitones is the kick's attack split off the next note;
8. legato(): a note found where nothing attacks is the note before going
   on (a short attacked note whose follower comes back to the pitch before
   it is a kick inside that held note); legato notes that slide there from
   the note before are made only on request, with evidence (_slide).

The template is learnt twice (from the re-attacks, then from the notes
found) from three sets of onsets: those with a drum attack, those with a
drum attack or a deep notch, and all; the one whose notes explain the most
of the band is kept, and within 1.5 % of it the one whose notes start on
drum attacks. One 808 of the suite dips 13 dB 75 ms into every note: a
template learnt from such dips misses the attack, and every note comes out
40-75 ms late; where the notes found start that late after the kicks just
before them, the template is learnt again that much earlier.

transcribe() returns what bass808.json holds; project() turns it into
analysis.json's notes_808 (t, t_attack, end, pitch, midi_float,
midi_settled, cents, slide, peak, glide, flags), so structure, groove and
make_plan read it as before. It costs 9-50 s on a suite beat (about 30 s
of stems, one core: analyze takes 36 s a beat, 6.6 s with the pYIN reader
it replaced) and 159 s on PIRAMIT's 126 s. That reader (librosa onsets
and a pYIN median over 60-450 ms) found 1,413 onsets for 336 notes on the
calibration stems, dropped every note under 140 ms and read pitches 14-27
cents sharp under 50 Hz (+33 cents on PIRAMIT, whose 808 sits at +11.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import soxr
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import fftconvolve, find_peaks, hilbert

from .audio import PRE, SR, attack, band, samples
from .hits import group

# the re-attack detector's thresholds (dB), chosen on six dev phrases of
# synthetic 808 lines and frozen: a notch at least NOTCH_DB under the lowest
# point 8-70 ms before it, recovering RECOVER_DB within 30 ms, or a rise of
# RISE_DB over the next 15 ms against the 40 ms before
NOTCH_DB, RECOVER_DB, RISE_DB, FLOOR_DB = 4.0, 6.0, 9.0, -50.0
DEEP_NOTCH_DB = 8.0  # a re-attack this deep is a note's start more often than not
DRUM_RISE_DB = 6.0  # the drum stem's low band rising this much marks a kick (an onset) there
VARIANTS = ("drums", "drums|notch", "all")  # the onsets a template is learnt from, tried in turn
TIE = 0.015  # templates whose notes explain this close a share of the band are as good a fit
SETTLE_CENTS = 3.0  # learn(): a sustain window's two halves agree this well (QIFFT) where the drop has settled
ATTACKED_NCC = 0.9  # a note whose template fit (attack and drop included) reaches this has an attack
# legato(): whether a note found where nothing attacks may be a legato note at all (_slide's evidence;
# off: merging scored better on the suite), semitones it must have moved from the note before (late in a
# held note the sustain of the suite's htdemucs stems reads 0.5-0.7 off), the shortest such note (seconds;
# 80 ms is 2.6 periods at 33 Hz), how far from the template its pitch may be (semitones: as far as
# candidates() looks; PIRAMIT's stems gave one at MIDI -7 without it), and the longest attacked note a
# kick inside a held note may be split off as (seconds)
SLIDES, LEGATO_STEP, LEGATO_SHORTEST, LEGATO_REACH, SPLIT_S = False, 0.75, 0.08, 12.75, 0.12
SLIDE_AFTER = 0.1  # _slide: seconds after the note before starts that a slide may leave it (sooner: its own drop)
# the 808 is silent where its band's RMS over 25 ms stays SILENT_DB or more under a note's peak (that RMS's highest
# point over the note's first 200 ms) for SILENT_S seconds or more: inside the suite's 5,448 truth notes (both
# models, 50 ms in to 10 ms before the end) it never fell more than 26.8 dB under the note's peak, and PIRAMIT's
# rests sit 52-81 dB under its notes' median peak, the shortest 62 ms long
SILENT_DB, SILENT_S = 40.0, 0.05


# ── Attacks ──

def envelope(x: np.ndarray, smooth_ms: float = 1.0) -> np.ndarray:
    """The analytic envelope of x, smoothed over smooth_ms."""
    e = np.abs(hilbert(np.asarray(x, dtype=np.float64)))
    w = max(1, samples(smooth_ms))
    return np.convolve(e, np.ones(w) / w, mode="same")


def rms_envelope(x: np.ndarray, ms: float = 5.0) -> np.ndarray:
    """The RMS of x over ms around each sample."""
    w = max(1, samples(ms))
    return np.sqrt(np.convolve(np.asarray(x, dtype=np.float64) ** 2, np.ones(w) / w, mode="same"))


def _ahead(a: np.ndarray, k: int, fill: float) -> np.ndarray:
    """out[t] = a[t + k], fill where that falls outside."""
    out = np.full(len(a), fill)
    if k >= 0:
        out[: len(a) - k] = a[k:]
    else:
        out[-k:] = a[: len(a) + k]
    return out


def _window(e: np.ndarray, a: int, b: int, op) -> np.ndarray:
    """op (maximum_filter1d or minimum_filter1d) of e over [t + a, t + b) for every t."""
    size = b - a
    fill = 0.0 if op is maximum_filter1d else np.inf
    return _ahead(op(e, size=size, mode="nearest"), a + size // 2, fill)


def _db(a: float, b: float) -> float:
    return round(float(20 * np.log10(max(a, 1e-12) / max(b, 1e-12))), 1)


def rise_db(env: np.ndarray, i: int, after_ms: float = 15.0, before_ms: float = 40.0, gap_ms: float = 5.0) -> float:
    """How far (dB) env rises at sample i: its maximum over the next
    after_ms against its maximum over before_ms ending gap_ms before i."""
    after = env[i:i + samples(after_ms)]
    before = env[max(0, i - samples(before_ms)):max(0, i - samples(gap_ms))]
    return _db(after.max() if len(after) else 0.0, before.max() if len(before) else 0.0)


def drum_attacks(env: np.ndarray, rise: float = DRUM_RISE_DB, merge_ms: float = 40.0) -> np.ndarray:
    """Samples where a drum stem's low-band RMS envelope (rms_envelope)
    rises by `rise` dB (its next 15 ms against the 40 ms ending 5 ms before)
    to within 30 dB of its peak: the kicks, each at the foot of its rise."""
    if not len(env):
        return np.zeros(0, dtype=int)
    after = _window(env, 0, samples(15), maximum_filter1d)
    before = _window(env, -samples(40), -samples(5), maximum_filter1d)
    hit = np.flatnonzero((after >= np.maximum(before, 1e-12) * 10 ** (rise / 20)) & (after > env.max() * 10 ** (-1.5)))
    out: list[int] = []
    for c in hit:
        if not out or c - out[-1] >= samples(merge_ms):
            out.append(int(c))
    return np.array([_foot(env, c) for c in out], dtype=int)


def _foot(env: np.ndarray, c: int) -> int:
    """Where a rise found at c starts: the minimum of env within 15 ms after
    c (a rise is found up to 15 ms early), then the last sample before env
    passes twice that minimum and a tenth of the peak of the next 30 ms.
    After silence the minimum is anywhere in the noise, and the zero-phase
    band filter and the analytic envelope spread a note's rise 10 ms ahead
    of it (on a synthetic note after a rest, the minimum sat 10 ms early,
    a tenth of the peak 2 ms early)."""
    i = c + int(np.argmin(env[c:c + samples(15)]))
    ahead = env[i:i + samples(30)]
    if not len(ahead):
        return i
    up = np.flatnonzero(ahead > max(2 * env[i], 0.1 * ahead.max()))
    return i + (max(0, int(up[0]) - 1) if len(up) else 0)


def reattacks(x: np.ndarray, notch_db: float = NOTCH_DB, recover_db: float = RECOVER_DB,
              rise: float = RISE_DB, floor_db: float = FLOOR_DB, merge_ms: float = 40.0) -> list[dict]:
    """Where an 808's envelope starts again: {"i" (sample of the envelope's
    foot), "notch_db" (how far under the lowest point 8-70 ms before it),
    "rise_db"} per candidate, merged within merge_ms (the earliest kept). A
    mono 808 retriggered shows a notch (the old note cut, the new one
    starting from zero); a note after a rest or a decay shows a rise. Levels
    more than -floor_db under the file's peak are ignored. On the suite's
    bass stems (htdemucs) it finds 77 % of the 808 notes within 30 ms at a
    precision of 0.80: distortion makes the envelope dip within each period,
    as deep as a cut."""
    e = envelope(x)
    if not len(e):
        return []
    floor = e.max() * 10 ** (floor_db / 20)
    lo_prev = _window(e, -samples(70), -samples(8), minimum_filter1d)
    hi_next30 = _window(e, 0, samples(30), maximum_filter1d)
    hi_next15 = _window(e, 0, samples(15), maximum_filter1d)
    hi_prev40 = _window(e, -samples(40), 0, maximum_filter1d)
    local_min = e <= minimum_filter1d(e, size=samples(10))
    notch = local_min & (e <= lo_prev * 10 ** (-notch_db / 20)) & (hi_next30 >= e * 10 ** (recover_db / 20)) \
        & (hi_next30 > floor)
    rising = (hi_next15 >= np.maximum(hi_prev40, 1e-12) * 10 ** (rise / 20)) & (hi_next15 > floor)
    starts: list[int] = []
    for c in np.flatnonzero(notch | rising):
        if not starts or c - starts[-1] >= samples(merge_ms):
            starts.append(int(c))
    out = []
    for c in starts:
        i = _foot(e, c)
        before = e[max(0, i - samples(70)):max(0, i - samples(8))]
        out.append({"i": i, "notch_db": _db(before.min() if len(before) else e[i], e[i]),
                    "rise_db": rise_db(e, i, 30.0, 40.0, 0.0)})
    return out


# ── Working rate ──

DS = 8  # the 808 band (under 1.5 kHz) is read at SR / DS, 5512.5 Hz
LOW_SR = SR / DS
PRE_LOW = PRE / DS  # a template starts PRE before its onset: 27.5 samples at LOW_SR


def lowband(x: np.ndarray) -> np.ndarray:
    """x in the 808 band at LOW_SR."""
    return soxr.resample(band(x, "808").astype(np.float64), SR, LOW_SR)


def low(ms: float) -> int:
    """Milliseconds to samples at LOW_SR."""
    return int(round(ms * LOW_SR / 1000))


def played(t: np.ndarray, cents: float) -> np.ndarray:
    """t (at LOW_SR) as a sampler plays it `cents` higher: faster, shorter."""
    return t if not cents else soxr.resample(t, LOW_SR * 2 ** (cents / 1200), LOW_SR)


def sliding(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """NCC of t against x[k:k + len(t)] for every k (a matched filter; zero
    where x is silent)."""
    n = len(t)
    if len(x) < n:
        return np.zeros(0)
    num = fftconvolve(x, t[::-1], mode="valid")
    c = np.r_[0.0, np.cumsum(x ** 2)]
    win = c[n:] - c[:-n]
    tn = float(np.dot(t, t)) or 1e-18
    return np.where(win > 1e-12 * tn, num / np.sqrt(tn * np.maximum(win, 1e-18)), 0.0)


def _gain(x: np.ndarray, y: np.ndarray) -> float:
    """The least-squares level of y in x."""
    return float(np.dot(x, y) / (float(np.dot(y, y)) or 1e-18))


def _explained(x: np.ndarray, y: np.ndarray, g: float) -> float:
    """The most energy of x that g * y explains from its start before it is
    cut: minus the least error-minus-energy over its prefixes."""
    n = min(len(x), len(y))
    return float(-min(0.0, float(np.cumsum(g * g * y[:n] ** 2 - 2 * g * x[:n] * y[:n]).min()))) if n else 0.0


# ── The song's own 808 ──

@dataclass
class Template:
    """The 808 as the song plays it at its most common pitch: the notes of
    that pitch lined up and averaged from PRE before their onsets (sr: the
    808 band at SR; audio: at LOW_SR, continued past its end by its last
    period, decaying as it does), the onsets used, the settled pitch (Hz)
    and how it was read, and its pitch curve (scenarios.pitch_curve:
    seconds after the onset, cents over hz)."""
    sr: np.ndarray
    audio: np.ndarray
    starts: list[int]
    hz: float
    method: str
    curve: list
    cache: dict = field(default_factory=dict, repr=False)

    def at(self, cents: float) -> np.ndarray:
        """The continued template played `cents` higher (cached to 0.1 cent)."""
        key = round(float(cents), 1)
        if key not in self.cache:
            self.cache[key] = played(self.audio, key)
        return self.cache[key]

    @property
    def midi(self) -> float:
        return 69 + 12 * float(np.log2(self.hz / 440.0))


def qifft(x: np.ndarray, lo_hz: float, hi_hz: float, n_fft: int = 1 << 18) -> float | None:
    """The strongest spectral peak of x between lo_hz and hi_hz (Hann
    window, zero-padded FFT, parabola through the log magnitudes), None
    when the strongest point is at either end: a steady tone's pitch to
    within 0.06 cent at 22-62 Hz, where pYIN with a 4096 frame reads 14-27
    cents sharp."""
    if len(x) < samples(50):
        return None
    spec = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64) * np.hanning(len(x)), n_fft))
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    idx = np.flatnonzero((f >= lo_hz) & (f <= hi_hz))
    if len(idx) < 3:
        return None
    k = int(idx[np.argmax(spec[idx])])
    if k in (idx[0], idx[-1]) or not spec[k] > 0:
        return None
    a, b, c = np.log(spec[k - 1:k + 2] + 1e-18)
    den = a - 2 * b + c
    return float((k + (float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den else 0.0)) * SR / n_fft)


def _rough_hz(x: np.ndarray, lo_hz: float = 15.0, hi_hz: float = 130.0) -> float | None:
    """A tone's pitch to within a fraction of a semitone: the first
    autocorrelation peak of x within lo_hz-hi_hz that reaches 0.8 of the
    highest (a period, not a multiple of it)."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    a, b = int(SR / hi_hz), int(SR / lo_hz)
    if len(x) < 2 * b:
        return None
    n = 1 << int(np.ceil(np.log2(2 * len(x))))
    r = np.fft.irfft(np.abs(np.fft.rfft(x, n)) ** 2, n)[: b + 2]
    r = r / (r[0] or 1.0)
    peaks = [k for k in range(a, b + 1) if r[k] >= r[k - 1] and r[k] >= r[k + 1] and r[k] > 0]
    if not peaks:
        return None
    top = max(r[k] for k in peaks)
    k = next(k for k in peaks if r[k] >= 0.8 * top)
    den = r[k - 1] - 2 * r[k] + r[k + 1]
    return float(SR / (k + (0.5 * (r[k - 1] - r[k + 1]) / den if den else 0.0)))


def _settled_hz(x: np.ndarray, near_hz: float, skip_s: float = 0.03) -> float | None:
    """Where the pitch of a falling tone settles: its period-by-period pitch
    (scenarios.crossings) from skip_s on, fitted with f(t) = a + b exp(-t/tau)
    (tau from 10 ms to 1 s), a; None without six periods. A template from
    short notes stops before its drop settles, and a QIFFT of its end reads
    the drop."""
    from .scenarios import crossings
    z = crossings(x, near_hz)
    if len(z) < 7:
        return None
    t, f = (z[1:] + z[:-1]) / 2, 1 / np.diff(z)
    t, f = t[t >= skip_s], f[t >= skip_s]
    if len(t) < 5:
        return None
    def fitted(taus):
        best = None
        for tau in taus:
            a = np.stack([np.ones_like(t), np.exp(-t / tau)], axis=1)
            coef, *_ = np.linalg.lstsq(a, f, rcond=None)
            err = float(np.sum((a @ coef - f) ** 2))
            if coef[0] > 0 and coef[1] >= 0 and (best is None or err < best[0]):
                best = (err, float(coef[0]), float(tau))
        return best

    # the asymptote moves 40-70 cents per 20 % of tau on a 330 ms template: coarse, then fine
    best = fitted(np.geomspace(0.01, 1.0, 50))
    best = fitted(best[2] * np.geomspace(0.9, 1.1, 41)) if best else None
    return best[1] if best else float(np.median(f[-3:]))


def _extended(t: np.ndarray, seconds: float, near_hz: float) -> np.ndarray:
    """t (at LOW_SR) continued to `seconds` by repeating its last period,
    each repeat scaled by the decay of its last two periods."""
    n = int(seconds * LOW_SR)
    period = int(round(LOW_SR / near_hz)) if near_hz else 0
    if len(t) >= n or period < 2 or len(t) < 3 * period:
        return t
    last, before = t[-period:], t[-2 * period:-period]
    ratio = float(np.clip(np.sqrt(np.sum(last ** 2) / (np.sum(before ** 2) or 1e-18)), 0.0, 1.0))
    reps = int(np.ceil((n - len(t)) / period))
    return np.r_[t, np.concatenate([last * ratio ** (k + 1) for k in range(reps)])][:n]


def _first_reattack(x: np.ndarray, after: int, rise_cents: float = 50.0) -> int:
    """Where x (an average of notes) starts again after `after` samples: a
    next note every member ran into (a pattern repeats). The envelope
    re-attacks there, or the pitch rises (by rise_cents over its lowest so
    far, period by period: an 808's own pitch only falls, and a retrigger
    restarts its drop). len(x) when it does neither."""
    later = [o["i"] for o in reattacks(x) if o["i"] > after]
    cut = later[0] if later else len(x)
    near = _rough_hz(x[after:min(len(x), after + samples(300))])
    if near:
        from .scenarios import crossings
        z = crossings(x, near)
        if len(z) > 3:
            cents = 1200 * np.log2(1 / np.diff(z))
            mid = (z[1:] + z[:-1]) / 2 * SR
            low_so_far = np.minimum.accumulate(np.where(mid > after, cents, np.inf))
            up = np.flatnonzero((mid > after) & (cents > low_so_far + rise_cents))
            if len(up):
                cut = min(cut, int(z[up[0]] * SR))
    return cut


def _own_length(lo: np.ndarray, s: int, before: int, max_s: float) -> int:
    """How long the note at onset s rings before something else starts
    (_first_reattack over its own audio from 60 ms in), up to the onset
    `before` (the next onset with strong evidence) and max_s."""
    end = min(len(lo), s + samples(1000 * max_s), before)
    x = lo[s:end]
    return _first_reattack(x, samples(60)) if len(x) > samples(120) else len(x)


def _members(lo: np.ndarray, notes: list[tuple[int, int, bool]], long_ms: float, max_s: float, strong_only: bool,
             least_ms: float = 120.0):
    """The long notes (long_ms, or the longest quarter of least_ms or more),
    each as long as it rings on its own (_own_length: a missed onset leaves a
    note's segment running over the next ones), that share the most common
    waveform (hits.group over their first 250 ms at most), as (start of the
    window PRE before the onset, sign x gain, length), and that span."""
    use = [(s, k) for s, _, k in notes if k or not strong_only]
    stops = [s for s, _, k in notes if k]
    lengths = []
    for s, _ in use:
        nxt = next((t for t in stops if t > s + samples(60)), len(lo))
        lengths.append(_own_length(lo, s, nxt, max_s))
    long = [(s, n) for (s, _), n in zip(use, lengths) if n >= samples(long_ms)]
    if len(long) < 3:
        long = sorted(zip([s for s, _ in use], lengths), key=lambda sn: -sn[1])[:max(3, len(use) // 4)]
        long = [(s, n) for s, n in long if n >= samples(least_ms)]
    if not long:
        return [], 0
    span = min(min(n for _, n in long), samples(250))
    groups = group(lo, starts=[s for s, _ in long], cluster_ms=1000 * span / SR, threshold=0.5, max_lag_ms=4)
    if not groups:  # nothing sounds there (a silent bass stem, kicks as onsets)
        return [], 0
    g = groups[0]
    length = {s: n + PRE for s, n in long}
    return [(s - PRE - lag, sign * gain, length[s]) for s, lag, sign, gain, _ in g.members
            if s - PRE - lag >= 0 and s - PRE - lag + length[s] <= len(lo)], span


def learn(lo: np.ndarray, notes: list[tuple[int, int, bool]], long_ms: float = 450.0, max_s: float = 1.5,
          sustain_ms: tuple[float, float] = (300.0, 600.0), agree: float = 0.5,
          least_ms: tuple[float, float] = (120.0, 200.0)) -> Template | None:
    """The template from notes ((onset, end, strong) samples of lo, the 808
    band at SR): the long ones that share the most common waveform
    (_members, of least_ms[0] or more), from those marked strong unless
    fewer than three of them agree, averaged where they still sound, up to
    max_s; None when that is under least_ms[1]. The average is cut where it
    starts again (a next note that every note ran into: a pattern repeats),
    and a note whose 100 ms windows stop matching it (NCC under `agree`) is
    cut there. The settled pitch is the QIFFT of the notes that still match
    over the first window from sustain_ms on where the drop has settled
    (_sustain_hz), searched from 200 cents under to 50 cents over the
    template's pitch at 150-400 ms. Where the drop is still falling in the
    last window three notes ring through, its later half, or the asymptote
    of the template's fitted drop where that lies under it by no more than
    twice the fall between the halves (on v1orig-1 the fit read 86 cents
    sharp); with no such window, that asymptote, else the QIFFT of the
    template's last 300 ms ("tail": sharp where the drop has not settled).
    The suite's 808s come within 5 cents of their settled pitch 0.14-0.98 s
    in (the truth's curves): v1orig-1's is still 88 cents over it at 0.3 s
    and 24 at 0.5 s, and its template read 35 cents sharp over 300-600 ms."""
    notes = [(int(s), int(e), bool(k)) for s, e, k in notes if s - PRE >= 0]
    members, span = _members(lo, notes, long_ms, max_s, True, least_ms[0])
    if len(members) < 3:
        members, span = _members(lo, notes, long_ms, max_s, False, least_ms[0])
    if not members:
        return None

    def average(ms):
        n = max(k for _, _, k in ms)
        total, count = np.zeros(n), np.zeros(n)
        for a, w, k in ms:
            total[:k] += lo[a:a + k] * w
            count[:k] += 1
        need = min(2, len(ms))
        k = int(np.argmax(count < need)) if np.any(count < need) else n
        avg = total[:k] / np.maximum(count[:k], 1)
        return avg[:_first_reattack(avg, PRE + samples(100))]

    first = average(members)
    step = samples(100)
    kept = []
    for a, w, k in members:
        end = min(k, len(first))
        for b in range(span, end - step + 1, step):
            x, y = lo[a + b:a + b + step], first[b:b + step]
            norm = float(np.linalg.norm(x) * np.linalg.norm(y))
            if norm and w * float(np.dot(x, y)) / norm < agree:
                end = b
                break
        kept.append((a, w, end))
    sr = average(kept).astype(np.float32)
    if len(sr) < PRE + samples(least_ms[1]):
        return None
    near = _rough_hz(sr[PRE + samples(150):PRE + samples(400)]) or _rough_hz(sr[PRE:]) \
        or _rough_hz(sr[PRE:], lo_hz=25.0)  # a template under 133 ms holds no two 15 Hz periods
    if not near:
        return None
    lo_hz, hi_hz = near * 2 ** (-200 / 1200), near * 2 ** (50 / 1200)
    found = _sustain_hz(lo, kept, lo_hz, hi_hz, sustain_ms, max_s)
    fitted = _settled_hz(sr[PRE:], near)
    fitted = fitted if fitted and lo_hz <= fitted <= hi_hz else None
    if found and found[2] is None:
        hz, method = found[:2]
    elif found:  # still falling: the fitted drop where it lies under the last reading by twice its fall at most
        hz, method = found[:2]
        if fitted and 0 <= 1200 * np.log2(found[0] / fitted) <= 2 * max(found[2], 0.0):
            hz, method = fitted, "drop_fit"
    elif fitted:
        hz, method = fitted, "drop_fit"
    else:
        hz = qifft(sr[max(PRE + samples(150), len(sr) - samples(300)):], lo_hz, hi_hz) or near
        method = "qifft_tail"
    from .scenarios import pitch_curve
    curve = pitch_curve(sr[PRE:], hz, upto_s=len(sr) / SR)
    audio = _extended(soxr.resample(sr.astype(np.float64), SR, LOW_SR), 3.0, hz)
    return Template(sr, audio, sorted(a + PRE for a, _, _ in kept), hz, method, curve)


def _sustain_hz(lo: np.ndarray, kept: list[tuple[int, float, int]], lo_hz: float, hi_hz: float,
                sustain_ms: tuple[float, float], max_s: float,
                settle: float = SETTLE_CENTS) -> tuple[float, str, float | None] | None:
    """(Hz, how, None) of the notes kept ((window start, gain, length) of
    lo) over the first window, from sustain_ms on in steps of half its
    length, where their drop has settled: the median QIFFT of its two halves
    agree within `settle` cents; the median QIFFT over the whole window.
    Where none has before fewer than three notes ring through a window, the
    last window's later half and how far its earlier half read over it
    (Hz, how, cents): the drop is still falling there. None without a first
    window. QIFFT reads a steady driven tone at 22-62 Hz within 0.8 cent
    over 150 ms (0.1 over 300 ms); an 808 falling 8 semitones with a 120 ms
    time constant reads 45 and 13 cents sharp over 300-450 and 450-600 ms,
    26 over 300-600 ms, and its halves come within 3 cents from 600 ms on."""
    def read(notes: list[int], x0: float, x1: float) -> float | None:
        r = [q for q in (qifft(lo[s + PRE + samples(x0):s + PRE + samples(x1)], lo_hz, hi_hz) for s in notes) if q]
        return float(np.median(r)) if len(r) >= 3 else None

    a, width = sustain_ms[0], sustain_ms[1] - sustain_ms[0]
    last = None
    while a + width <= 1000 * max_s:
        notes = [s for s, _, k in kept if k >= PRE + samples(a + width)]
        if len(notes) < 3:
            break
        first, second = read(notes, a, a + width / 2), read(notes, a + width / 2, a + width)
        whole = read(notes, a, a + width)
        if first and second and whole and abs(1200 * np.log2(second / first)) <= settle:
            return whole, f"qifft_{a:.0f}_{a + width:.0f}ms", None
        if first and second:
            last = (second, f"qifft_{a + width / 2:.0f}_{a + width:.0f}ms_falling", 1200 * np.log2(first / second))
        a += width / 2
    return last


def attack_s(tpl: Template, cents: float = 0.0) -> float:
    """Seconds from the template's start to its attack, played `cents`
    higher: audio.attack's 30 % point of the 808 band's envelope, searched
    from the notch where the template's note starts (its first PRE + 5 ms
    hold the previous note's cut tail, which crosses 30 % first: the
    attack read from the window's start came out 6-12 ms early)."""
    e = np.abs(tpl.sr[: PRE + samples(5)])
    w = samples(1)
    notch = int(np.argmin(np.convolve(e, np.ones(w) / w, mode="same")))
    i = notch + attack(tpl.sr[notch:], search=samples(250))
    return i / SR / 2 ** (cents / 1200)


# ── Notes ──

def _best_fit(x: np.ndarray, y: np.ndarray, k0: int, shifts: np.ndarray, n: int, w: int) -> tuple[float, int, float]:
    """Of the template copy y starting at k0 + each shift: (the energy of x it
    explains before it is cut, _explained, over the next w samples; that
    start; its gain, least squares over the first n), the best start."""
    ok = shifts[(k0 + shifts >= 0) & (k0 + shifts + w <= len(x))]
    if not len(ok) or len(y) < w:
        return -1.0, k0, 0.0
    rows = np.lib.stride_tricks.sliding_window_view(x[k0 + ok[0]:k0 + ok[-1] + w], w)[ok - ok[0]]
    y = y[:w]
    g = rows[:, :n] @ y[:n] / (float(np.dot(y[:n], y[:n])) or 1e-18)
    f = g[:, None] ** 2 * np.cumsum(y ** 2)[None, :] - 2 * g[:, None] * np.cumsum(rows * y[None, :], axis=1)
    gain = -np.minimum(f.min(axis=1), 0.0)
    j = int(np.argmax(gain))
    return float(gain[j]), int(k0 + ok[j]), float(g[j])


def candidates(x: np.ndarray, tpl: Template, extra=(), min_ncc: float = 0.45, head_ms: float = 80.0,
               gap_ms: float = 30.0, top: int = 3, fine=(-75, -50, -25, 25, 50, 75), span: int = 1200,
               fit_ms: float = 300.0, shift_ms: float = 20.0, step_ms: float = 0.75, reach_ms: float = 20.0
               ) -> list[dict]:
    """Where a note may start and at which pitch (x: the bass stem's 808 band
    at LOW_SR): per position (a peak of the matched filter of the template's
    first head_ms at every semitone within +-span cents, or an `extra`
    position: an envelope re-attack) the `top` semitones that score best
    within reach_ms of it, each refined, by `fine` cents and up to
    shift_ms either way, to what explains the most of the next fit_ms
    (_best_fit). On a falling attack, time and pitch trade off over a short
    head (a note 3 semitones up scored best at 2 up over 80 ms; over 300 ms,
    3 up, 10 ms later), and 16th notes of synthetic 808s that drop 8
    semitones came out 18 ms early and a semitone sharp with up to 4 ms of
    shift; the choice between a position's hypotheses is left to segment(),
    over each note. k is the template's start (LOW_SR samples), e the
    energy of x over the head there."""
    n, w = low(head_ms), low(fit_ms)
    cs = list(range(-span, span + 1, 100))
    scores = np.array([sliding(x, tpl.at(c)[:n]) for c in cs], dtype=np.float32)
    if not scores.size:
        return []
    best = scores.max(axis=0)
    ks = set(find_peaks(best, height=min_ncc, distance=low(gap_ms))[0].tolist())
    ks |= {int(round(i)) for i in extra if 0 <= i < len(best)}
    shifts = np.arange(-low(shift_ms), low(shift_ms) + 1, max(1, low(step_ms)))
    out, seen = [], set()
    for k in sorted(ks):
        a, b = max(0, k - low(reach_ms)), min(len(best), k + low(reach_ms) + 1)
        near = scores[:, a:b].max(axis=1)
        for j in np.argsort(-near)[:top]:
            if (k // low(5), j) in seen:
                continue
            seen.add((k // low(5), j))
            fits = [(*_best_fit(x, tpl.at(cs[j] + d), k, shifts, n, w), cs[j] + d) for d in (0, *fine)]
            gain, kk, g, c = max(fits)
            if gain > 0:
                head = x[kk:kk + n]
                out.append({"k": kk, "c": float(c), "g": g, "ncc": round(float(near[j]), 3),
                            "e": float(np.dot(head, head))})
    return out


def _note_costs(x: np.ndarray, tpl: Template, c: dict, m: int) -> np.ndarray:
    """F(e) for e = 0..: the squared error of the note playing from c["k"]
    for e samples (then silence) minus the energy of x there; its running
    minimum is the cost of the note when the next one starts e samples in
    (it may be cut before)."""
    y = tpl.at(c["c"])[:m]
    seg = x[c["k"]:c["k"] + len(y)]
    y = y[:len(seg)]
    return np.r_[0.0, np.cumsum(c["g"] ** 2 * y ** 2 - 2 * c["g"] * seg * y)]


def segment(x: np.ndarray, tpl: Template, cands: list[dict], penalty: float = 0.5,
            longest_s: float = 2.5) -> list[dict]:
    """The notes that explain x best: at most one candidate per position,
    each note the template at its transposition and gain from its start
    until it is cut (then silence) or the next note starts, chosen by
    dynamic programming over the squared error plus `penalty` times each
    note's head energy. A candidate inside a note (a kick, the ripple of a
    distorted low tone) would have to explain the note's sustain with the
    template's attack and drop, and mostly loses to the note going on (not
    always: legato() joins what wins back to the held note); a retrigger
    is explained by nothing else; of a position's transpositions, the one
    that explains its whole note wins. Returns the notes with "end" (the k
    of the cut or of the next note)."""
    if not cands:
        return []
    X = np.r_[0.0, np.cumsum(x ** 2)]
    m = int(longest_s * LOW_SR)
    ks = np.array([c["k"] for c in cands])
    pos = np.unique(ks)
    at = {int(p): np.flatnonzero(ks == p) for p in pos}
    best = np.full(len(cands), np.inf)
    back = np.full(len(cands), -1)
    incoming: dict[int, list[tuple[float, int]]] = {int(p): [] for p in pos}  # position: (cost so far, node)
    gone = (np.inf, -1)  # the best node whose note can no longer reach: cost up to its end, minus X there
    ended: list[tuple[int, float, int]] = []  # (position from which a note is gone, cost - X, node)
    for p in pos:
        p = int(p)
        while ended and ended[0][0] <= p:
            _, v, h = ended.pop(0)
            gone = min(gone, (v, h))
        options = [(float(X[p]), -1)] + incoming.pop(p) + ([(gone[0] + X[p], gone[1])] if gone[1] >= 0 else [])
        v, h = min(options)
        for i in at[p]:
            best[i], back[i] = v + penalty * cands[i]["e"], h
        for i in at[p]:
            rm = np.minimum.accumulate(_note_costs(x, tpl, cands[i], m))
            later = pos[(pos > p) & (pos < p + len(rm))]
            for q, cost in zip(later, best[i] + X[later] - X[p] + rm[later - p]):
                incoming[int(q)].append((float(cost), int(i)))
            ended.append((p + len(rm), best[i] - X[p] + float(rm[-1]), int(i)))
        ended.sort()
    end = len(x)
    final = []
    for i in range(len(cands)):
        rm = np.minimum.accumulate(_note_costs(x, tpl, cands[i], m))
        final.append((best[i] + X[end] - X[ks[i]] + float(rm[min(end - ks[i], len(rm) - 1)]), i))
    v, i = min(final)
    if v >= X[end]:
        return []
    chosen = []
    while i >= 0:
        chosen.append(int(i))
        i = int(back[i])
    chosen.reverse()
    notes = []
    for a, b in zip(chosen, chosen[1:] + [None]):
        f = _note_costs(x, tpl, cands[a], m)
        f = f[:min((ks[b] if b is not None else end) - ks[a], len(f) - 1) + 1]
        notes.append({**cands[a], "end": int(ks[a] + int(np.argmin(f)))})
    return notes


def _restart_at(lo: np.ndarray, tpl: Template, q: dict, kicks: np.ndarray, same: bool) -> bool:
    """Whether the 808 starts again at candidate q inside a note: its
    period-by-period pitch over 10-40 ms after q's onset sits half the
    template's own excess there (a semitone or more) over q's settled pitch
    and as much over the 35 ms before it, and by 50-85 ms is within half a
    semitone of where the template's drop puts q (with q up to 20 ms off
    the restart: a candidate 13 ms early read 0.9 semitone under the drop
    it had restarted). A retrigger at the
    note's own transposition (same) needs no drum attack from 30 ms before
    to 15 ms after it: a kick leaking into a held note reads high too."""
    from .scenarios import drop
    t0 = (q["k"] + PRE_LOW) / LOW_SR
    if same and len(kicks) and np.any((kicks >= t0 - 0.03) & (kicks <= t0 + 0.015)):
        return False
    settled = tpl.midi + q["c"] / 100
    rate = 2 ** (q["c"] / 1200)
    expected = float(np.median(drop(tpl.curve, np.linspace(0.01, 0.04, 7) * rate))) / 100
    if expected < 1.0:
        return False
    t, midi = _track(lo, t0 - 0.04, t0 + 0.085, 1.3 * 440 * 2 ** ((settled + expected - 69) / 12), skip_s=0.0)
    before, after, late = (midi[(t >= a) & (t <= b)] for a, b in ((0.005, 0.035), (0.05, 0.08), (0.09, 0.125)))
    if not (len(before) and len(after) and len(late)):
        return False
    then = [settled + float(np.median(drop(tpl.curve, (np.linspace(0.05, 0.08, 7) + d) * rate))) / 100
            for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]  # q may sit up to 20 ms off the restart
    rise = float(np.median(after))
    return rise - settled >= 0.5 * expected and rise - float(np.median(before)) >= 0.5 * expected \
        and min(abs(float(np.median(late)) - v) for v in then) < 0.5


def retriggers(found: list[dict], cands: list[dict], lo: np.ndarray, tpl: Template, kicks: np.ndarray,
               gap_ms: float = 60.0, tail_ms: float = 40.0, cluster_ms: float = 40.0) -> list[dict]:
    """segment()'s notes split where the 808 starts again inside one
    (_restart_at), at the candidate there (within cluster_ms of the first
    that shows it, the best matched-filter score), gap_ms or more after the
    start of the note's part and tail_ms or more before its end. At 160 BPM
    a pair of 16th notes at one pitch (94 ms each) was explained best by
    one note a semitone sharp: 502 of the pair's 579 units of energy,
    against 537 for the two right notes, whose extra note costs 119. Split
    here, the unit test's bar at 160 BPM gives 26 notes of 28 (24 before,
    4 written a semitone high; 2 now)."""
    out = []
    for f in found:
        part = dict(f)
        inside = sorted((q for q in cands if f["k"] + low(gap_ms) <= q["k"] <= f["end"] - low(tail_ms)),
                        key=lambda q: q["k"])
        i = 0
        while i < len(inside):
            q = inside[i]
            if q["k"] < part["k"] + low(gap_ms) or not _restart_at(lo, tpl, q, kicks, abs(q["c"] - part["c"]) < 75):
                i += 1
                continue
            near = [r for r in inside[i:] if r["k"] <= q["k"] + low(cluster_ms)
                    and _restart_at(lo, tpl, r, kicks, abs(r["c"] - part["c"]) < 75)]
            q = max(near, key=lambda r: r["ncc"])
            out.append({**part, "end": q["k"]})
            part = {**q, "end": f["end"]}
            i = next((j for j, r in enumerate(inside) if r["k"] > q["k"] + low(cluster_ms)), len(inside))
        out.append(part)
    return out


def explained_share(x: np.ndarray, tpl: Template, notes: list[dict], longest_s: float = 2.5) -> float:
    """The share of x's energy the notes explain (each from its start to its
    cut): how well a template and its notes fit, to choose between templates."""
    m = int(longest_s * LOW_SR)
    gain = 0.0
    for n in notes:
        f = _note_costs(x, tpl, n, m)
        gain -= float(f[min(n["end"] - n["k"], len(f) - 1)])
    return gain / (float(np.dot(x, x)) or 1e-18)


def _ncc_lags(x: np.ndarray, y: np.ndarray, k: int, lag: int) -> tuple[float, int]:
    """(best NCC of y against x[k + l:] for |l| <= lag, that l)."""
    n = len(y)
    if k - lag < 0 or k + lag + n > len(x) or not n:
        return -1.0, 0
    seg = x[k - lag:k + lag + n]
    corr = np.correlate(seg, y, mode="valid")
    e = np.r_[0.0, np.cumsum(seg ** 2)]
    v = corr / np.sqrt(max(float(np.dot(y, y)), 1e-18) * np.maximum(e[n:] - e[:-n], 1e-18))
    j = int(np.argmax(v))
    return float(v[j]), j - lag


def refine(x: np.ndarray, tpl: Template, note: dict, feet=(), reach_ms: float = 25.0, fit_ms: float = 300.0,
           shift_ms: float = 20.0) -> dict:
    """The note's start and transposition to the cent: the NCC of the
    template over the note (up to fit_ms) from its start and from every
    envelope foot within reach_ms (`feet`, template positions: where a cut
    and a re-attack put the onset to the millisecond), at every 10 cents
    within 150 of its own and every lag within shift_ms; then every 2
    cents within 10 of the best and a parabola. A falling attack lets a
    start 18 ms early pass for a note a semitone sharp over 107 ms; the
    foot does not. Kept only where it matches better than the note did."""
    n = min(note["end"] - note["k"], low(fit_ms))
    if n < low(40):
        return note
    lag = low(shift_ms)
    starts = [note["k"]] + [int(round(f)) for f in feet if abs(f - note["k"]) <= low(reach_ms)]
    own = _ncc_lags(x, tpl.at(note["c"])[:n], note["k"], 0)[0]
    best = (own, note["k"], note["c"])
    for k in starts:
        for c in note["c"] + np.arange(-150, 151, 10):
            v, d = _ncc_lags(x, tpl.at(c)[:n], k, lag)
            if v > best[0]:
                best = (v, k + d, float(c))
    if best[1:] == (note["k"], note["c"]) and best[0] == own:
        cs = note["c"] + np.arange(-10, 11, 2)
    else:
        cs = best[2] + np.arange(-10, 11, 2)
    k = best[1]
    v = np.array([_ncc_lags(x, tpl.at(c)[:n], k, 0)[0] for c in cs])
    j = int(np.argmax(v))
    d = 0.0
    if 0 < j < len(v) - 1:
        den = v[j - 1] - 2 * v[j] + v[j + 1]
        d = float(np.clip(0.5 * (v[j - 1] - v[j + 1]) / den, -0.5, 0.5)) if den else 0.0
    if v[j] < own:
        return {**note, "ncc": round(own, 3)}
    c = float(cs[j] + 2 * d)
    y = tpl.at(c)[:n]
    return {**note, "k": int(k), "c": c, "ncc": round(float(v[j]), 3), "g": _gain(x[k:k + n], y)}


# ── Transcription ──

def sounding(tpl: Template, settled: float, cents: float, dur: float) -> float:
    """What a note heard as the template played `cents` higher sounds like
    (MIDI), by the truth's rule (scenarios.sounding): the settled pitch plus
    the template's own drop read at the note's rate, the median over the
    note's read window. A 110 ms note never reaches its settled pitch."""
    from .scenarios import drop
    from .scenarios import sounding as heard
    rate = 2 ** (cents / 1200)
    return heard(lambda t: settled + drop(tpl.curve, t * rate) / 100, dur)


def _lag(ref: Template, tpl: Template, span_ms: float = 150.0, reach_ms: float = 30.0) -> int:
    """How many samples after ref's onset tpl's onset sits, on the same
    waveform: ref played at tpl's pitch against tpl over their first span_ms
    after the onset, the best NCC within reach_ms."""
    from .audio import transpose
    a = transpose(ref.sr, tpl.midi - ref.midi)[PRE:PRE + samples(span_ms)]
    b = tpl.sr[PRE:PRE + samples(span_ms)]
    n = min(len(a), len(b))
    if n < samples(50):
        return 0
    corr = [float(np.dot(a[max(0, d):n + min(0, d)], b[max(0, -d):n - max(0, d)]))
            for d in range(-samples(reach_ms), samples(reach_ms) + 1, samples(0.5))]
    return (int(np.argmax(corr)) * samples(0.5)) - samples(reach_ms)


def settle(lo: np.ndarray, tpl: Template, spans: list[tuple[float, float, float]], least: int = 3,
           width_s: float = 0.05, steady: float = 50.0) -> Template:
    """The template with its settled pitch read on the notes found, where
    learn() had no window in which its notes' drop had settled (a drop fit,
    a still-falling window, or the tail). spans: each note's (start, end,
    transposition in cents). A note c cents up plays the sample 2^(c/1200)
    times as fast, so its period-by-period pitch less c, at that many times
    its own time, is the template's: the notes reach past the template's
    end. The reading: the median over the latest width_s of template time
    that `least` notes reach, where its two halves agree within `steady`
    cents (further apart, the notes slide there) and it lies under the
    template's (a drop still falling is read sharp: one exponential leaves
    out its slow tail). The pitch curve is moved with it, so every note's
    heard pitch stays as it was.

    On the suite's 48 beats, against the truth, learn() had no settled
    window on 11 (htdemucs) and 6 (htdemucs_ft), and this reading was taken
    on 3 and 4 of them: bass_low-1's template read 59.7 and 40.9 cents
    sharp from 110-330 ms notes, its +7 notes (330 ms, 0.45 s of template
    time) 13.8 and 10.0; swing66-1's 28.0, now 9.5; full-2's moved from +1.4
    and +1.0 to -2.8 and -6.4, meter34-2's from +1.6 to +1.4 and v1orig-1's
    from -1.0 to -1.8 (htdemucs_ft). On the others the reading was over the
    template's, or not steady (producer-2: halves 298 and 442 cents apart,
    where the notes' ends slide). Replayed on the notes legato() was given."""
    if tpl.method.startswith("qifft_") and tpl.method != "qifft_tail" and not tpl.method.endswith("_falling"):
        return tpl
    pts = []
    for j, (t, end, c) in enumerate(spans):
        rate = 2 ** (c / 1200)
        s, midi = _track(lo, t, end, 1.3 * tpl.hz * rate * 2 ** (2 / 12), skip_s=0.0)
        keep = s < (end - t) - 0.01  # the cut's last period aside
        pts += [(a * rate, m - c / 100, j) for a, m in zip(s[keep], midi[keep])]
    if not pts:
        return tpl
    pts = np.array(pts)
    reach = np.sort([pts[pts[:, 2] == j, 0].max() for j in np.unique(pts[:, 2])])[::-1]
    if len(reach) < least:
        return tpl
    b = float(reach[least - 1])
    first = pts[(pts[:, 0] >= b - width_s) & (pts[:, 0] < b - width_s / 2), 1]
    second = pts[(pts[:, 0] >= b - width_s / 2) & (pts[:, 0] <= b), 1]
    if not len(first) or not len(second) or 100 * abs(np.median(first) - np.median(second)) > steady:
        return tpl
    midi = float(np.median(np.r_[first, second]))
    if midi >= tpl.midi:
        return tpl
    hz = 440 * 2 ** ((midi - 69) / 12)
    shift = 1200 * np.log2(tpl.hz / hz)
    return replace(tpl, hz=hz, method=f"notes_{1000 * (b - width_s):.0f}_{1000 * b:.0f}ms",
                   curve=[[s, round(c + shift, 1)] for s, c in tpl.curve])


def _late(starts: list[int], kicks: np.ndarray, before_ms: float = 100.0, least: int = 5,
          late_ms: float = 35.0) -> int:
    """How late (samples) the notes found start against the drum attacks
    just before them (within before_ms): the median, when at least `least`
    notes have one and it is over late_ms; 0 otherwise. An 808 starts with
    its kick or 0-20 ms after it, and the drum stem holds some of the 808's
    own attack: on the suite's bass_low beats, whose templates are on time,
    the median came to 29 ms. A template learnt from notes found inside
    themselves (one 808 of the suite dips 13 dB 75 ms into every note) puts
    every note 40-75 ms late."""
    if not len(kicks):
        return 0
    d = []
    for s in starts:
        j = int(np.searchsorted(kicks, s + samples(10), side="right")) - 1
        if j >= 0 and s - kicks[j] <= samples(before_ms):
            d.append(s - kicks[j])
    med = float(np.median(d)) if len(d) >= least else 0.0
    return int(med) if med > samples(late_ms) else 0


def _heard(tpl: Template, n: dict) -> float:
    """sounding() of a note, its legato glide included: from the note before
    to its own pitch, linear over the glide's ms (the sample read on, its
    drop long settled)."""
    g = n.get("glide") or {}
    dur = n["end"] - n["t"]
    if g.get("kind") == "legato":
        from .scenarios import sounding as heard
        a, b, ms = g["from_midi"], g["to_midi"], max(g["ms"], 1.0) / 1000
        return heard(lambda t: a + (b - a) * np.clip(t / ms, 0.0, 1.0), dur)
    return sounding(tpl, n["midi_settled"], 100 * (n["midi_settled"] - tpl.midi), dur)


def _learnt(lo: np.ndarray, x: np.ndarray, att: list[dict], strong, extra: list[float], passes: int,
            penalty: float, flag, kicks: np.ndarray, least_ms: tuple[float, float] = (120.0, 200.0)
            ) -> tuple[float, Template | None, list[dict], list[dict]]:
    """A template learnt from the re-attacks that `flag` marks, then from the
    notes it finds, `passes` times: the pass whose notes explain the most of
    x (explained_share), with that share, its notes and its candidates. A later pass keeps
    the first one's onset (_lag: a note found 16 ms late, a falling attack
    trading time for pitch, moves every note of a template learnt from it),
    and a template whose notes start late against the kicks (_late) is
    learnt again that much earlier."""
    onsets = [o["i"] for o in att]
    notes = [(s, e, flag(strong(o["i"]), o["notch_db"] >= DEEP_NOTCH_DB))
             for s, e, o in zip(onsets, onsets[1:] + [len(lo)], att)]
    best: tuple = (-np.inf, None, [], [])
    first = None
    for _ in range(passes):
        tpl = learn(lo, notes, least_ms=least_ms)
        if tpl is None:
            break
        if first is not None and abs(d := _lag(first, tpl)) > samples(1):
            tpl = learn(lo, [(s + d, e, k) for s, e, k in notes], least_ms=least_ms) or tpl
        found = segment(x, tpl, cands := candidates(x, tpl, extra), penalty)
        starts = [int(round((f["k"] + PRE_LOW) * DS)) for f in found]
        if late := _late(starts, kicks):
            tpl = learn(lo, [(s - late, e, k) for s, e, k in notes], least_ms=least_ms) or tpl
            found = segment(x, tpl, cands := candidates(x, tpl, extra), penalty)
            starts = [int(round((f["k"] + PRE_LOW) * DS)) for f in found]
        first = first or tpl
        share = explained_share(x, tpl, found)
        if share > best[0]:
            best = (share, tpl, found, cands)
        notes = [(s, int(round((f["end"] + PRE_LOW) * DS)), flag(strong(s), False)) for s, f in zip(starts, found)]
    return best


def _track(lo: np.ndarray, t: float, end: float, near_hz: float, skip_s: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """(seconds after t, MIDI) of every period of lo from t + skip_s to end
    (scenarios.crossings near near_hz)."""
    from .scenarios import crossings
    a, b = int(round(t * SR)), int(round(end * SR))
    z = crossings(lo[a:b], near_hz) if b - a > samples(60) else np.zeros(0)
    if len(z) < 4:
        return np.zeros(0), np.zeros(0)
    mid, midi = (z[1:] + z[:-1]) / 2, 69 + 12 * np.log2(1 / np.diff(z) / 440)
    keep = mid >= skip_s
    return mid[keep], midi[keep]


def _settles_at(lo: np.ndarray, a: dict, n: dict, *others: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """(seconds after a's start, MIDI) of every period from a's start to n's
    end (_track, near 1.3 times the highest of the notes' settled pitches),
    and the pitch n settles at: the median of its later 60 %."""
    hz = 440 * 2 ** ((max(m["midi_settled"] for m in (a, n, *others)) - 69) / 12)
    t, midi = _track(lo, a["t"], n["end"], 1.3 * hz)
    later = midi[t >= (n["t"] - a["t"]) + 0.4 * (n["end"] - n["t"])] if len(t) else midi
    return t, midi, float(np.median(later)) if len(later) else n["midi_settled"]


def _top(env: np.ndarray, n: dict, ms: float = 200.0) -> int:
    """The sample of env's highest point over a note's first ms, or up to
    its end: where its own peak is read (a note's "peak" reads 200 ms on,
    the next note's attack too)."""
    i = min(max(0, int(round(n["t"] * SR))), len(env) - 1)
    return i + int(np.argmax(env[i:i + max(1, samples(1000 * min(ms / 1000, n["end"] - n["t"])))]))


def silent_from(env: np.ndarray, a: int, b: int, under: float, least_s: float = SILENT_S) -> int | None:
    """The first sample in [a, b) from which env stays under `under` for
    least_s or more (the stretch may run on past b); None where it does not."""
    n = samples(1000 * least_s)
    quiet = env[a:min(len(env), b + n)] < under
    edges = np.flatnonzero(np.diff(np.r_[0, quiet.astype(np.int8), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if s >= b - a:
            break
        if e - s >= n:
            return a + int(s)
    return None


def audible(notes: list[dict], env: np.ndarray, under_db: float = SILENT_DB) -> list[dict]:
    """The notes whose own peak (env, the 808 band's 25 ms RMS, up to the
    note's end: _top) is less than under_db under the median of the
    notes' peaks: a note found where the 808 is silent is not one. On
    PIRAMIT, 6 notes were found in its rests, 49-78 dB under its notes'
    median peak (NCC 0.12-0.65); legato() then joined them to the note
    before, which held the rest, or the 808 notes after them to them. On
    the suite's beats no note is: the quietest sits 34.3 dB under."""
    if not notes:
        return notes
    peaks = np.array([env[_top(env, n)] for n in notes])
    return [n for n, p in zip(notes, peaks) if p > float(np.median(peaks)) * 10 ** (-under_db / 20)]


def legato(notes: list[dict], lo: np.ndarray, tpl: Template, feet: np.ndarray, kicks: np.ndarray,
           attacked_ncc: float = ATTACKED_NCC, slides: bool = SLIDES, step: float = LEGATO_STEP,
           shortest: float = LEGATO_SHORTEST, reach: float = LEGATO_REACH, split_s: float = SPLIT_S,
           silent_db: float = SILENT_DB) -> list[dict]:
    """Notes found where nothing attacks (no envelope foot within 20 ms, no
    drum attack from 30 ms before to 15 ms after, a template fit under
    attacked_ncc, and no restart of the sample's drop: _restarts) are not
    notes of their own but the note before going on. A slide without an
    attack made segment() explain it with three or four short restarts at
    wrong pitches (glides scenario: 48 truth notes, up to 130 found), and
    the sustain of a long note, once the template's own has decayed, with
    short notes 8-10 semitones low (kick808-2).

    Nothing goes on across a silence (the 808 band's 25 ms RMS silent_db
    under the note's peak for SILENT_S: silent_from): a note found after
    the note before has fallen silent starts again, and every note ends
    where it falls silent. On PIRAMIT a note of 0.06 s took in four notes
    of an 808 playing alone after a 1.2 s rest, then a 7.4 s rest, and held
    15.3 s: those notes start 27 ms before their envelope rises, and their
    template fit is 0.887-0.893 (the fit trades time for pitch: at the
    foot, at the template's own pitch, 0.848-0.861).

    A rise (a foot whose envelope rises RISE_DB or more, as reattacks()
    finds a note after a rest or a decay) from 20 ms before a note's onset
    to its peak (up to its end, and nearer its onset than the next note's)
    attacks it too. PIRAMIT's notes of an 808 without a kick rise 11.1-20.6
    dB 26.6-27.5 ms after their onsets; on the suite's 48 beats (both
    models), the 41 notes merged with a foot 20-40 ms after their onset
    rise -5.6 to +7.9 dB.

    A note kept by a rest or a rise alone was placed by segment() on the
    template's fit alone. It starts on the foot that attacks it (the rise
    of RISE_DB or more after the rest, or that rise: the one that rises
    most; none, it stays), its attack moved with it (_seam), and once its
    end is known (merged, cut at a silence) it takes the template's whole
    transposition nearest the pitch its periods settle at, where its own
    lies more than half a semitone from it (_repitch). On PIRAMIT the 8
    such notes attacked 21.4-22.3 ms before their foot (6), 50.2 ms after
    it (1) and 28.8 ms after a foot in the rest that no rise follows (1:
    the attack's own foot was merged into it, and the note stays); the 7
    moved now attack 5.2-5.4 ms after it (the template's own attack), and
    the two written C#2 (+1310 cents) and C#1 (+84.5) where the stem holds
    C1 (24.03 by QIFFT) are C1: of the notes of 200 ms or more, 88 of 98
    are written as the stem holds them (86 before).

    A kick inside a held note: an attacked note of split_s or less whose
    unattacked follower settles (period by period) within half a semitone
    of the note before it, which it cut, is a kick leaking into the bass
    stem; it and the follower are that note going on (kick808-2: notes of
    54-73 ms at MIDI 17-23 inside a note held at 22.1, each followed by a
    slide back to 22.1).

    With slides, an unattacked note becomes a legato note (glide: from, to,
    t, ms, linear; at the template's nearest semitone) where the evidence
    for a slide holds (_slide). Without it (the default) none does: on the
    suite's 24 lane beats the earlier rule made 79 legato notes for the
    truth's 24 (htdemucs; 120 on htdemucs_ft), 6 of them right (5), and
    merging every one scored better on every 808 number (_slide has the
    replay)."""
    env = rms_envelope(lo, 25.0)
    quiet = 10 ** (-silent_db / 20)
    e = envelope(lo)
    feet = np.asarray(feet, dtype=float)
    up = np.array([rise_db(e, int(round(f * SR)), 30.0, 40.0, 0.0) for f in feet])

    def silent(n: dict, until: int) -> int | None:  # where n falls silent before sample `until`
        i = _top(env, n)
        return silent_from(env, i, until, env[i] * quiet)

    def rest(a: dict, b: dict) -> np.ndarray | None:  # a falls silent before b's peak: the rises from there on
        top = _top(env, b)
        s = silent(a, top)
        return None if s is None else np.flatnonzero((up >= RISE_DB) & (feet >= s / SR) & (feet <= top / SR))

    def risen(n: dict, nxt: float) -> np.ndarray:  # rises from 20 ms before n's onset to its peak, nearer n than nxt
        top = _top(env, n) / SR
        return np.flatnonzero((up >= RISE_DB) & (feet >= n["t"] - 0.02) & (feet <= min(top, (n["t"] + nxt) / 2)))

    out: list[dict] = []
    seams: list[dict] = []
    for j, n in enumerate(notes):
        nxt = notes[j + 1]["t"] if j + 1 < len(notes) else np.inf
        if not out or _attacked(n, feet, kicks, attacked_ncc) or _restarts(lo, tpl, n):
            out.append(dict(n))
            continue
        after, rose = rest(out[-1], n), risen(n, nxt)
        if after is not None or len(rose):
            at = rose if len(rose) else after
            out.append(_seam(n, float(feet[at[np.argmax(up[at])]]) if len(at) else None))
            seams.append(out[-1])
            continue
        prev = out[-1]
        if len(out) > 1 and "legato" not in prev["flags"] and prev["end"] - prev["t"] <= split_s \
                and out[-2]["end"] >= prev["t"] - 0.005:
            held = out[-2]
            if abs(_settles_at(lo, prev, n, held)[2] - held["midi_settled"]) < 0.5:
                out.pop()
                held["end"] = n["end"]
                continue
        glide = _slide(lo, tpl, prev, n, step, shortest, reach) if slides else None
        if glide is None:
            prev["end"] = n["end"]
            continue
        prev["end"] = glide["t"]
        out.append({**n, "t": glide["t"], "t_attack": glide["t"], "midi_settled": glide["to_midi"],
                    "transposition": round(100 * (glide["to_midi"] - tpl.midi), 1), "glide": glide,
                    "flags": ["legato"]})
    for n in out:
        cut = silent(n, int(round(n["end"] * SR)))
        if cut is not None:
            n["end"] = round(cut / SR, 5)
        if any(n is m for m in seams):
            _repitch(lo, tpl, n)
    return out


def _reread(lo: np.ndarray, tpl: Template, n: dict, least: int = 4) -> float | None:
    """The settled pitch (MIDI) of note n read on the stem: the median of
    its periods (_track near 1.3 times the higher of its own and the
    template's pitch) over the later 60 % of the note, or of its part before
    the 808 starts again (from 60 ms in, a period half a semitone over the
    lowest so far: its own pitch only falls), the last period, which the
    cut ends, aside; less the template's drop there at the rate that
    reading implies. None with fewer than `least` periods."""
    hz = 440 * 2 ** ((max(n["midi_settled"], tpl.midi) - 69) / 12)
    t, midi = _track(lo, n["t"], n["end"], 1.3 * hz, skip_s=0.0)
    again = np.flatnonzero((t >= 0.06) & (midi >= np.minimum.accumulate(np.where(t >= 0.06, midi, np.inf)) + 0.5))
    span = t[again[0]] if len(again) else n["end"] - n["t"]
    keep = np.flatnonzero((t >= 0.4 * span) & (t < span))[:-1]
    if len(keep) < least:
        return None
    from .scenarios import drop
    rate = 2 ** ((float(np.median(midi[keep])) - tpl.midi) / 12)
    return float(np.median(midi[keep] - drop(tpl.curve, t[keep] * rate) / 100))


def _seam(n: dict, foot: float | None) -> dict:
    """Note n moved to the foot that attacks it, its attack with it."""
    m = dict(n)
    if foot is not None:
        m["t"], m["t_attack"] = round(foot, 5), round(n["t_attack"] + foot - n["t"], 5)
    return m


def _repitch(lo: np.ndarray, tpl: Template, n: dict, apart: float = 0.5) -> None:
    """Where note n's settled pitch lies more than `apart` semitones from
    the stem's (_reread), n takes the template's nearest whole
    transposition to the stem's, its attack read at that rate (in place)."""
    heard = _reread(lo, tpl, n)
    if heard is None or abs(heard - n["midi_settled"]) <= apart:
        return
    c = 100.0 * round(heard - tpl.midi)
    lead = (n["t_attack"] - n["t"] + PRE / SR) * 2 ** ((100 * (n["midi_settled"] - tpl.midi) - c) / 1200)
    n.update(midi_settled=round(tpl.midi + c / 100, 4), transposition=c, t_attack=round(n["t"] + lead - PRE / SR, 5))


def _slide(lo: np.ndarray, tpl: Template, prev: dict, n: dict, step: float = LEGATO_STEP,
           shortest: float = LEGATO_SHORTEST, reach: float = LEGATO_REACH, after: float = SLIDE_AFTER) -> dict | None:
    """The glide of an unattacked note n that slides from prev, or None.
    Evidence for a slide: the period-by-period pitch settles `step`
    semitones or more from prev's, within `reach` of the template and 15
    cents of one of its whole transpositions, steady there (an interquartile
    range of 0.35 semitone at most over n's later 60 %), n lasts `shortest`
    or more, prev is a note of its own, and the pitch leaves prev for good
    `after` or more after prev starts. For good: the last run of periods
    before n that sit 0.3 semitone or more past prev (its own drop
    included), towards the new pitch; a slide that never stayed on prev
    (the first period already moved) is not one.

    Replayed on the suite's 24 lane beats (the step's inputs captured from
    the 808-after4 runs; analysis level, mean over the 10 scenarios), the
    earlier rule (no evidence past the step, the start falling back to the
    first period), merging every note and this rule gave note F1@50 0.905,
    0.917 and 0.920, note F1@20 0.880, 0.902 and 0.899, onset F1@20 0.911,
    0.936 and 0.931 (htdemucs_ft: 0.856, 0.862, 0.862; 0.825, 0.837,
    0.834; 0.875, 0.891, 0.888), with 18 legato notes of which 6 right
    (htdemucs_ft: 13 and 5) for the truth's 24, `after` 0.2 s; 0.1 s (a
    slide one 16th into a note) gave 0.919 and 0.863, 20 legato notes each.
    Its thresholds were chosen on those beats, and it still loses on two
    numbers of three: slides stay off by default."""
    t, midi, to = _settles_at(lo, prev, n)
    if "legato" in prev["flags"] or abs(to - prev["midi_settled"]) < step or abs(to - tpl.midi) > reach \
            or n["end"] - n["t"] < shortest or abs((to - tpl.midi) - round(to - tpl.midi)) > 0.15:
        return None
    later = midi[t >= (n["t"] - prev["t"]) + 0.4 * (n["end"] - n["t"])]
    if len(later) and np.subtract(*np.percentile(later, [75, 25])) > 0.35:
        return None
    to = tpl.midi + round(to - tpl.midi)
    moved = np.sign(to - prev["midi_settled"]) * (midi - prev["midi_settled"] - _drop(tpl, prev, t)) > 0.3
    before = np.flatnonzero(t < n["t"] - prev["t"])
    if len(before) and moved[before[-1]]:  # back from the note found while the pitch stays moved
        back = np.flatnonzero(~moved[before])
        if not len(back):
            return None
        start = prev["t"] + float(t[before[back[-1] + 1]])
    else:
        start = n["t"]
    if start - prev["t"] < after:
        return None
    there = np.flatnonzero((t >= start - prev["t"]) & (np.abs(midi - to) < 0.3))
    ms = float(np.clip(1000 * ((prev["t"] + t[there[0]] if len(there) else n["t"]) - start), 20, 400))
    return {"from_midi": prev["midi_settled"], "to_midi": round(to, 4), "t": round(start, 5), "ms": round(ms, 1),
            "shape": "lin", "kind": "legato"}


def kick_splits(notes: list[dict], kicks: np.ndarray, longest: float = 0.08, edge: float = 1200.0) -> list[dict]:
    """A note under `longest` seconds that starts on a drum attack (30 ms
    before to 15 ms after it) at a transposition past candidates()' whole
    semitones (|cents| over `edge`, reached only by its fine steps and
    refine) and runs straight into the next note is the kick's attack split
    off that note, which starts there instead (flag "split"). On PIRAMIT 13
    notes read as C#2 (+1256 to +1310 cents, 60-90 ms, 10 on a kick, each
    followed at once by a C1 note, with an autocorrelation of 0.40-0.51 at
    their own period): the 10 on a kick are joined to their C1 notes, which
    start 60 ms earlier. None of the suite's beats has one (not verified:
    PIRAMIT has no truth)."""
    out: list[dict] = []
    for n, nxt in zip(notes, notes[1:] + [None]):
        kick = len(kicks) and bool(np.any((kicks >= n["t"] - 0.03) & (kicks <= n["t"] + 0.015)))
        if nxt is not None and kick and n["end"] - n["t"] < longest and abs(n["transposition"]) > edge \
                and nxt["t"] - n["end"] < 0.005:
            nxt.update(t_attack=round(n["t"] + nxt["t_attack"] - nxt["t"], 5), t=n["t"], kick=True,
                       flags=list(nxt["flags"]) + ["split"])
            continue
        out.append(n)
    return out


def _drop(tpl: Template, n: dict, t: np.ndarray) -> np.ndarray:
    """Semitones the note sits over its settled pitch t seconds in: the
    template's own drop read at the note's rate."""
    from .scenarios import drop
    return drop(tpl.curve, np.asarray(t) * 2 ** ((n["midi_settled"] - tpl.midi) / 12)) / 100


def _attacked(n: dict, feet: np.ndarray, kicks: np.ndarray, attacked_ncc: float) -> bool:
    """Whether something attacks where the note starts: an envelope foot
    within 20 ms, a drum attack from 30 ms before to 15 ms after, or the
    template's attack and drop matching (NCC at least attacked_ncc)."""
    foot = len(feet) and float(np.min(np.abs(feet - n["t"]))) <= 0.02
    kick = len(kicks) and bool(np.any((kicks >= n["t"] - 0.03) & (kicks <= n["t"] + 0.015)))
    return bool(foot or kick or n.get("ncc", 1.0) >= attacked_ncc)


def _restarts(lo: np.ndarray, tpl: Template, n: dict) -> bool:
    """Whether the note's pitch starts high again as the template's drop
    does (from 10 to 40 ms in, at least half the template's own excess
    there, where that is a semitone or more): a retrigger restarts the
    sample, a legato slide does not."""
    from .scenarios import drop
    rate = 2 ** ((n["midi_settled"] - tpl.midi) / 12)
    expected = float(np.median(drop(tpl.curve, np.linspace(0.01, 0.04, 7) * rate)))
    if expected < 100:
        return False
    hz = 440 * 2 ** ((n["midi_settled"] + expected / 100 - 69) / 12)
    t, midi = _track(lo, n["t"], min(n["end"], n["t"] + 0.06), 1.3 * hz, skip_s=0.01)
    early = midi[t <= 0.04]
    return bool(len(early)) and float(np.median(early) - n["midi_settled"]) * 100 >= 0.5 * expected


FLAGS = {"drums": lambda drum, notch: drum, "drums|notch": lambda drum, notch: drum or notch,
         "all": lambda drum, notch: True}


def written(lo: np.ndarray, tpl: Template, found: list[dict], feet: np.ndarray, kicks: np.ndarray, strong,
            slides: bool = SLIDES) -> tuple[Template, list[dict]]:
    """The notes found (refine's, at LOW_SR) as bass808.json holds them, and
    the template, its settled pitch read on them where learn() had none
    (settle): audible ones, kick_splits, legato, then each note's written
    pitch, cents and heard pitch. feet: the envelope's feet (seconds);
    kicks: the drum attacks (seconds); strong(i): a drum attack at sample i."""
    env = rms_envelope(lo, 25.0)
    starts = [(f["k"] + PRE_LOW) / LOW_SR for f in found]
    tpl = settle(lo, tpl, [(s, min((f["end"] + PRE_LOW) / LOW_SR, nxt), f["c"])
                           for s, f, nxt in zip(starts, found, starts[1:] + [np.inf])])
    tuning = tpl.midi - int(np.floor(tpl.midi + 0.5))
    notes = []
    for f, t, nxt in zip(found, starts, starts[1:] + [np.inf]):
        end = min((f["end"] + PRE_LOW) / LOW_SR, nxt)
        settled = tpl.midi + f["c"] / 100
        pitch = int(np.floor(settled - tuning + 0.5))
        i = int(round(t * SR))
        notes.append({"t": round(t, 5), "t_attack": round(f["k"] / LOW_SR + attack_s(tpl, f["c"]), 5),
                      "end": round(end, 5), "pitch": pitch, "midi_settled": round(settled, 4),
                      "midi_float": round(sounding(tpl, settled, f["c"], end - t), 4),
                      "cents": round(100 * (settled - pitch), 1), "transposition": round(f["c"], 1),
                      "gain": round(f["g"], 4), "ncc": f["ncc"],
                      "peak": round(float(env[i:i + samples(200)].max()) if i < len(env) else 0.0, 5),
                      "kick": bool(strong(i)), "flags": []})
    notes = legato(kick_splits(audible(notes, env), kicks), lo, tpl, feet, kicks, ATTACKED_NCC, slides)
    for n, nxt in zip(notes, notes[1:] + [None]):
        n["end"] = round(min(n["end"], nxt["t"]) if nxt else n["end"], 5)
        n["pitch"] = int(np.floor(n["midi_settled"] - tuning + 0.5))
        n["cents"] = round(100 * (n["midi_settled"] - n["pitch"]), 1)
        n["midi_float"] = round(_heard(tpl, n), 4)
    return tpl, notes


def transcribe(bass: np.ndarray, drums: np.ndarray | None = None, passes: int = 2, penalty: float = 0.5,
               slides: bool = SLIDES, log=None) -> dict:
    """The 808 notes of a bass stem (drums: the drum stem, whose kicks flag
    the notes that start on them, choose between templates, and mark onsets
    where the re-attacks are too few to learn a template from): what
    bass808.json holds, {"template": ..., "notes": [...]}. With slides,
    legato notes (legato()).

    Per note: t (where it starts: the template's onset), t_attack (where it
    attacks: the template's attack at the note's rate, attack_s), end (cut,
    or the next note), pitch (the written MIDI note: the settled pitch less
    the 808's own tuning, rounded), midi_settled (the template's settled
    pitch plus the note's transposition), midi_float (the pitch heard,
    sounding()), cents (midi_settled over pitch), transposition (cents re
    the template), gain (re the template), ncc, peak (the 808 band's 25 ms
    RMS peak over the first 200 ms, as notes_808 had it), kick (a drum
    attack at its onset), flags ("legato"), and a legato note's glide
    (from_midi, to_midi, t, ms, shape "lin", kind "legato": legato())."""
    lo = band(bass, "808")
    x = lowband(bass)
    att = reattacks(lo)
    if drums is not None:
        ed = rms_envelope(band(drums, "low"))
        loud = ed.max() * 10 ** (-30 / 20)
    else:
        ed, loud = None, 0.0
    kicks = drum_attacks(ed) if ed is not None else np.zeros(0, dtype=int)

    def strong(i: int) -> bool:
        return ed is not None and rise_db(ed, i) >= DRUM_RISE_DB and ed[i:i + samples(15)].max() > loud

    def tries(att: list[dict], label: str = "", least_ms=(120.0, 200.0)) -> list[tuple]:
        extra = [o["i"] / DS - PRE_LOW for o in att]
        out, seen = [], set()
        for name in VARIANTS:
            key = tuple(sorted(o["i"] for o in att if FLAGS[name](strong(o["i"]), o["notch_db"] >= DEEP_NOTCH_DB)))
            if key in seen:
                continue
            seen.add(key)
            share, tpl, found, cands = _learnt(lo, x, att, strong, extra, passes, penalty, FLAGS[name], kicks,
                                               least_ms)
            if tpl is None:
                continue
            on_kicks = float(np.mean([strong(s) for s in tpl.starts])) if tpl.starts else 0.0
            if log:
                log(f"  808 template from {name}{label} onsets: {len(tpl.starts)} notes ({on_kicks:.0%} on a drum "
                    f"attack), {tpl.hz:.2f} Hz ({tpl.method}), {len(found)} notes explain {share:.1%} of the 808 band")
            out.append((share, on_kicks, name + label, tpl, found, cands))
        return out

    tried = tries(att)
    if not tried and len(kicks):
        # the re-attacks too few to learn from (a driven 33 Hz tone's notch at a cut can be 0.5-2.3 dB, under
        # NOTCH_DB; synthetic lines at 160-170 BPM gave 6-10 of 28-32): the kicks mark onsets too
        att = sorted(att + [{"i": int(k), "notch_db": 0.0, "rise_db": 0.0} for k in kicks
                            if not any(abs(o["i"] - k) <= samples(30) for o in att)], key=lambda o: o["i"])
        tried = tries(att, "+kicks")
    if not tried:  # no note rings 200 ms: a template of shorter ones, its settled pitch read on the drop's tail
        tried = tries(att, "+short", (80.0, 100.0))
    if not tried:
        return {"template": None, "notes": []}
    # within TIE of the best fit, the template whose notes start on drum attacks (on two suite beats a template
    # of notes found inside their notes explained 0.2-0.7 % more and put every note 40-54 ms late), then one
    # whose settled pitch is within 10 cents of their median (a drop fit on a 360 ms template read 78 cents
    # flat where two others, as much on drum attacks, read the line's pitch: a synthetic line at 170 BPM)
    top = max(t[0] for t in tried)
    close = [t for t in tried if t[0] >= top - TIE]
    mid = float(np.median([t[3].hz for t in close]))
    best = max(close, key=lambda t: (t[1], abs(1200 * np.log2(t[3].hz / mid)) <= 10, t[0]))
    share, on_kicks, name, tpl, found, cands = best
    feet = [o["i"] / DS - PRE_LOW for o in att]
    found = [refine(x, tpl, f, feet) for f in retriggers(found, cands, lo, tpl, kicks / SR)]
    tpl, notes = written(lo, tpl, found, np.array([o["i"] for o in att]) / SR, kicks / SR, strong, slides)
    root = int(np.floor(tpl.midi + 0.5))
    tuning = tpl.midi - root
    curve = tpl.curve[::2]
    return {"template": {"notes_used": len(tpl.starts), "onsets_s": [round(s / SR, 5) for s in tpl.starts],
                         "hz": round(tpl.hz, 4), "midi": round(tpl.midi, 4), "root_midi": root,
                         "tuning_cents": round(100 * tuning, 2), "method": tpl.method,
                         "length_s": round(len(tpl.sr) / SR, 3), "attack_ms": round(1000 * attack_s(tpl), 2),
                         "learnt_from": name, "on_drum_attacks": round(on_kicks, 3), "explained": round(share, 4),
                         "curve": curve},
            "notes": notes}


def project(b808: dict) -> list[dict]:
    """bass808.json's notes as analysis.json's notes_808: t, t_attack, end,
    pitch (written), midi_float (heard), midi_settled, cents, slide (the
    slide out of the note in semitones: the next note's legato glide, as
    the pYIN reader read a slide at a note's end), peak, glide and flags."""
    notes = b808["notes"]
    out = []
    for n, nxt in zip(notes, notes[1:] + [None]):
        g = (nxt or {}).get("glide") or {}
        slide = g["to_midi"] - g["from_midi"] if g.get("kind") == "legato" else 0.0
        out.append({"t": n["t"], "t_attack": n["t_attack"], "end": n["end"], "pitch": n["pitch"],
                    "midi_float": n["midi_float"], "midi_settled": n["midi_settled"], "cents": n["cents"],
                    "slide": round(float(slide), 2), "peak": n["peak"], "glide": n.get("glide"),
                    "flags": list(n.get("flags", []))})
    return out
