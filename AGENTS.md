# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

MNE-Connectivity estimates connectivity between sensors or sources of neurophysiological
data (MEG, EEG, iEEG) and stores the result in xarray-backed container classes. MNE-Python
owns what comes in (`Epochs`, `SourceEstimate`, plain arrays) and the spectral machinery
(multitaper, Morlet, `EpochsTFR`); this package owns everything from the cross-spectral
density onward, plus the containers, their netCDF I/O, and the connectivity-specific
visualization.

## Follow MNE-Python's conventions

This package is a subsidiary of MNE-Python. Unless something below says otherwise, follow
[MNE-Python's AGENTS.md](https://github.com/mne-tools/mne-python/blob/main/AGENTS.md)
(read it rather than guessing): naming, numpydoc style with its local deviations, absolute
and lazily-nested imports, `@verbose`, shared docstrings via `@fill_doc`, `# TODO VERSION`
markers, towncrier changelog fragments, compact tests, license rules for adapted code, and
in particular its
[policy on AI assistance](https://github.com/mne-tools/mne-python/blob/main/CONTRIBUTING.md#policy-on-ai-assistance-in-contributions):

- Work test-first: write (or extend) a test that fails for the right reason, then make it
  pass. Promote anything a throwaway script caught into a real test.
- Do not open pull requests, push, or commit unless explicitly asked; the human submitting
  the change must review, understand, and disclose AI use in the PR description.
- Keep changes minimal and scoped to the request; mention, don't silently fix, unrelated
  problems you notice.

What does *not* carry over from MNE-Python:

- There are no lazy `__init__.pyi` stubs. The public API is the explicit import list in
  `mne_connectivity/__init__.py` (plus `mne_connectivity.decoding` and
  `mne_connectivity.viz`), and anything public must also be listed in `doc/api.rst` or
  Sphinx cross-references to it will not resolve.
- The docdict is this package's own (`mne_connectivity/utils/docs.py`, used via
  `from .utils import fill_doc`), *not* MNE's — MNE's entries are not visible here. Grep
  that file before writing a parameter description by hand.
- Deprecations do not use `@mne.utils.deprecated`. The pattern is `mne.utils.warn(...,
  FutureWarning)` at the top of the function plus a `.. version-deprecated:: X.Y` note in
  the docstring, saying what to use instead and naming the version that removes it (see
  `datasets/surrogate.py` and `datasets/frequency.py`).
- Tests use simulated data (`make_signals_in_freq_bands`, `make_surrogate_*`), not the MNE
  testing dataset; only `viz/tests/test_3d.py` needs the dataset.
- There is no `make ruff`; run `pre-commit run -a` (or `make pre-commit`). Most other
  Makefile targets are stale copies from MNE-Python and still reference `mne`.

## Project context

The next release is v1.0 and it is deliberately a breaking one — separating array-like from
MNE-object inputs, requiring precomputed spectra, reworking all-to-all vs. symmetric
indexing, moving Granger causality to dedicated functions, and dropping VAR-/epoch-specific
methods from containers that should never have had them. See
[#440](https://github.com/mne-tools/mne-connectivity/issues/440) and the "Upcoming breaking
changes" section of `doc/changes/v0.9.rst` before designing anything that touches those
areas.

## Verify before you trust this file

Most changes here are made by humans, and nothing keeps this file in sync with the code.
What it describes — especially in the two sections that follow — is meant to be the
*durable* shape of the package, but specific names, string options, and parameter sets do
change, and indexing in particular is actively being reworked. So treat this file as a map
of where to look, not as an API reference: grep for the name, read the current docstring or
docdict entry, and check the behavior in a REPL before you rely on it.

If something here turns out to be stale, update this file as part of your change and
mention it. If the drift is real but outside the scope of what you were asked to do, say so
in your summary rather than quietly working around it.

## Layout

- `base.py` — the connectivity containers: an `xarray.DataArray` of data plus mixins that
  add the epoch, frequency, and time dimensions. Storage layout, `get_data`, `save`,
  node renaming, and the VAR-model methods all live here.
- `spectral/` — the two spectral entry points. `spectral_connectivity_epochs` uses
  estimator classes that accumulate over epochs (bivariate and multivariate ones in
  separate modules); `spectral_connectivity_time` is a separate per-epoch implementation.
  The method registries and the lists that classify methods by capability sit next to the
  estimators.
- `effective.py`, `envelope.py`, `wsmi.py`, `vector_ar/` — the non-spectral estimators, one
  concern each. VAR model coefficients ride in a container and drive its `predict`/
  `simulate` methods.
- `decoding/` — scikit-learn-style estimators (`fit`/`transform`, trailing-underscore
  attributes, their own plotting methods).
- `datasets/` — simulated data with known ground truth; most tests are built on it.
- `io.py` — the netCDF round-trip. `viz/` — the plotting entry points (`plot_connectivity`
  for matrices, `plot_spectral_connectivity`/`plot_temporal_connectivity` for lines with
  circle-plot overviews, `plot_spectrotemporal_connectivity` for images) sharing one
  `helpers.py`, plus the older circle and 3D plots that are thin wrappers over MNE-Python.
- `utils/` — this package's docdict and the `indices` helpers.

## Things that bite

- **Connectivity data is stored raveled, not dense**, with the number of stored connections
  depending on how `indices` was specified; the container reshapes on request. `get_data()`
  defaults to a "compact" format whose shape therefore depends on how the object was
  constructed — ask for the format you actually want rather than inferring it.
- **`indices` is the central abstraction and the one most in flux.** It spans string forms
  (all-to-all, triangular) and explicit tuples of seed/target arrays, with different rules
  for bivariate vs. multivariate methods and for directed vs. undirected ones, and
  reconstructing a dense matrix from a partial one is not always possible. Read the current
  `indices` docdict entry and the helpers in `utils/` and route through them; do not
  type-sniff or hard-code the accepted values inline.
- **Multivariate connections are ragged** — each connection has its own number of seed and
  target channels — so they are padded into rectangular masked arrays with a sentinel fill
  value, including on disk. A sentinel must never reach fancy indexing, where it silently
  means "some real channel". `examples/handling_ragged_arrays.py` is the user-facing
  explanation.
- A multivariate "node" is a *set* of channels, not a channel, so dense output for
  multivariate data cannot be a plain per-channel matrix and the call returns extra
  information to map back. Code that assumes a bare array back breaks on multivariate input.
- **The two spectral entry points are independent implementations** with overlapping but
  unequal sets of supported methods; bivariate methods are written twice. Adding or changing
  a method means the registries, the per-method capability lists, the docdict entry, the
  method lists in both docstrings, and the parametrized tests — check whether it belongs in
  both before assuming it does.
- Multivariate methods carry machinery bivariate ones do not (rank reduction, multiple
  components, spatial patterns, model order), and not every multivariate method supports
  every piece: the capability lists next to the estimators are the source of truth, not the
  method name. Granger causality is the numerically fragile one — the solver raises on
  non-convergence, and reducing rank is usually the fix.
- Container state ends up as xarray `attrs` and must survive a netCDF round-trip: no `None`,
  no dicts, no nested structures — which is why some attributes are stored in a flattened
  or padded form. Add an attribute, add a save/read round-trip assertion.
- **Container attributes are handed out by reference.** `con.names` returns the list living
  in the xarray `attrs`, not a copy, so `names = con.names; names[idx] = ...` renames the
  nodes on the object the caller passed in — and on any list they built it from. Copy before
  you write.
- **`con.indices` is not one type.** Depending on how the object was built it comes back as
  `None`, a string, a tuple of 1-D integer arrays, a 2-D padded masked array, or a tuple of
  *lists* of variable-length arrays. Anything that indexes into it (`indices[0][picks]`) has
  to handle all of them, so normalize once at the entry point rather than type-sniffing later.
- **Channel-level `picks`/`exclude` do not map onto connections one-to-one**, because a
  connection has two endpoints. Say explicitly whether a connection survives when *any* or
  *all* of its endpoints do — `picks` usually means "any", `exclude` usually means "none" —
  and make the docstring and the code agree.
- MNE-Python is a moving target here: CI runs against both `mne` stable and `mne` main, and
  a few private MNE APIs are used. Prefer public API, and guard imports with `try`/`except
  ImportError` plus a `# TODO VERSION` comment when you can't.

## Tests

```bash
pytest mne_connectivity              # ~600 tests
pytest -n auto mne_connectivity      # what CI does; ~1.5 min on a fast laptop
pytest -n 0 --pdb mne_connectivity/spectral/tests/test_spectral.py -k cacoh
```

Warnings are errors (see `mne_connectivity/conftest.py`). The spectral tests dominate the
runtime, so `-n auto` is worth it, and adding another `@pytest.mark.parametrize` axis over
all methods is expensive — extend an existing test where you can.

Simulate rather than load: `make_signals_in_freq_bands` gives you two "regions" with a known
interaction in a known band, which is what most tests assert against. `viz/tests/test_3d.py`
needs the MNE testing dataset plus pyvistaqt and Qt (skipped otherwise); run Qt/PyVista tests
headless with `xvfb-run -a` or `QT_QPA_PLATFORM=offscreen`.

Interactive Matplotlib plots are driven with `_fake_click` / `_fake_keypress` /
`_fake_scroll` from `mne.viz.utils`, which work on any figure — these are plain Matplotlib
figures, not MNE-Python's browser `MNEFigure`, so there is no `fig._fake_*` method to reach
for. Call `fig.canvas.draw()` before faking events, or the coordinates go through
pre-layout transforms; clicking a line at its own data coordinates also fires `pick_event`.

Run `pre-commit run -a` before handing work back.

## Check the numbers, not just the shapes

This is a numerical package: a change can produce an array of exactly the right shape and
still be wrong. Assertions on `.shape` and `pytest.raises` are the cheap half of the work.
Before calling a change done, pin the values down against something you know independently:

- simulated data with a known interaction (`make_signals_in_freq_bands`) — the connectivity
  should peak in the simulated band and sit near the noise floor elsewhere;
- an equivalence that must hold — a multivariate method with one channel per seed/target
  against its bivariate counterpart, `gc_tr` against `gc` computed on time-reversed data,
  a directed method against itself with seeds and targets swapped, a symmetric method
  against its transpose, `spectral_connectivity_epochs` against
  `spectral_connectivity_time` on the same data (within tolerance);
- known bounds and identities — coherence in [0, 1], `imcoh` zero for zero-lag coupling,
  `dpli` at 0.5 for no preferred direction;
- a save/read round-trip through `read_connectivity` whenever containers are touched.

Keep exploratory scripts in scratch space, and promote whatever they caught into a real test.

## Look at it before you believe it

Plotting code can pass every assertion and still be wrong on screen. Before calling a visual
change done, render a representative figure, actually look at the PNG, *and* print the
artists' properties — the two catch different bugs. A squashed colorbar or an unreadable
pile of labels is obvious in an image and invisible to assertions; a connection drawn in the
colormap's "bad" color merely looks dark until you print `line.get_color()`.

```python
import matplotlib; matplotlib.use("agg")
fig, (line_ax, circle_ax) = plot_spectral_connectivity(con, show=False)
fig.canvas.draw()  # constrained layout: transforms are not final until this runs
fig.savefig("shot.png")  # then read the PNG
print([line.get_color() for line in line_ax.lines], line_ax.get_title())
```

When you change plotting code that already exists, do it as an A/B. Script a battery of
cases that saves one PNG per case — recording an exception as an outcome too, so the same
script runs against both versions — render it, `git checkout HEAD -- <the files you
touched>`, render into a second directory, restore, and compare the two pixel-wise. Most
cases should come back identical, which is what tells you a refactor really was a no-op, and
every difference should be one you can name and defend as a fix.

The docstring is the test plan: exercise every documented *form* of every parameter, not
just the common one — a shape the docstring promises but nothing ever plots (`(2,)`
alongside `(n, 2)`) is exactly where these functions break. Cover the degenerate ends too:
one connection, two nodes, a constant-valued array, all-negative data. For multivariate
results those are not edge cases but the normal shape of a seed-and-target analysis, and
because warnings are errors a `RuntimeWarning` out of a zero-width color range is a test
failure, not a cosmetic complaint.

Honor `show` through `mne.viz.utils.plt_show` rather than calling `plt.show()`, which warns
under the Agg backend the tests run on, and return the figure(s) you made.

## Changelog

User-facing changes need `doc/changes/dev/<PR-number>.<type>.rst` (types: `notable`,
`dependency`, `bugfix`, `apichange`, `newfeature`, `other`). One short sentence ending with
the contributor's name link, e.g. "Fix bug where X did Y, by `Jane Doe`_." — read a couple of
the entries in `doc/changes/v0.9.rst` first. The name anchor must exist in
`doc/changes/names.inc` (pre-commit keeps that file sorted). A CI job enforces the fragment;
the `no-changelog-entry-needed` label is the escape hatch.

The `<PR-number>` for a not-yet-opened PR is one more than the highest number currently in
use; issues and PRs share one sequence, so query the most recently created of either:

```bash
gh api "repos/mne-tools/mne-connectivity/issues?state=all&per_page=1&sort=created&direction=desc" \
    --jq '.[0].number'
```

New functionality should generally also show up in an example under `examples/` (built by
sphinx-gallery into the docs) for discoverability. Build the docs with `make -C doc html`,
or `make -C doc html-noplot` to skip running the examples.
