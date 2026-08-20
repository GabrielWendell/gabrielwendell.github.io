# Topological Representation of Stellar Spectra

Ready-to-publish package for Gabriel Wendell's website.

## Contents

- `blog/topological-representation-stellar-spectra.html` — complete article, matched to the existing post template.
- `assets/stellar-spectrum-topology-overview.png` — synthetic spectrum and persistence diagram.
- `assets/stellar-spectrum-sublevel-filtration.png` — visual explanation of the filtration.
- `assets/stellar-spectrum-broadening-comparison.png` — narrow/broadened comparison and robust Betti curves.
- `code/topological_stellar_spectra.py` — complete reproducibility script.
- `blog-index-card.html` — card to paste into the `card-grid` in `blog.html`.
- `requirements.txt` — minimal Python dependencies.

## Publish on the existing site

From the website repository root, copy the package contents into the matching
folders:

```text
blog/topological-representation-stellar-spectra.html  -> blog/
assets/*.png                                          -> assets/
code/topological_stellar_spectra.py                   -> code/
```

Then paste the article in `blog-index-card.html` inside the `card-grid` element
of `blog.html`, above the introductory post if it should appear as the newest
entry.

The post uses the site's existing `style.css`, MathJax, Prism.js, and relative
navigation paths. No edit to `style.css` is required; post-specific styles are
scoped inside the article file.

## Reproduce the figures

Create or activate a Python environment and run:

```bash
python -m pip install -r requirements.txt
python code/topological_stellar_spectra.py
```

The script overwrites the three PNG files in `assets/` and prints the numerical
metrics quoted in the article.

## Methodological note

The synthetic line list and Gaussian broadening kernel are pedagogical. They
are designed to demonstrate lower-star degree-zero persistence, not to replace
a stellar-atmosphere or radiative-transfer synthesis.
