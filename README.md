# gabrielwendell.github.io

Static source for Gabriel Wendell's academic website. The site is served from
the repository root; no build step is required for the HTML pages.

## CNN time-series classification post

The post **1D versus 2D CNNs for Time-Series Classification** is available at
`blog/cnn_1d_vs_2d_time_series.html`. It is listed first in `blog.html`, appears
in the homepage news, and follows the finite-fields article in the post
navigation. Its original article date, November 21, 2026, is retained.

| Path | Purpose |
| --- | --- |
| `blog/cnn_1d_vs_2d_time_series.html` | Article, including the pretypeset SVG equations |
| `blog_assets/cnn_time_series/post.css` | Article-specific styles, loaded after the shared `style.css` |
| `blog_assets/cnn_time_series/*.svg` | Five figures used by the article; the existing PNG copies are also retained |
| `blog_assets/cnn_time_series/results.json` | Recorded configuration, timings, training histories, and scores |
| `python_codes/cnn_time_series/experiments.py` | Downloadable experiment script |
| `python_codes/cnn_time_series/requirements.txt` | Tested Python dependencies |
| `files/cnn_time_series/cnn_1d_vs_2d_blog_package.zip` | Complete companion package, including editable text, code, figures, and raw results |

The article uses the website's shared navigation, footer, fonts, and background.
Its mathematical content and numerical results are preserved from the approved
draft. The equations are pretypeset, so the article does not require a client
MathJax download. Figure and download URLs are relative to the repository root.

For full numerical reproduction, download and extract the companion ZIP and
follow its README. It contains the supporting scripts and per-seed predictions
in addition to the files exposed individually on the website.

## Preview locally

From the repository root, run:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/blog.html` and select the CNN article. Check the
contents links, full navigation, figures, and companion downloads. Local file
and anchor checks were completed during integration; a browser preview was
unavailable in the editing environment.
