# MaRK: Markov-adapted Recurrent Kernels

Project website for **MaRK: Markov-adapted Recurrent Kernels for Dynamic Operator Conditioning in State Space Models**.

**Live site: <https://ibitec7.github.io/mark/>**

## Links

- **Paper**: <https://arxiv.org/pdf/2610.09092>
- **Abstract**: <https://arxiv.org/abs/2610.09092>
- **Code**: <https://github.com/ibitec7/mark>

## About

MaRK (Markov-adapted Recurrent Kernels) is a dynamic operator-conditioning framework that maps context
vectors directly into bounded modulations of a frozen SSM's recurrence (A), read-in (B), read-out (C),
skip (D), and discretization parameters. Viewed through the lens of Linear Parameter-Varying systems,
MaRK induces a context-indexed family of Markov parameter sequences, allowing each diffusion timestep to
reshape the model's input-output memory kernel. The adapters are provably stable, the operator is
identifiable, and the framework is evaluated on diffusion language models.

## Repository layout

- `index.html` — the entire site (content lives here)
- `static/css/index.css` — styling
- `static/fonts/inter-latin.woff2` — self-hosted Inter (variable, latin subset)
- `static/images/` — figures, favicons, social preview
- `static/js/index.js` — galleries and copy-to-clipboard
- `static/vendor/katex/` — self-hosted KaTeX
- `tools/render-math.mjs` — pre-renders the LaTeX in `index.html`

## Equations

The maths is rendered **at build time**, not in the visitor's browser: `tools/render-math.mjs` turns each
expression into static KaTeX markup and stores the original LaTeX in a `data-tex` attribute. The page
therefore ships no LaTeX runtime, and the equations paint with the rest of the document.

After editing any equation (either in its `data-tex` attribute or by writing new `\\( ... \\)` in the body),
regenerate the markup and commit the result:

```bash
node tools/render-math.mjs
```

## Running locally

The site is plain static files with no build step. Serve the repository root:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

## Performance

The site is deliberately self-contained and small:

- **No third-party requests.** Fonts (Inter), icons (inline SVG) and KaTeX are all served locally, so
  there is no DNS/TLS round trip to a font or icon CDN before the first paint.
- **~650 KB for a first visit** (gzip + WebP), down from ~2.4 MB, with the largest figures re-encoded as
  256-colour PNGs for line art and WebP for shaded plots.
- **Explicit `width`/`height` on every figure** so the layout cannot shift while images arrive, plus
  `loading="lazy"` for everything below the fold and `fetchpriority="high"` on the teaser.
- **No icon webfont.** Five icons are inlined as SVG.

## Deployment

Published with [GitHub Pages](https://pages.github.com/) from the `gh-pages` branch of
<https://github.com/ibitec7/mark> (repository root), which serves the site at
<https://ibitec7.github.io/mark/>. Any commit pushed to that branch is redeployed automatically; no build
step is required.

The previous address, <https://ibitec7.github.io/mark-page/>, redirects here.

## Acknowledgments

Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page and the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).

## Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
