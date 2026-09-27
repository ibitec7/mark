# MaRK: Markov-adapted Recurrent Kernels

Project website for **MaRK: Markov-adapted Recurrent Kernels for Dynamic Operator Conditioning in State Space Models**.

**Live site: <https://ibitec7.github.io/mark-page/>**

## Links

- **Paper** (OpenReview): <https://openreview.net/pdf?id=yIjFWwvG7b>
- **Code**: <https://github.com/ibitec7/mark>
- **arXiv**: <https://arxiv.org/abs/2601.22157> (coming soon)

## About

MaRK (Markov-adapted Recurrent Kernels) is a dynamic operator-conditioning framework that maps context
vectors directly into bounded modulations of a frozen SSM's recurrence (A), read-in (B), read-out (C),
skip (D), and discretization parameters. Viewed through the lens of Linear Parameter-Varying systems,
MaRK induces a context-indexed family of Markov parameter sequences, allowing each diffusion timestep to
reshape the model's input-output memory kernel. The adapters are provably stable, the operator is
identifiable, and the framework is evaluated on diffusion language models.

## Pages

- `index.html` — the entire site (content lives here)
- `static/css/index.css` — styling
- `static/js/index.js` — galleries, LaTeX rendering, copy-to-clipboard
- `static/images/` — figures, favicons, social preview
- `static/vendor/katex/` — self-hosted KaTeX (equations render with no external dependency)

## Running locally

The site is plain static files with no build step. Serve the repository root:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

## Deployment

Published with [GitHub Pages](https://pages.github.com/) from the `main` branch (repository root).
Any commit pushed to `main` is redeployed automatically; no build step is required.

## Acknowledgments

Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page and the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).

## Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
