#!/usr/bin/env node
/**
 * Pre-renders the LaTeX in index.html to static KaTeX HTML.
 *
 * The page ships the rendered markup, so visitors never download or execute
 * katex.min.js (277 KB) and the maths is painted with the rest of the document
 * instead of appearing a moment later. The original LaTeX is kept in each
 * element's data-tex attribute, which makes this script idempotent: edit the
 * LaTeX (in the attribute, or write new \( ... \) in the body) and re-run
 *
 *     node tools/render-math.mjs
 *
 * The stylesheet (static/vendor/katex/katex.min.css) and its webfonts are still
 * served to the page; only the runtime is gone.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const katex = require('../static/vendor/katex/katex.min.js');
const file = new URL('../index.html', import.meta.url);

const OPTS = { throwOnError: false, errorColor: '#cc0000' };
const OPEN = '<span class="katex-math" data-tex="';
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const unesc = (s) => s.replace(/&quot;/g, '"').replace(/&gt;/g, '>').replace(/&lt;/g, '<').replace(/&amp;/g, '&');
const render = (tex) => `${OPEN}${esc(tex)}">${katex.renderToString(tex, OPTS)}</span>`;

/** Re-render every element this script already emitted. The rendered markup
 *  contains nested <span>s, so the wrapper's end is found by counting them. */
function rerender(html) {
  let out = '';
  let i = 0;
  let count = 0;
  for (;;) {
    const start = html.indexOf(OPEN, i);
    if (start === -1) {
      out += html.slice(i);
      return [out, count];
    }
    const attrEnd = html.indexOf('">', start + OPEN.length);
    if (attrEnd === -1) throw new Error('malformed data-tex attribute');
    const tex = unesc(html.slice(start + OPEN.length, attrEnd));
    out += html.slice(i, start);

    let depth = 0;
    let j = start;
    let end = -1;
    while (j < html.length) {
      const nextOpen = html.indexOf('<span', j);
      const nextClose = html.indexOf('</span>', j);
      if (nextClose === -1) break;
      if (nextOpen !== -1 && nextOpen < nextClose) {
        depth += 1;
        j = nextOpen + 5;
      } else {
        depth -= 1;
        j = nextClose + 7;
        if (depth === 0) {
          end = nextClose;
          break;
        }
      }
    }
    if (end === -1) throw new Error('unbalanced span in rendered maths');
    out += render(tex);
    count += 1;
    i = end + 7;
  }
}

let html = readFileSync(file, 'utf8');
let rendered = 0;

[html, rendered] = rerender(html);

const before = html;
html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_m, tex) => {
  rendered += 1;
  return render(tex.trim());
});
html = html.replace(/\\\(([\s\S]+?)\\\)/g, (_m, tex) => {
  rendered += 1;
  return render(tex.trim());
});

if (html !== before) writeFileSync(file, html);
console.log(`rendered ${rendered} expression(s)`);
