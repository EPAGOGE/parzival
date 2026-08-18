# docs/ — the GitHub Pages site

Serves the parzival numerics note. Uses the **Organic** design system, shared with
the `papers` site: same `styles.css`, same Caprasimo/Figtree pairing, same tokens.
Separate project, separate content, one house style. The three study pages here cross-link in sequence; the papers site remains separate.

**To turn it on:** Settings → Pages → Deploy from a branch → `main`, folder `/docs`.

## The link preview

A bare `github.com/...` link renders GitHub's own card with the octocat. A Pages URL
serves this HTML, so the `og:*` tags in `index.html` control the preview instead.
Share the Pages URL when you want the branded card.

Absolute URLs are required in the OG tags. This build points at
`https://epagoge.github.io/parzival`. Find-and-replace that string in `index.html` if the
live base differs (e.g. `https://epagoge.github.io/parzival`). No trailing slash.

Previews cache hard. LinkedIn: Post Inspector. X: append `?v=2` once. Slack/Discord:
same cache-busting query.

## Rebuilding the share card

The card is real HTML in the design system, rendered headless so it uses the actual
fonts rather than a matplotlib approximation:

    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
      --headless --disable-gpu --hide-scrollbars --window-size=1200,630 \
      --screenshot="docs/og-card-1200x630.png" --virtual-time-budget=6000 \
      "file://$PWD/docs/_card/card.html"

## Files

| file | role |
|---|---|
| `index.html` | the page; OG and Twitter tags in `<head>` |
| `styles.css` | the Organic design system, shared with the papers site |
| `_card/card.html` | source for the share image |
| `og-card-1200x630.png` | the share image |
| `note.pdf` | the paper, copied from `note/note.pdf` |
| `.nojekyll` | serve files as-is |

When the note changes: `cp note/note.pdf docs/note.pdf`
