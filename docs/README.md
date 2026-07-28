# docs/ — the GitHub Pages site

Serves <https://epagoge.github.io/parzival/>.

**To turn it on (one time):** repo Settings → Pages → Source: *Deploy from a branch*
→ Branch: `main`, Folder: `/docs` → Save. First build takes a minute or two.

## Why the link preview shows EPAGOGE and not the octocat

A bare `github.com/...` link is rendered by GitHub's own card (octocat, repo name).
A **Pages URL** is your HTML, so the `og:*` meta tags in `index.html` control the
preview: title, description, and `og-card.png` (1200×630). Share the Pages URL, not
the repo URL, when you want the branded card.

After changing `og-card.png`, social platforms cache aggressively. Force a refresh:
- Slack/iMessage: append `?v=2` to the URL once
- X: <https://cards-dev.twitter.com/validator>
- LinkedIn: <https://www.linkedin.com/post-inspector/>
- Facebook/Meta: <https://developers.facebook.com/tools/debug/>

## Files

| file | role |
|---|---|
| `index.html` | landing page; all `og:*` / `twitter:*` tags live in `<head>` |
| `og-card.png` | 1200×630 preview image (built by `/tmp/mkcard.py`, archived in `repro/scripts/`) |
| `favicon.png` | tab icon |
| `note.pdf` | the paper — copy of `note/note.pdf` |
| `fig2_free_residual.png` | the figure shown inline |
| `.nojekyll` | tells Pages to serve files as-is, no Jekyll processing |

**When the note changes**, refresh the copies:

    cp note/note.pdf docs/note.pdf
    cp note/fig/fig2_free_residual.png docs/
