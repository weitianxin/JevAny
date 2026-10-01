# JevAny homepage

The homepage is a static site, published at https://simplejev.org/JevAny/.
It has no runtime dependencies or third-party requests.

Preview from the repository root:

```bash
python3 -m http.server 4173 --bind 127.0.0.1 --directory site
```

Open http://127.0.0.1:4173. For browser checks:

```bash
python3 -m pip install -r site/tests/requirements.txt
python3 -m playwright install chromium
python3 site/tests/check_site.py
python3 site/tests/browser_checks.py
```

The homepage includes interactive benchmark results, all 30 archived replays,
released checkpoints and the supported-model catalog. Guides, the technical
report and linked reference files are hosted on the site. GitHub remains an
explicit source-code link; checkpoint download links point to Hugging Face.

Regenerate the homepage's data sections and documentation from the repository:

```bash
python3 -m pip install -r site/tools/requirements.txt
python3 site/tools/build_content.py
```

This reads `results/model-family-v2.json`, `docs/supported-models.json`, and the
existing Markdown guides. It updates the marked sections of `index.html`,
`assets/data`, `docs`, and `files`. Edit the source guides rather than generated
pages. GitHub Actions rebuilds and checks the site before publishing changes to
`main`, uploading only the public site files.

Thirty replay videos and the six scenes in the background come from
`docs/demos/cases`. The scenes are joined without gutters and blurred together,
with blended intermediate frames encoded into one 24 fps video. Landscape screens
load a 720×480 video; tall portrait screens load a 360×720 version. Only one background stream plays, and
the browser does not apply a live blur filter. Model-family logos scroll across
the top and pause outside the viewport. The motion toggle pauses both the logo strip and videos;
reduced-motion and data-saving preferences disable automatic playback.
Individual replays also have native video controls.
Each featured case plays twice before advancing to the next. The selected
case's highlight fills over both plays, following the video's actual progress.
Pausing or leaving the player preserves that progress. Selecting any case
disables automatic switching for the rest of the page visit and loops that
recording instead; its highlight stays full to mark the selection.

Regenerate media with `python3 site/tools/prepare_media.py` after installing
ffmpeg and Pillow. Use `--background-only` to regenerate just the two background
layouts. Foreground videos preserve the source timing; background scenes play
at half speed. The generator creates VP9 WebM and H.264
MP4 files with static WebP posters, including videos for the documentation.

Brand marks reuse the project's existing assets. Font licenses live in
`assets/fonts`; model-logo licenses and provenance live in `assets/model-logos`.
DM Sans and IBM Plex Sans Condensed are self-hosted. The latter uses IBM's
original outlines with a Latin character subset.

The prominent Star link opens the GitHub repository. Visitors confirm the star
on GitHub; the website does not request GitHub credentials or tokens.

Design references: [Supabase](https://supabase.com/) for the path from product
purpose to examples; [Ollama](https://ollama.com/) for a concise introduction and
quickstart; [vLLM](https://vllm.ai/) for visible model support and documentation.
The page uses JevAny's own brand and recordings.
