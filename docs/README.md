# Docs

A one-page site for the extension, built with the same Quarto template Inspect's
own docs use — `meridianlabs-ai/inspect-docs`, vendored in `_extensions/` at the
version `inspect_ai` itself pins. There is no separate Inspect house style; that
template is it.

Building needs Quarto plus a few Python packages the template's pre-render step
and reference filter import:

```bash
pip install pyyaml griffe panflute markdown rich
quarto preview docs
```

`_include.yml` and `reference/refs.json` are generated during the build and are
not checked in.

## Publishing

`.github/workflows/docs.yml` builds and deploys to GitHub Pages, but only when
run by hand from the Actions tab, and only once Pages is set to "GitHub Actions"
in the repository settings. Nothing publishes on a push.

Extensions here publish their own site and are then listed from a parent's docs:
Petri Bloom and Petri Dish each have a site plus a short page in
[Petri's docs](https://meridianlabs-ai.github.io/inspect_petri) pointing at it.
Inspect's own docs link out to nothing, so a listing there would mean asking the
maintainers for a section that does not yet exist.

Neither the site nor any listing PR should go out before the package is on PyPI
and the paper is on arXiv.
