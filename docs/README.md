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

Live at <https://adrskapars.github.io/inspect_logittilt/>.

`.github/workflows/docs.yml` rebuilds and deploys, but only when run by hand
from the Actions tab — nothing publishes on a push.

## Getting listed on the Inspect site

Inspect keeps a directory of third-party extensions at
<https://inspect.aisi.org.uk/extensions/> — note the trailing slash; the page at
`/extensions.html` is a different one about the extension *types*, and the
listing renders client-side, so fetching the HTML as text shows nothing.

Entries live in `docs/extensions/extensions.yml` in `UKGovernmentBEIS/inspect_ai`.
`extensions.json` is generated from that YAML at build time, so do not edit it.
Their CONTRIBUTING calls this "a one-line PR". Ours, agreed and ready to submit:

```yaml
- name: "[Inspect LogitTilt](https://adrskapars.github.io/inspect_logittilt/)"
  description: |
    Steers a model's sampling towards any target behaviour, surfacing on-policy
    examples in fewer samples, accessing only the model's logits.
  author: "[Adrians Skapars](https://scholar.google.com/citations?user=9Vf0mWkAAAAJ&hl=en)"
  categories: ["Tooling"]
```

Waiting on the arXiv announcement, so the docs site the entry points at can cite
the paper. The blog post follows once both are done.
