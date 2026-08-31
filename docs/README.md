# Docs

`logittilt.qmd` is a draft documentation page for the extension, written to be
framework-neutral: it is about surfacing behaviours in fewer samples than
repeated sampling needs, which applies to any eval, not only auditing.

Where it should go is undecided. Inspect's own docs have no slot for a
third-party extension: `extensions.qmd` documents the extension *types*, and
every provider in `providers.qmd` ships inside `inspect_ai` itself. So placing
it there means either asking the maintainers for a community-extensions
section, or upstreaming the provider.

[Petri](https://github.com/meridianlabs-ai/inspect_petri) does have that slot,
at `docs/extensions/`, alongside the `petri-bloom` and `petri-dish` pages for
other external packages.

Either way, not before this package is on PyPI: the page says `pip install
inspect-logittilt`.
