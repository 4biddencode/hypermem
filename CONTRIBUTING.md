# Contributing

HyperMEM is source-available. Pull requests and issues are welcome.

## Setup

```bash
git clone https://github.com/4biddencode/hypermem.git
cd hypermem
pip install -e ".[test]"
```

## Running the tests

```bash
python -m pytest tests -q
```

All tests are hermetic — they stub the LLM transport, so no model or network
is needed. Live-model tests (marked `live`) are skipped by default.

## What makes a good PR

- Keep the public API stable. The public surface is small on purpose.
- Add a test for any behavior change. Tests are the spec.
- Run the full suite before opening the PR.

## License

See [LICENSE](LICENSE). Contributing to the project means your contribution is
licensed under the same terms.