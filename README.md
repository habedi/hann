## Hann

[![Tests](https://img.shields.io/github/actions/workflow/status/habedi/hann/tests.yml?label=tests&style=flat&labelColor=555555&logo=github)](https://github.com/habedi/hann/actions/workflows/tests.yml)
[![Lints](https://img.shields.io/github/actions/workflow/status/habedi/hann/lints.yml?label=lints&style=flat&labelColor=555555&logo=github)](https://github.com/habedi/hann/actions/workflows/lints.yml)
[![Code Coverage](https://img.shields.io/codecov/c/github/habedi/hann?label=coverage&style=flat&labelColor=555555&logo=codecov)](https://codecov.io/gh/habedi/hann)
[![CodeFactor](https://img.shields.io/codefactor/grade/github/habedi/hann?label=code%20quality&style=flat&labelColor=555555&logo=codefactor)](https://www.codefactor.io/repository/github/habedi/hann)
[![Go Version](https://img.shields.io/github/go-mod/go-version/habedi/hann?label=go%20version&style=flat&labelColor=555555&logo=go)](go.mod)
[![Go Reference](https://img.shields.io/badge/Go%20Reference-Docs-3776ab?label=reference&style=flat&labelColor=555555&logo=go)](https://pkg.go.dev/github.com/habedi/hann)
[![License](https://img.shields.io/badge/license-MIT-007ec6?label=license&style=flat&labelColor=555555&logo=open-source-initiative)](LICENSE)
[![Release](https://img.shields.io/github/release/habedi/hann.svg?label=release&style=flat&labelColor=555555&logo=github)](https://github.com/habedi/hann/releases/latest)

Hann is a high-performance approximate nearest neighbor search (ANN) library for Go.
It provides a collection of index data structures for efficient similarity search in high-dimensional spaces.

### Supported Indexes

- [Hierarchical Navigable Small World Graph (HNSW)](https://arxiv.org/abs/1603.09320)
- [Product Quantization with Inverted File (PQIVF)](https://ieeexplore.ieee.org/document/5432202)
- [Random Projection Tree (RPT)](https://dl.acm.org/doi/10.1145/1374376.1374452)

### Examples

Check out the [examples](examples) directory for usage examples.

| Index | Example File                              | Description                                                              |
|-------|-------------------------------------------|--------------------------------------------------------------------------|
| 1     | [hnsw.go](examples/cmd/hnsw.go)           | Create and use an HNSW index: insert, delete, update, and search vectors |
| 2     | [pqivf.go](examples/cmd/pqivf.go)         | Create and use a PQIVF index: insert, delete, update, and search vectors |
| 3     | [rpt.go](examples/cmd/rpt.go)             | Create and use an RPT index: insert, delete, update, and search vectors  |
| 4     | [load_data.go](examples/cmd/load_data.go) | Helper functions to load datasets for the examples                       |

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

### License

Hann is licensed under the MIT License ([LICENSE](LICENSE)).
