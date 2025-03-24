<div align="center">
<h1>Hann</h1>

[![Tests](https://img.shields.io/github/actions/workflow/status/habedi/hann/tests.yml?label=tests&style=flat&labelColor=282c34&logo=github)](https://github.com/habedi/hann/actions/workflows/tests.yml)
[![Lints](https://img.shields.io/github/actions/workflow/status/habedi/hann/lints.yml?label=lints&style=flat&labelColor=282c34&logo=github)](https://github.com/habedi/hann/actions/workflows/lints.yml)
[![Code Coverage](https://img.shields.io/codecov/c/github/habedi/hann?label=coverage&style=flat&labelColor=282c34&logo=codecov)](https://codecov.io/gh/habedi/hann)
[![CodeFactor](https://img.shields.io/codefactor/grade/github/habedi/hann?label=code%20quality&style=flat&labelColor=282c34&logo=codefactor)](https://www.codefactor.io/repository/github/habedi/hann)
[![Go Reference](https://img.shields.io/badge/reference-docs-3776ab?style=flat&labelColor=282c34&logo=go)](https://pkg.go.dev/github.com/habedi/hann)
[![License](https://img.shields.io/badge/license-MIT-00acc1?label=license&style=flat&labelColor=282c34&logo=open-source-initiative)](LICENSE)
[![Release](https://img.shields.io/github/release/habedi/hann.svg?label=release&style=flat&labelColor=282c34&logo=github&color=f06623)](https://github.com/habedi/hann/releases/latest)

A fast approximate nearest neighbor search library for Go

</div>

---

Hann is a high-performance approximate nearest neighbor search (ANN) library for Go.
It provides a collection of index data structures for efficient similarity search in high-dimensional spaces.

Hann can be seen as a core component of a vector database (e.g., Milvus, Weaviate, Qdrant, etc.).
It can be used to add fast in-memory similarity search capabilities to your Go applications.

### Features

- Unified interface for different index data structures (see [core/index.go](core/index.go))
- Support for indexing and searching vectors of arbitrary dimension
- Support for different distances like Euclidean, Manhattan, and cosine distances (see [core/distance.go](core/distance.go))
- Fast distance computation using SIMD (AVX) instructions (see [core/simd_distance.c](core/simd_distance.c))
- Support for bulk insertion, deletion, and update of vectors
- Support for saving indexes to disk and loading them back

#### Supported Indexes

| Index Name                                            | Space Complexity | Build Complexity | Search Complexity                             | 
|-------------------------------------------------------|------------------|------------------|-----------------------------------------------|
| [HNSW](https://arxiv.org/abs/1603.09320)              | $O(nd + nM)$     | $O(n\log n)$     | $O(\log n)$ average case<br>$O(n)$ worst case | 
| [PQIVF](https://ieeexplore.ieee.org/document/5432202) | $O(nk + kd)$     | $O(nki)$         | $O(\frac{n}{k})$                              | 
| [RPT](https://dl.acm.org/doi/10.1145/1374376.1374452) | $O(nd)$          | $O(n\log n)$     | $O(\log n)$ average case<br>$O(n)$ worst case | 

- $n$: number of vectors
- $d$: number of dimensions (vector length)
- $M$: links per node (in the HNSW graph)
- $k$: number of clusters (in PQIVF)
- $i$: iterations for clustering (in PQIVF)

#### Supported Distances

The indexes support the use of Euclidean, squared Euclidean, Manhattan, and cosine distances.
If cosine distance is used, the vectors are normalized before they are used in the index or for search.

### Installation

Hann can be installed as a typical Go module using the following command:

```bash
go get github.com/habedi/hann@main
```

*Hann requires Go 1.21 or later, and a CPU with AVX support.*

### Examples

| Example File                                 | Description                                                               |
|----------------------------------------------|---------------------------------------------------------------------------|
| [simple_hnsw.go](example/cmd/simple_hnsw.go) | Create and use an HNSW index with inline data                             |
| [hnsw.go](example/cmd/hnsw.go)               | Create and use an HNSW index                                              |
| [hnsw_large.go](example/cmd/hnsw_large.go)   | Create and use an HNSW index (using large datasets)                       |
| [pqivf.go](example/cmd/pqivf.go)             | Create and use a PQIVF index                                              |
| [pqivf_large.go](example/cmd/pqivf_large.go) | Create and use a PQIVF index (using large datasets)                       |
| [rpt.go](example/cmd/rpt.go)                 | Create and use an RPT index                                               |
| [rpt_large.go](example/cmd/rpt_large.go)     | Create and use an RPT index (using large datasets)                        |
| [load_data.go](example/load_data.go)         | Helper functions to load datasets for the examples                        |
| [utils.go](example/utils.go)                 | Extra helper functions for the examples                                   |
| [run_datasets.go](example/run_dataset.go)    | The code to create different indexes and try them with different datasets |

#### Datasets

You can download the datasets used in the examples using the following commands:

```shell
make download-data
```

```shell
# Only needed to run the examples that use large datasets
make download-data-large
```

Note that to run the examples using the large datasets, you may need a machine with large amounts of memory to build the
indexes.

See the [data](example/data) directory for information about the datasets.

### Documentation

The documentation for Hann can be found on [pkg.go.dev](https://pkg.go.dev/github.com/habedi/hann).

#### Logging

You can control the logging behavior of Hann using the `HANN_LOG` environment variable.

- `0`, `false`, or `off` to disable logging altogether
- `full` or `all` to enable full logging (`DEBUG` level)
- Any other value to enable basic logging (`INFO` level; default behavior)

#### Random Seed

You can make the indexes deterministic by setting the `HANN_SEED` environment variable to am integer value.
The value is used to initialize the random number generator used by the indexes.

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

### License

Hann is licensed under the MIT License ([LICENSE](LICENSE)).
