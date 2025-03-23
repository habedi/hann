## Datasets

You can download the nearest neighbor datasets used in the examples from Hugging Face:

- [nearest-neighbors-datasets](https://huggingface.co/datasets/habedi/nearest-neighbors-datasets)
- [nearest-neighbors-datasets-large](https://huggingface.co/datasets/habedi/nearest-neighbors-datasets-large)

Make sure that the downloaded datasets are stored in the `example/data` directory.

### Using Hugging Face CLI Client

You can use [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) to download the datasets.

```shell
huggingface-cli download habedi/nearest-neighbors-datasets --repo-type dataset \
 --local-dir nearest-neighbors-datasets
```

```shell
huggingface-cli download habedi/nearest-neighbors-datasets-large --repo-type dataset \
 --local-dir nearest-neighbors-datasets-large
```

The command must be run inside this directory (`example/data`).

For convenience, you can use the [`pyproject.toml`](../../pyproject.toml) file to set up a Python environment with the
required dependencies using [Poetry](https://python-poetry.org/).
