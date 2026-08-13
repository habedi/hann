# AGENTS.md

This file provides guidance to coding agents collaborating on this repository.

## Mission

Hann is an approximate nearest neighbor search library for Go.
It provides a set of index data structures (HNSW, PQIVF, and RPT) behind one interface, with distance computation written
in C and vectorized with AVX instructions.
Priorities, in order:

1. Correctness of index operations: insertion, deletion, update, and search must keep the index consistent.
2. Search quality and speed, measured by recall and query latency on the example datasets.
3. Clean separation between the shared interface and helpers (`core/`) and the index implementations (`hnsw/`, `pqivf/`, and `rpt/`).
4. Safety of the cgo boundary: no out-of-bounds reads, and no pointers into Go memory that outlive the call.

## Core Rules

- Use English for code, comments, docs, and tests.
- Prefer small, focused changes over large refactoring.
- Add comments only when they clarify non-obvious behavior.
- Do not add features, error handling, or abstractions beyond what is needed for the current task.
- Keep external dependencies minimal: do not add new `go.mod` entries without prior discussion.

## Backward Compatibility

Hann is a public Go module that other programs import. The following must stay backward-compatible:

- The `core.Index` interface. Adding a method breaks every implementation outside this repository, so a new capability belongs on the concrete index
  types, or on a separate optional interface that callers can assert.
- Exported types and constructor signatures (`hnsw.NewHNSW`, `pqivf.NewPQIVFIndex`, and `rpt.NewRPTIndex`). New tuning parameters belong in an options
  struct or a setter, not in a longer parameter list.
- The shapes of `core.Neighbor` and `core.IndexStats`: fields may be added, not removed or renamed.
- The gob encoding written by `Save`. An index file written by an older version must still load. The `serializedIndex`, `serializedPQIVF`, and
  `rptSerialized` structs are the on-disk format, so fields may be added with sensible zero values, but they may not be removed, renamed, or reordered
  in meaning.
- The keys in `core.Distances` and the names reported by `IndexStats.Distance`.
- The environment variables `HANN_LOG`, `HANN_SEED`, and `HANN_BENCH_NTRD`, along with the values they accept.
- The minimum Go version declared in `go.mod`. Raising it drops users, so it is a deliberate decision, not a side effect of using a newer standard
  library function.

## Writing Style

- Use Oxford commas in inline lists: "a, b, and c" not "a, b, c".
- Do not use em dashes, in documentation or in code comments. Restructure the sentence, or use a colon or semicolon instead.
- Avoid colorful adjectives and adverbs. Write "rate limiter" not "smart rate limiter".
- Prefer noun phrases for checklist items over imperative verbs. Write "rate limit enforcement" not "enforce rate limits".
- Headings in Markdown files must be in title case: "Build from Source" not "Build from source". Minor words stay lowercase unless they are the first
  word: the articles (a, an, the), the coordinating conjunctions (and, but, or, nor, so, yet, for), and the short prepositions (in, on, at, to, by,
  of, up, as, from, with, into, over). The prepositions are named because "from" has to be lowercase for "Build from Source" to be correct.
- Do not bold the lead-in of a list item. Write "Unit tests: ..." not "**Unit tests**: ...".
- Use sentence case for the lead-in of a list item. Write "Seed selection: ..." not "Seed Selection: ...". Proper nouns keep their capitals.
- Capitalize only the first part of a hyphenated compound: "Nearest-neighbor Search" in a heading, "Nearest-neighbor" at the start of a sentence, and
  "nearest-neighbor search" elsewhere. Never write "Nearest-Neighbor".
- Start each sentence with a capital letter, capitalize proper nouns (Go, AVX, SIMD, HNSW, PQIVF, RPT), and leave common nouns lowercase in the middle
  of a sentence.
- Write correct and complete sentences.
- Avoid made-up words.
- Do not use a colon in place of a verb. Three uses are fine: joining two clauses inside a complete sentence (the replacement the em-dash rule above
  calls for), introducing the gloss of a list item, and introducing an enumeration, whether as a list or inline ("Targets: `make test`,
  `make lint`, ..."). What a colon must not do is turn a sentence into a label and a definition: write "Splits a vector into subspaces, then quantizes
  each one" rather than "Product quantization: splits a vector into subspaces". That shape belongs to a list item, and carrying it into prose (a doc
  comment summary, a paragraph) leaves a fragment where a sentence was required.
- Use participial phrases and abbreviations scarcely.

## Repository Layout

- `core/`: the shared interface and helpers. `index.go` declares `Index`, `Neighbor`, and `IndexStats`; `distance.go` wraps the C distance functions
  and exposes the `Distances` map; `vector_ops.go` holds normalization, single and batched; `cpu_check.go` detects AVX and AVX2 at startup and tells
  the C side which implementation to install; `log_config.go` reads `HANN_LOG`; `utils.go` reads `HANN_SEED`.
- `core/*.c` and `core/*.h`: the C implementations. `simd_distance.c` holds Euclidean, squared Euclidean, Manhattan, and cosine distance; `simd_ops.c`
  holds normalization and the `hann_cpu_init` entry point. Each function has a fallback, an AVX, and an AVX2 variant, selected once through a function
  pointer.
- `hnsw/index.go`: the HNSW graph index, its layered neighbor lists, and its gob codec.
- `pqivf/index.go`: the PQIVF index, coarse clustering, product quantization, and `Train`.
- `rpt/index.go`: the RPT index and its random projection tree.
- `example/`: dataset loading (`load_data.go`), shared helpers and recall computation (`utils.go`), and the index runners (`run_datasets.go`).
- `example/cmd/`: one `main` per example and per benchmark, run with `go run`. This directory is excluded from the test and lint targets.
- `example/data/`: the dataset download script and notes on the datasets.
- `.github/workflows/`: CI workflows for tests and lints.
- `Makefile`: all developer tasks (format, test, lint, examples, benchmarks, and datasets).

## Architecture

### Layers

Hann is organized into three layers that should not have upward dependencies:

1. `core/`: the interface, distance functions, and vector operations; no knowledge of any index.
2. `hnsw/`, `pqivf/`, and `rpt/`: index implementations; each depends on `core/` and on nothing else in the repository, and the three never import
   each other.
3. `example/` and `example/cmd/`: programs that exercise the indexes; nothing imports them.

### Boundaries Worth Keeping

- An index is reached through `core.Index`. Code that is written against a concrete type gives up the ability to swap indexes, so keep example and
  benchmark code on the interface where the operation is part of it. `pqivf.Train` is the exception, because it has no counterpart in the other
  indexes.
- The distance function is passed into `NewHNSW` along with its name. The name is what `Stats` reports and what the gob codec stores, so the two must
  agree; do not derive one from the other by comparing function values.
- Each index owns a mutex and is safe for concurrent use through its exported methods. The unexported helpers assume the lock is already held. Keep
  that split: a helper must not take the lock itself, and an exported method must not call another exported method of the same index while holding it.
- Random behavior goes through the package-level `seededRand`, guarded by `seededRandMu`, so a run with `HANN_SEED` set is reproducible. Do not call
  the global `math/rand` functions, and do not create a new generator per operation.
- Logging goes through `zerolog`, configured once in `core/log_config.go`. A library must stay quiet by default, so new log lines belong at debug
  level unless they report a condition the caller cannot see.

### The cgo Boundary

`core/distance.go` and `core/vector_ops.go` pass `&slice[0]` into C together with the length. The rules that keep this sound:

- Check for an empty slice before taking the address of its first element, and check that both operands have the same length before the call. The C
  side trusts the length it is given.
- Do not retain a Go pointer on the C side. Every call reads or writes the vector and returns.
- A new distance function needs the fallback, AVX, and AVX2 variants, an entry in the function pointer table in `init_distance_functions`, a
  declaration in the header, and an entry in `core.Distances`.
- The AVX and AVX2 variants are compiled behind `#ifdef __AVX__` and `#ifdef __AVX2__`, and selected at runtime by `hann_cpu_init`. A machine without
  AVX must still build and run through the fallback path.
- `NormalizeBatch` fans out over a worker pool, so each worker must own its own vector. Do not share a slice between tasks.

### Persistence

`Save` and `Load` are gob over an `io.Writer` and an `io.Reader`. Each index has a serialized form (a plain struct of exported fields) and a
`GobEncode`/`GobDecode` pair that converts between it and the live index, which is what lets pointer-linked structures such as the HNSW graph and the
RPT tree round-trip. Types are registered in each package's `init`. A change to a serialized struct is a change to the on-disk format, so read the
backward compatibility section above before making one.

## Go Conventions

- Go version: the minimum is declared in `go.mod`, and CI runs the test suite against every release from that version onward.
- A C compiler is needed, because `core/` uses cgo. `CGO_ENABLED=0` does not produce a working build.
- Formatting is enforced by `gofmt` (via `make format`). Run it before committing.
- Naming follows Go standard conventions: `PascalCase` for exported identifiers, `camelCase` for unexported identifiers and local variables, and
  `SCREAMING_SNAKE_CASE` for top-level constants where idiomatic.
- Errors are returned, never logged and swallowed. Wrap with context using `fmt.Errorf("…: %w", err)` so callers can use `errors.Is`/`errors.As`.
- Use `zerolog` (already imported as `github.com/rs/zerolog/log`) for all logging; do not use `fmt.Print*` for diagnostic output.
- Every exported identifier carries a doc comment that starts with its name.

## Required Validation

Run the relevant targets for any change:

| Target          | Command                    | What It Runs                                           |
|-----------------|----------------------------|--------------------------------------------------------|
| Format          | `make format`              | `go fmt ./...`                                         |
| Unit tests      | `make test`                | `go test` with coverage and the race detector          |
| Lint            | `make lint`                | `golangci-lint run ./...`                              |
| Coverage report | `make showcov`             | Displays per-function coverage after running the tests |
| Examples        | `make run-examples`        | Runs the examples that use the small datasets          |
| Large examples  | `make run-examples-large`  | Runs the examples that use the large datasets          |
| Benchmarks      | `make run-benches`         | Runs the local benchmarks                              |
| Datasets        | `make download-data`       | Downloads the datasets the examples use                |
| Large datasets  | `make download-data-large` | Downloads the large datasets                           |
| Git hooks       | `make setup-hooks`         | Installs the pre-commit and pre-push hooks             |

The examples and the benchmarks need the datasets, so run `make download-data` first. The large variants need a machine with a lot of memory (32 GB or
more).

## First Contribution Flow

1. Read `core/index.go` to see the contract, then the index package the change touches.
2. Add or update `_test.go` files in the changed package to describe the new behavior.
3. Run `make test` and watch the new test fail.
4. Implement the smallest change that makes it pass.
5. Run `make test` and `make lint` again, then refactor with the tests green.
6. If the change touches search, distance computation, or serialization, also run `make run-examples` and check that the reported recall has not
   dropped.

Good first tasks:

- New unit test for an existing untested helper in `core/`.
- Error message refinement in an index package, paired with a test that asserts the returned error.
- New `make` target or script improvement in `Makefile` or `example/data/download_datasets.sh`.
- A doc comment for an exported identifier that lacks one.

## Testing Expectations

Follow a red-green cycle. Write the test first, run it, and see it fail for the reason you expect; a test that passes before the change is not testing
the change. Then write the smallest change that makes it pass, and refactor with the test green. A bug fix starts with a test that reproduces the bug,
so the failure is captured before it disappears.

- Unit tests live in `_test.go` files alongside the package they cover. The index packages are tested from outside (`package hnsw_test`), so the tests
  exercise the exported surface the same way a user does.
- Every new exported function or behavior change must ship with at least one test that exercises it, including error paths where applicable.
- Test the interface, not the internals. An index test asserts on `Search` results, on `Stats`, and on returned errors, not on the shape of the graph
  or the tree.
- Cover the four operations that can leave an index inconsistent: delete, bulk delete, update, and bulk update. Each must be followed by a search that
  shows the removed ids are gone and the surviving ids are still reachable.
- Serialization round-trips are tested by saving to a buffer or a `t.TempDir()` file, loading into a fresh index, and comparing search results with
  the original. Do not write into the repository.
- Set `HANN_SEED` in a test that depends on the outcome of a random choice, and use the returned error rather than an exact distance value where the
  result is approximate.
- Run the race detector on anything that touches goroutines, which includes bulk operations, parallel search, and `NormalizeBatch`:
  `go test -race -count=1` on the affected packages, more than once.
- A test must not depend on the example datasets, because they are downloaded separately and are not present in CI.
- Tests for the AVX paths must still pass on a machine without AVX, so assert on distance values with a tolerance rather than on bit-exact equality.

## Change Design Checklist

Before coding:

1. Packages affected by the change (`core`, `hnsw`, `pqivf`, `rpt`, or `example`).
2. Whether the change alters the `core.Index` interface, an exported signature, or the gob format.
3. Whether the change touches C code, and if so, whether every one of the fallback, AVX, and AVX2 paths was updated.
4. Whether a new external dependency is required, and if so, whether it has been discussed.
5. Whether the change affects recall or query latency, and how that will be measured.

Before submitting:

1. `make format` passes (no diff).
2. `make test` passes with the race detector enabled.
3. `make lint` passes.
4. The examples run locally if the change touches search, distance computation, or serialization.

## Commit and PR Hygiene

- Keep commits scoped to one logical change.
- PR descriptions should include:

1. Behavioral change summary.
2. Tests added or updated.
3. Whether the examples or the benchmarks were run locally (yes/no), and on which CPU, since the SIMD path taken depends on it.
