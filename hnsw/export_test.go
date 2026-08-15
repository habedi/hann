package hnsw

// Test hooks. The tests live in package hnsw_test, so the internals they
// need are re-exported here for the test binary only.
var LevelFromUnit = levelFromUnit

// MaxLevelCap mirrors maxLevelCap for the level tests.
const MaxLevelCap = maxLevelCap
