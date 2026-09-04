package config

import (
	"testing"

	"gopkg.in/yaml.v3"
)

func TestBatchClassificationConfigSignalBatchingEnabled(t *testing.T) {
	var cfg struct {
		Batch BatchClassificationConfig `yaml:"batch"`
	}
	if err := yaml.Unmarshal([]byte("batch:\n  signal_batching_enabled: true\n"), &cfg); err != nil {
		t.Fatalf("unmarshal config: %v", err)
	}
	if !cfg.Batch.SignalBatchingEnabled {
		t.Fatal("signal_batching_enabled was not parsed")
	}
}
