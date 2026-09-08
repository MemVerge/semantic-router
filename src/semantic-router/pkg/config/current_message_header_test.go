package config

import (
	"testing"

	"gopkg.in/yaml.v2"
)

// The gate must survive the canonical import AND export: the management API
// and the k8s reconciler normalize every config write by round-tripping
// through CanonicalGlobalFromRouterConfig, so an export that drops the key
// silently turns the feature off on the next write (the same guard
// skip_processing carries).
func TestCanonicalRouterGlobalCurrentMessageHeaderRoundTrips(t *testing.T) {
	const yamlInput = `
router:
  current_message_header:
    enabled: true
services: {}
stores: {}
integrations: {}
model_catalog:
  embeddings: {}
  system: {}
  modules: {}
`
	var global CanonicalGlobal
	if err := yaml.Unmarshal([]byte(yamlInput), &global); err != nil {
		t.Fatalf("failed to unmarshal canonical global: %v", err)
	}
	if !global.Router.CurrentMessageHeader.IsEnabled() {
		t.Fatal("expected canonical global to surface current_message_header.enabled=true")
	}

	cfg := &RouterConfig{}
	if err := applyCanonicalGlobal(cfg, &global); err != nil {
		t.Fatalf("applyCanonicalGlobal failed: %v", err)
	}
	if !cfg.CurrentMessageHeader.IsEnabled() {
		t.Fatal("applyCanonicalGlobal should propagate current_message_header.enabled to runtime config")
	}

	exported := CanonicalGlobalFromRouterConfig(cfg)
	if exported == nil {
		t.Fatal("expected non-nil canonical global on export")
	}
	if !exported.Router.CurrentMessageHeader.IsEnabled() {
		t.Fatal("CanonicalGlobalFromRouterConfig should round-trip current_message_header.enabled=true")
	}
}
