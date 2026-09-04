package classification

import (
	"fmt"
	"sync"
	"testing"
	"time"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSignalBatchCollectorFlushesPartialBatchOnceFromFirstArrival(t *testing.T) {
	batchSizes := make(chan int, 2)
	collector := newSignalBatchCollector(func(inputs []signalEvaluationInput) []*SignalResults {
		batchSizes <- len(inputs)
		return signalBatchTestResults(inputs)
	})

	first := collector.enqueue(signalEvaluationInput{text: "first"})
	collector.mu.Lock()
	firstTimer := collector.timer
	collector.mu.Unlock()
	second := collector.enqueue(signalEvaluationInput{text: "second"})
	collector.mu.Lock()
	secondTimer := collector.timer
	collector.mu.Unlock()
	if firstTimer != secondTimer {
		t.Fatal("later arrival reset the batch timer")
	}

	for i, resultCh := range []<-chan *SignalResults{first, second} {
		select {
		case result := <-resultCh:
			want := []string{"first", "second"}[i]
			if got := result.SignalValues["row"]; got != signalBatchTestMarker(want) {
				t.Fatalf("result %d row marker = %v, want %v", i, got, signalBatchTestMarker(want))
			}
		case <-time.After(time.Second):
			t.Fatalf("timed out waiting for partial-batch result %d", i)
		}
	}
	if got := <-batchSizes; got != 2 {
		t.Fatalf("batch size = %d, want 2", got)
	}
	time.Sleep(signalBatchWindow + 20*time.Millisecond)
	select {
	case got := <-batchSizes:
		t.Fatalf("partial batch evaluated more than once; extra size %d", got)
	default:
	}
}

func TestSignalBatchCollectorFlushesThirtyTwoImmediatelyAndKeepsNextGeneration(t *testing.T) {
	batchSizes := make(chan int, 3)
	collector := newSignalBatchCollector(func(inputs []signalEvaluationInput) []*SignalResults {
		batchSizes <- len(inputs)
		return signalBatchTestResults(inputs)
	})

	results := make([]<-chan *SignalResults, signalBatchMaxSize)
	for i := range signalBatchMaxSize {
		results[i] = collector.enqueue(signalEvaluationInput{text: fmt.Sprintf("row-%02d", i)})
	}
	select {
	case got := <-batchSizes:
		if got != signalBatchMaxSize {
			t.Fatalf("full batch size = %d, want %d", got, signalBatchMaxSize)
		}
	case <-time.After(time.Second):
		t.Fatal("full batch did not flush immediately")
	}
	collected := make([]*SignalResults, len(results))
	for i, resultCh := range results {
		result := <-resultCh
		collected[i] = result
		want := signalBatchTestMarker(fmt.Sprintf("row-%02d", i))
		if got := result.SignalValues["row"]; got != want {
			t.Fatalf("row %d marker = %v, want %v", i, got, want)
		}
	}
	collected[0].SignalValues["cross-request"] = 1
	for i := 1; i < len(collected); i++ {
		if _, ok := collected[i].SignalValues["cross-request"]; ok {
			t.Fatalf("result %d shares its SignalValues map with row 0", i)
		}
	}

	collector.mu.Lock()
	staleGeneration := collector.generation
	collector.mu.Unlock()
	next := collector.enqueue(signalEvaluationInput{text: "next-generation"})
	collector.mu.Lock()
	currentGeneration := collector.generation
	collector.mu.Unlock()
	if currentGeneration == staleGeneration {
		t.Fatal("new batch reused the previous generation")
	}
	collector.flush(staleGeneration)
	select {
	case <-next:
		t.Fatal("stale timer flushed the next generation")
	case <-time.After(20 * time.Millisecond):
	}
	collector.flush(currentGeneration)
	select {
	case result := <-next:
		if got := result.SignalValues["row"]; got != signalBatchTestMarker("next-generation") {
			t.Fatalf("next-generation marker = %v", got)
		}
	case <-time.After(time.Second):
		t.Fatal("current generation did not flush")
	}
	if got := <-batchSizes; got != 1 {
		t.Fatalf("next batch size = %d, want 1", got)
	}
}

func TestSignalBatchCollectorEvaluatesOutsideMutex(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	var startedOnce sync.Once
	collector := newSignalBatchCollector(func(inputs []signalEvaluationInput) []*SignalResults {
		startedOnce.Do(func() { close(started) })
		<-release
		return signalBatchTestResults(inputs)
	})

	results := make([]<-chan *SignalResults, signalBatchMaxSize)
	for i := range signalBatchMaxSize {
		results[i] = collector.enqueue(signalEvaluationInput{text: fmt.Sprintf("row-%d", i)})
	}
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("detached full batch did not start")
	}

	enqueued := make(chan (<-chan *SignalResults), 1)
	go func() {
		enqueued <- collector.enqueue(signalEvaluationInput{text: "row-33"})
	}()
	var next <-chan *SignalResults
	select {
	case next = <-enqueued:
	case <-time.After(time.Second):
		t.Fatal("row 33 could not enqueue while detached batch was evaluating")
	}
	close(release)
	for _, result := range results {
		<-result
	}
	collector.mu.Lock()
	generation := collector.generation
	collector.mu.Unlock()
	collector.flush(generation)
	<-next
}

func TestSignalBatchCollectorOutputMismatchUnblocksEveryCaller(t *testing.T) {
	collector := newSignalBatchCollector(func([]signalEvaluationInput) []*SignalResults {
		return []*SignalResults{newSignalResults()}
	})
	results := []<-chan *SignalResults{
		collector.enqueue(signalEvaluationInput{text: "first"}),
		collector.enqueue(signalEvaluationInput{text: "second"}),
	}
	collector.mu.Lock()
	generation := collector.generation
	collector.mu.Unlock()
	collector.flush(generation)
	for i, result := range results {
		select {
		case got := <-result:
			if got == nil || got.SignalValues == nil || got.SignalConfidences == nil {
				t.Fatalf("row %d received an invalid fallback result", i)
			}
		case <-time.After(time.Second):
			t.Fatalf("row %d remained blocked after output mismatch", i)
		}
	}
}

func TestSignalBatchingLoadGateInvariant(t *testing.T) {
	for _, maxConcurrency := range []int{1, signalBatchMaxSize - 1} {
		cfg := &config.RouterConfig{}
		cfg.API.BatchClassification.SignalBatchingEnabled = true
		cfg.API.BatchClassification.MaxConcurrency = maxConcurrency
		if _, err := newClassifierWithOptions(cfg); err == nil {
			t.Fatalf("MaxConcurrency=%d succeeded with signal batching enabled", maxConcurrency)
		}
	}
	for _, maxConcurrency := range []int{0, signalBatchMaxSize, signalBatchMaxSize + 1} {
		cfg := &config.RouterConfig{}
		cfg.API.BatchClassification.SignalBatchingEnabled = true
		cfg.API.BatchClassification.MaxConcurrency = maxConcurrency
		classifier, err := newClassifierWithOptions(cfg)
		if err != nil {
			t.Fatalf("MaxConcurrency=%d: %v", maxConcurrency, err)
		}
		if classifier.signalBatchCollector == nil {
			t.Fatalf("MaxConcurrency=%d did not create a collector", maxConcurrency)
		}
	}
}

func TestSignalBatchingDisabledBypassesCollector(t *testing.T) {
	rule := config.KeywordRule{Name: "scalar-keyword", Operator: "OR", Keywords: []string{"scalar"}}
	keywordClassifier, err := NewKeywordClassifier([]config.KeywordRule{rule})
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{}
	cfg.KeywordRules = []config.KeywordRule{rule}
	classifier, err := newClassifierWithOptions(cfg, withKeywordClassifier(keywordClassifier))
	if err != nil {
		t.Fatal(err)
	}
	if classifier.signalBatchCollector != nil {
		t.Fatal("disabled signal batching created a collector")
	}
	started := time.Now()
	result := classifier.EvaluateAllSignalsWithForceOption("scalar", true)
	if elapsed := time.Since(started); elapsed >= signalBatchWindow {
		t.Fatalf("disabled path waited for batch window: %v", elapsed)
	}
	if result == nil || result.SignalValues == nil || result.SignalConfidences == nil {
		t.Fatal("disabled path did not return initialized scalar results")
	}
	if !containsString(result.MatchedKeywordRules, "scalar-keyword") {
		t.Fatalf("disabled path did not preserve active scalar signal: %v", result.MatchedKeywordRules)
	}
}

func TestSignalBatchingNormalizesNativeTextInputsWithoutDroppingSignals(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.API.BatchClassification.SignalBatchingEnabled = true
	cfg.API.BatchClassification.MaxConcurrency = signalBatchMaxSize
	cfg.API.BatchClassification.ConcurrencyThreshold = 1
	cfg.ComplexityRules = []config.ComplexityRule{{Name: "complexity", Method: config.ComplexityMethodModel, Threshold: 0.5}}
	complexity := &complexityBatchSpy{results: []candle.ClassResult{{Class: 2, Confidence: 0.9}}}
	classifier, err := newClassifierWithOptions(cfg, withComplexityModel(difficultyMapping(), nil, complexity))
	if err != nil {
		t.Fatal(err)
	}
	captured := make(chan signalEvaluationInput, 1)
	evaluate := classifier.signalBatchCollector.evaluate
	classifier.signalBatchCollector.evaluate = func(inputs []signalEvaluationInput) []*SignalResults {
		captured <- inputs[0]
		return evaluate(inputs)
	}

	priorUserMessages := []string{"prior\x00suffix"}
	nonUserMessages := []string{"system\x00suffix"}
	result := classifier.EvaluateAllSignalsWithContext(
		"body\x00suffix", "context\x00unchanged", "current\x00suffix",
		priorUserMessages, nonUserMessages, false, true, "raw\x00suffix",
		map[string]bool{config.SignalTypeComplexity: true}, ConversationFacts{}, "",
	)
	input := <-captured
	if input.text != "body\uFFFDsuffix" || input.currentUserText != "current\uFFFDsuffix" || input.uncompressedText != "raw\uFFFDsuffix" {
		t.Fatalf("normalized input = %+v", input)
	}
	assertStringSlice(t, "prior user messages", input.priorUserMessages, []string{"prior\uFFFDsuffix"})
	assertStringSlice(t, "non-user messages", input.nonUserMessages, []string{"system\uFFFDsuffix"})
	if input.contextText != "context\x00unchanged" {
		t.Fatalf("context-only text changed: %q", input.contextText)
	}
	assertStringSlice(t, "caller prior user messages", priorUserMessages, []string{"prior\x00suffix"})
	assertStringSlice(t, "caller non-user messages", nonUserMessages, []string{"system\x00suffix"})
	if !containsString(result.MatchedComplexityRules, "complexity:hard") {
		t.Fatalf("normalized row lost its model signal: %v", result.MatchedComplexityRules)
	}
	if complexity.batchCalls != 1 || complexity.scalarCalls != 0 {
		t.Fatalf("model calls: batch=%d scalar=%d", complexity.batchCalls, complexity.scalarCalls)
	}
	assertStringSlice(t, "complexity inputs", complexity.inputs, []string{"raw\uFFFDsuffix"})
}

func TestSignalBatchingIsPerClassifierAndIgnoresConfiguredMaxBatchSize(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.API.BatchClassification.SignalBatchingEnabled = true
	cfg.API.BatchClassification.MaxBatchSize = 1
	first, err := newClassifierWithOptions(cfg)
	if err != nil {
		t.Fatal(err)
	}
	second, err := newClassifierWithOptions(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if first.signalBatchCollector == second.signalBatchCollector {
		t.Fatal("classifier instances share a signal batch collector")
	}

	result := first.signalBatchCollector.enqueue(signalEvaluationInput{text: "first"})
	select {
	case <-result:
		t.Fatal("configured max_batch_size changed the fixed signal batch size")
	case <-time.After(20 * time.Millisecond):
	}
	for i := 1; i < signalBatchMaxSize; i++ {
		first.signalBatchCollector.enqueue(signalEvaluationInput{text: fmt.Sprintf("row-%d", i)})
	}
	select {
	case <-result:
	case <-time.After(time.Second):
		t.Fatal("fixed 32-row batch did not flush")
	}
}

func signalBatchTestResults(inputs []signalEvaluationInput) []*SignalResults {
	results := make([]*SignalResults, len(inputs))
	for i, input := range inputs {
		results[i] = newSignalResults()
		results[i].SignalValues["row"] = signalBatchTestMarker(input.text)
	}
	return results
}

func signalBatchTestMarker(text string) float64 {
	marker := 0
	for _, value := range []byte(text) {
		marker = marker*31 + int(value)
	}
	return float64(marker)
}
