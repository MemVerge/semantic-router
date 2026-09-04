package classification

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	metricspkg "github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type complexityBatchSpy struct {
	mu          sync.Mutex
	batchCalls  int
	scalarCalls int
	inputs      []string
	results     []candle.ClassResult
	err         error
}

func (s *complexityBatchSpy) Classify(string) (candle.ClassResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.scalarCalls++
	return candle.ClassResult{}, fmt.Errorf("unexpected scalar complexity call")
}

func (s *complexityBatchSpy) ClassifyBatch(texts []string) ([]candle.ClassResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.batchCalls++
	s.inputs = append([]string(nil), texts...)
	return append([]candle.ClassResult(nil), s.results...), s.err
}

type categoryBatchSpy struct {
	mu          sync.Mutex
	batchCalls  int
	scalarCalls int
	inputs      []string
	results     []candle.ClassResult
	err         error
}

func (s *categoryBatchSpy) Classify(string) (candle.ClassResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.scalarCalls++
	return candle.ClassResult{}, fmt.Errorf("unexpected scalar domain call")
}

func (s *categoryBatchSpy) ClassifyWithProbabilities(string) (candle.ClassResultWithProbs, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.scalarCalls++
	return candle.ClassResultWithProbs{}, fmt.Errorf("unexpected scalar domain probabilities call")
}

func (s *categoryBatchSpy) ClassifyBatch(texts []string) ([]candle.ClassResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.batchCalls++
	s.inputs = append([]string(nil), texts...)
	return append([]candle.ClassResult(nil), s.results...), s.err
}

func TestSignalModelFamiliesUseOneSharedBatchInFIFOOrder(t *testing.T) {
	complexity := &complexityBatchSpy{results: []candle.ClassResult{
		{Class: 2, Confidence: 0.91},
		{Class: 0, Confidence: 0.82},
	}}
	domain := &categoryBatchSpy{results: []candle.ClassResult{
		{Class: 0, Confidence: 0.88},
		{Class: 1, Confidence: 0.79},
	}}

	factCalls := 0
	var factInputs []string
	previousFactBatch := classifyMmBert32KFactcheckBatch
	classifyMmBert32KFactcheckBatch = func(texts []string) ([]candle.ClassResult, error) {
		factCalls++
		factInputs = append([]string(nil), texts...)
		return []candle.ClassResult{{Class: 1, Confidence: 0.92}, {Class: 0, Confidence: 0.87}}, nil
	}
	t.Cleanup(func() { classifyMmBert32KFactcheckBatch = previousFactBatch })

	embeddingCalls := 0
	var embeddingInputs []string
	var embeddingTargetDim int
	previousEmbeddingBatch := encodeMmBert32KTextBatch
	encodeMmBert32KTextBatch = func(texts []string, targetDim int) ([][]float32, error) {
		embeddingCalls++
		embeddingInputs = append([]string(nil), texts...)
		embeddingTargetDim = targetDim
		return [][]float32{{1, 0, 0}, {0, 1, 0}}, nil
	}
	t.Cleanup(func() { encodeMmBert32KTextBatch = previousEmbeddingBatch })

	classifier := newFourFamilyBatchTestClassifier(t, complexity, domain)
	metricLabels := [][2]string{
		{config.SignalTypeComplexity, "complexity:hard"},
		{config.SignalTypeComplexity, "complexity:easy"},
		{config.SignalTypeDomain, "science"},
		{config.SignalTypeDomain, "math"},
		{config.SignalTypeFactCheck, "needs_fact_check"},
		{config.SignalTypeFactCheck, "no_fact_check_needed"},
		{config.SignalTypeEmbedding, "topic"},
	}
	extractionsBefore := make([]float64, len(metricLabels))
	matchesBefore := make([]float64, len(metricLabels))
	for i, labels := range metricLabels {
		extractionsBefore[i] = testutil.ToFloat64(metricspkg.SignalExtractionTotal.WithLabelValues(labels[0], labels[1]))
		matchesBefore[i] = testutil.ToFloat64(metricspkg.SignalMatchTotal.WithLabelValues(labels[0], labels[1]))
	}
	inputs := []signalEvaluationInput{
		{
			text:                   "compressed-first",
			uncompressedText:       "raw-first",
			skipCompressionSignals: fourModelSignalTypes(),
			forceEvaluateAll:       true,
		},
		{text: "second", forceEvaluateAll: true},
	}
	results := classifier.evaluateSignalInputs(inputs, true)

	wantInputs := []string{"raw-first", "second"}
	assertStringSlice(t, "complexity inputs", complexity.inputs, wantInputs)
	assertStringSlice(t, "domain inputs", domain.inputs, wantInputs)
	assertStringSlice(t, "fact-check inputs", factInputs, wantInputs)
	assertStringSlice(t, "embedding inputs", embeddingInputs, wantInputs)
	if complexity.batchCalls != 1 || complexity.scalarCalls != 0 {
		t.Fatalf("complexity calls: batch=%d scalar=%d", complexity.batchCalls, complexity.scalarCalls)
	}
	if domain.batchCalls != 1 || domain.scalarCalls != 0 {
		t.Fatalf("domain calls: batch=%d scalar=%d", domain.batchCalls, domain.scalarCalls)
	}
	if factCalls != 1 {
		t.Fatalf("fact-check batch calls = %d, want 1", factCalls)
	}
	if embeddingCalls != 1 || embeddingTargetDim != 3 {
		t.Fatalf("embedding calls=%d targetDim=%d, want 1 and 3", embeddingCalls, embeddingTargetDim)
	}

	if !containsString(results[0].MatchedComplexityRules, "complexity:hard") ||
		!containsString(results[1].MatchedComplexityRules, "complexity:easy") {
		t.Fatalf("complexity row alignment failed: %v / %v", results[0].MatchedComplexityRules, results[1].MatchedComplexityRules)
	}
	if !containsString(results[0].MatchedDomainRules, "science") ||
		!containsString(results[1].MatchedDomainRules, "math") {
		t.Fatalf("domain row alignment failed: %v / %v", results[0].MatchedDomainRules, results[1].MatchedDomainRules)
	}
	if !containsString(results[0].MatchedFactCheckRules, "needs_fact_check") ||
		!containsString(results[1].MatchedFactCheckRules, "no_fact_check_needed") {
		t.Fatalf("fact-check row alignment failed: %v / %v", results[0].MatchedFactCheckRules, results[1].MatchedFactCheckRules)
	}
	if !containsString(results[0].MatchedEmbeddingRules, "topic") || containsString(results[1].MatchedEmbeddingRules, "topic") {
		t.Fatalf("embedding row alignment failed: %v / %v", results[0].MatchedEmbeddingRules, results[1].MatchedEmbeddingRules)
	}
	for i, labels := range metricLabels {
		extractionsAfter := testutil.ToFloat64(metricspkg.SignalExtractionTotal.WithLabelValues(labels[0], labels[1]))
		matchesAfter := testutil.ToFloat64(metricspkg.SignalMatchTotal.WithLabelValues(labels[0], labels[1]))
		if delta := extractionsAfter - extractionsBefore[i]; delta != 1 {
			t.Fatalf("%s/%s extraction metric delta = %v, want 1", labels[0], labels[1], delta)
		}
		if delta := matchesAfter - matchesBefore[i]; delta != 1 {
			t.Fatalf("%s/%s match metric delta = %v, want 1", labels[0], labels[1], delta)
		}
	}
	if results[0].Metrics.Complexity.ExecutionTimeMs != results[1].Metrics.Complexity.ExecutionTimeMs ||
		results[0].Metrics.Domain.ExecutionTimeMs != results[1].Metrics.Domain.ExecutionTimeMs ||
		results[0].Metrics.FactCheck.ExecutionTimeMs != results[1].Metrics.FactCheck.ExecutionTimeMs ||
		results[0].Metrics.Embedding.ExecutionTimeMs != results[1].Metrics.Embedding.ExecutionTimeMs {
		t.Fatalf("model batch duration was not shared by row: %+v / %+v", results[0].Metrics, results[1].Metrics)
	}
}

func TestComplexityBatchLengthMismatchFailsEveryRowWithoutScalarRetry(t *testing.T) {
	for _, resultCount := range []int{1, 3} {
		t.Run(fmt.Sprintf("result_count_%d", resultCount), func(t *testing.T) {
			batchResults := make([]candle.ClassResult, resultCount)
			spy := &complexityBatchSpy{results: batchResults}
			cfg := &config.RouterConfig{}
			cfg.ComplexityRules = []config.ComplexityRule{{Name: "complexity", Method: config.ComplexityMethodModel, Threshold: 0.5}}
			classifier, err := newClassifierWithOptions(cfg, withComplexityModel(difficultyMapping(), nil, spy))
			if err != nil {
				t.Fatal(err)
			}
			results := classifier.evaluateSignalInputs([]signalEvaluationInput{
				{text: "first", forceEvaluateAll: true},
				{text: "second", forceEvaluateAll: true},
			}, true)
			if spy.batchCalls != 1 || spy.scalarCalls != 0 {
				t.Fatalf("calls: batch=%d scalar=%d", spy.batchCalls, spy.scalarCalls)
			}
			for i, result := range results {
				if len(result.MatchedComplexityRules) != 0 {
					t.Fatalf("row %d mutated before output count validation: %v", i, result.MatchedComplexityRules)
				}
			}
		})
	}
}

func TestMmBERT32KCategoryBatchUsesConfiguredNonCandleBackend(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "openvino")

	classifier := &MmBERT32KCategoryInferenceImpl{}
	_, err := classifier.ClassifyBatch([]string{"first", "second"})
	if err == nil || !strings.Contains(err.Error(), "openvino backend requires") {
		t.Fatalf("ClassifyBatch() error = %v, want OpenVINO backend error", err)
	}
}

func TestFactCheckBatchPreservesEmptyAndThresholdSemantics(t *testing.T) {
	previous := classifyMmBert32KFactcheckBatch
	var gotTexts []string
	classifyMmBert32KFactcheckBatch = func(texts []string) ([]candle.ClassResult, error) {
		gotTexts = append([]string(nil), texts...)
		return []candle.ClassResult{
			{Class: 1, Confidence: 0.6},
			{Class: 1, Confidence: 0.8},
		}, nil
	}
	t.Cleanup(func() { classifyMmBert32KFactcheckBatch = previous })

	classifier := &FactCheckClassifier{
		config:       &config.FactCheckModelConfig{Threshold: 0.7},
		initialized:  true,
		useMmBERT32K: true,
	}
	results, err := classifier.ClassifyBatch([]string{"", "below", "above"})
	if err != nil {
		t.Fatal(err)
	}
	assertStringSlice(t, "non-empty fact-check inputs", gotTexts, []string{"below", "above"})
	if results[0].NeedsFactCheck || results[0].Confidence != 1 {
		t.Fatalf("empty result = %+v", results[0])
	}
	if results[1].NeedsFactCheck || results[1].Label != FactCheckLabelNotNeeded || results[1].Confidence < 0.399 || results[1].Confidence > 0.401 {
		t.Fatalf("below-threshold result = %+v", results[1])
	}
	if !results[2].NeedsFactCheck || results[2].Label != FactCheckLabelNeeded || results[2].Confidence != 0.8 {
		t.Fatalf("above-threshold result = %+v", results[2])
	}
}

func TestEmbeddingBatchSkipsWhitespaceAndKeepsImageResultOnTextFailure(t *testing.T) {
	rules := []config.EmbeddingRule{
		{
			Name:                      "text-topic",
			Candidates:                []string{"text-anchor"},
			SimilarityThreshold:       0.5,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
		{
			Name:                      "image-topic",
			Candidates:                []string{"image-anchor"},
			SimilarityThreshold:       0.5,
			AggregationMethodConfiged: config.AggregationMethodMax,
			QueryModality:             config.QueryModalityImage,
		},
	}
	embeddingConfig := config.HNSWConfig{ModelType: "mmbert", TargetDimension: 3, PreloadEmbeddings: true}.WithDefaults()
	embeddingClassifier, err := NewEmbeddingClassifier(rules, embeddingConfig)
	if err != nil {
		t.Fatal(err)
	}
	embeddingClassifier.candidateEmbeddings = map[string][]float32{
		"text-anchor":  {1, 0, 0},
		"image-anchor": {0, 1, 0},
	}
	embeddingClassifier.rebuildRulePrototypeBanks()
	embeddingClassifier.preloadComplete = true

	previousBatch := encodeMmBert32KTextBatch
	var gotTexts []string
	encodeMmBert32KTextBatch = func(texts []string, _ int) ([][]float32, error) {
		gotTexts = append([]string(nil), texts...)
		return nil, errors.New("text batch failed")
	}
	t.Cleanup(func() { encodeMmBert32KTextBatch = previousBatch })
	previousImage := getMultiModalImageEmbedding
	getMultiModalImageEmbedding = func(string, int) ([]float32, error) {
		return []float32{0, 1, 0}, nil
	}
	t.Cleanup(func() { getMultiModalImageEmbedding = previousImage })

	cfg := &config.RouterConfig{}
	cfg.EmbeddingRules = rules
	classifier, err := newClassifierWithOptions(cfg, withKeywordEmbeddingClassifier(nil, embeddingClassifier))
	if err != nil {
		t.Fatal(err)
	}
	results := classifier.evaluateSignalInputs([]signalEvaluationInput{
		{text: "   ", forceEvaluateAll: true},
		{text: "query", imageURL: "image", forceEvaluateAll: true},
	}, true)
	assertStringSlice(t, "embedding batch inputs", gotTexts, []string{"query"})
	if len(results[0].MatchedEmbeddingRules) != 0 || results[0].Metrics.Embedding.ExecutionTimeMs != 0 {
		t.Fatalf("whitespace-only row was evaluated: rules=%v metrics=%+v", results[0].MatchedEmbeddingRules, results[0].Metrics.Embedding)
	}
	if !containsString(results[1].MatchedEmbeddingRules, "image-topic") || containsString(results[1].MatchedEmbeddingRules, "text-topic") {
		t.Fatalf("image result did not survive text failure: %v", results[1].MatchedEmbeddingRules)
	}
}

func TestEmbeddingBatchWithoutRulesReturnsInitializedRows(t *testing.T) {
	results, err := (&EmbeddingClassifier{}).ClassifyDetailedBatch([]string{"first", "second"})
	if err != nil {
		t.Fatal(err)
	}
	for i, result := range results {
		if result == nil {
			t.Fatalf("row %d is nil", i)
		}
	}
}

func TestSignalBatchWaitsForFamilyAndAppliesOutputPolicyBeforeDelivery(t *testing.T) {
	rules := []config.EmbeddingRule{
		{
			Name:                      "first-topic",
			Candidates:                []string{"first-anchor"},
			SimilarityThreshold:       0.5,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
		{
			Name:                      "second-topic",
			Candidates:                []string{"second-anchor"},
			SimilarityThreshold:       0.5,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
	}
	embeddingConfig := config.HNSWConfig{ModelType: "mmbert", TargetDimension: 3, PreloadEmbeddings: true}.WithDefaults()
	embeddingClassifier, err := NewEmbeddingClassifier(rules, embeddingConfig)
	if err != nil {
		t.Fatal(err)
	}
	embeddingClassifier.candidateEmbeddings = map[string][]float32{
		"first-anchor":  {1, 0, 0},
		"second-anchor": {0.9, 0.1, 0},
	}
	embeddingClassifier.rebuildRulePrototypeBanks()
	embeddingClassifier.preloadComplete = true

	started := make(chan struct{})
	release := make(chan struct{})
	previousBatch := encodeMmBert32KTextBatch
	encodeMmBert32KTextBatch = func([]string, int) ([][]float32, error) {
		close(started)
		<-release
		return [][]float32{{1, 0, 0}}, nil
	}
	t.Cleanup(func() { encodeMmBert32KTextBatch = previousBatch })

	cfg := &config.RouterConfig{}
	cfg.API.BatchClassification.SignalBatchingEnabled = true
	cfg.EmbeddingRules = rules
	classifier, err := newClassifierWithOptions(cfg, withKeywordEmbeddingClassifier(nil, embeddingClassifier))
	if err != nil {
		t.Fatal(err)
	}
	resultCh := classifier.signalBatchCollector.enqueue(signalEvaluationInput{text: "query", forceEvaluateAll: true})
	classifier.signalBatchCollector.mu.Lock()
	generation := classifier.signalBatchCollector.generation
	classifier.signalBatchCollector.mu.Unlock()
	go classifier.signalBatchCollector.flush(generation)
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("embedding family did not start")
	}
	select {
	case <-resultCh:
		t.Fatal("result delivered while a signal family was still running")
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	select {
	case result := <-resultCh:
		if len(result.MatchedEmbeddingRules) != 1 || result.MatchedEmbeddingRules[0] != "first-topic" {
			t.Fatalf("output policy was not applied before delivery: %v", result.MatchedEmbeddingRules)
		}
	case <-time.After(time.Second):
		t.Fatal("result not delivered after signal family completed")
	}
}

func TestSignalModelBatchFailureDoesNotRetryAndCheapFamilyStillCompletes(t *testing.T) {
	modelErr := errors.New("batch failed")
	complexity := &complexityBatchSpy{err: modelErr}
	domain := &categoryBatchSpy{err: modelErr}
	factCalls := 0
	previousFactBatch := classifyMmBert32KFactcheckBatch
	classifyMmBert32KFactcheckBatch = func([]string) ([]candle.ClassResult, error) {
		factCalls++
		return nil, modelErr
	}
	t.Cleanup(func() { classifyMmBert32KFactcheckBatch = previousFactBatch })
	embeddingCalls := 0
	previousEmbeddingBatch := encodeMmBert32KTextBatch
	encodeMmBert32KTextBatch = func([]string, int) ([][]float32, error) {
		embeddingCalls++
		return nil, modelErr
	}
	t.Cleanup(func() { encodeMmBert32KTextBatch = previousEmbeddingBatch })

	classifier := newFourFamilyBatchTestClassifier(t, complexity, domain)
	keywordRule := config.KeywordRule{Name: "cheap", Operator: "OR", Keywords: []string{"first", "second"}}
	keywordClassifier, err := NewKeywordClassifier([]config.KeywordRule{keywordRule})
	if err != nil {
		t.Fatal(err)
	}
	classifier.Config.KeywordRules = []config.KeywordRule{keywordRule}
	classifier.keywordClassifier = keywordClassifier
	results := classifier.evaluateSignalInputs([]signalEvaluationInput{
		{text: "first", forceEvaluateAll: true},
		{text: "second", forceEvaluateAll: true},
	}, true)
	if complexity.batchCalls != 1 || complexity.scalarCalls != 0 {
		t.Fatalf("complexity calls: batch=%d scalar=%d", complexity.batchCalls, complexity.scalarCalls)
	}
	if domain.batchCalls != 1 || domain.scalarCalls != 0 {
		t.Fatalf("domain calls: batch=%d scalar=%d", domain.batchCalls, domain.scalarCalls)
	}
	if factCalls != 1 || embeddingCalls != 1 {
		t.Fatalf("fact-check calls=%d embedding calls=%d, want one each", factCalls, embeddingCalls)
	}
	for i, result := range results {
		if len(result.MatchedComplexityRules) != 0 || len(result.MatchedDomainRules) != 0 ||
			len(result.MatchedFactCheckRules) != 0 || len(result.MatchedEmbeddingRules) != 0 {
			t.Fatalf("row %d retained a failed model-family result: %+v", i, result)
		}
		if !containsString(result.MatchedKeywordRules, "cheap") {
			t.Fatalf("row %d cheap family did not complete: %v", i, result.MatchedKeywordRules)
		}
	}
}

func newFourFamilyBatchTestClassifier(t *testing.T, complexity ComplexityInference, domain CategoryInference) *Classifier {
	t.Helper()
	embeddingRules := []config.EmbeddingRule{{
		Name:                      "topic",
		Candidates:                []string{"candidate"},
		SimilarityThreshold:       0.5,
		AggregationMethodConfiged: config.AggregationMethodMax,
	}}
	embeddingConfig := config.HNSWConfig{
		ModelType:         "mmbert",
		TargetDimension:   3,
		PreloadEmbeddings: true,
	}.WithDefaults()
	embeddingClassifier, err := NewEmbeddingClassifier(embeddingRules, embeddingConfig)
	if err != nil {
		t.Fatal(err)
	}
	embeddingClassifier.candidateEmbeddings = map[string][]float32{"candidate": {1, 0, 0}}
	embeddingClassifier.rebuildRulePrototypeBanks()
	embeddingClassifier.preloadComplete = true

	factConfig := config.FactCheckModelConfig{
		ModelID:      "fact-model",
		Threshold:    0.7,
		UseMmBERT32K: true,
	}
	factClassifier := &FactCheckClassifier{
		config:       &factConfig,
		initialized:  true,
		useMmBERT32K: true,
	}

	categoryMapping := &CategoryMapping{
		CategoryToIdx: map[string]int{"science": 0, "math": 1},
		IdxToCategory: map[string]string{"0": "science", "1": "math"},
	}
	cfg := &config.RouterConfig{}
	cfg.Categories = []config.Category{
		{CategoryMetadata: config.CategoryMetadata{Name: "science"}},
		{CategoryMetadata: config.CategoryMetadata{Name: "math"}},
	}
	cfg.CategoryModel.ModelID = "domain-model"
	cfg.CategoryModel.CategoryMappingPath = "mapping.json"
	cfg.CategoryModel.Threshold = 0.2
	cfg.HallucinationMitigation.FactCheckModel = factConfig
	cfg.FactCheckRules = []config.FactCheckRule{{Name: "needs_fact_check"}, {Name: "no_fact_check_needed"}}
	cfg.EmbeddingRules = embeddingRules
	cfg.ComplexityRules = []config.ComplexityRule{{Name: "complexity", Method: config.ComplexityMethodModel, Threshold: 0.5}}
	classifier, err := newClassifierWithOptions(
		cfg,
		withCategory(categoryMapping, nil, domain),
		withComplexityModel(difficultyMapping(), nil, complexity),
		withKeywordEmbeddingClassifier(nil, embeddingClassifier),
	)
	if err != nil {
		t.Fatal(err)
	}
	classifier.factCheckClassifier = factClassifier
	return classifier
}

func fourModelSignalTypes() map[string]bool {
	return map[string]bool{
		config.SignalTypeComplexity: true,
		config.SignalTypeDomain:     true,
		config.SignalTypeFactCheck:  true,
		config.SignalTypeEmbedding:  true,
	}
}

func assertStringSlice(t *testing.T, name string, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s = %v, want %v", name, got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s = %v, want %v", name, got, want)
		}
	}
}

func TestFactCheckBatchFailedRowYieldsNoResultInsteadOfNotNeeded(t *testing.T) {
	previous := classifyMmBert32KFactcheckBatch
	classifyMmBert32KFactcheckBatch = func(texts []string) ([]candle.ClassResult, error) {
		return []candle.ClassResult{
			{Class: -1, Confidence: 0.0},
			{Class: 1, Confidence: 0.9},
		}, nil
	}
	t.Cleanup(func() { classifyMmBert32KFactcheckBatch = previous })

	classifier := &FactCheckClassifier{
		config:       &config.FactCheckModelConfig{Threshold: 0.7},
		initialized:  true,
		useMmBERT32K: true,
	}
	results, err := classifier.ClassifyBatch([]string{"failed", "ok"})
	if err != nil {
		t.Fatalf("one bad row failed the wave: %v", err)
	}
	if results[0] != nil {
		t.Fatalf("failed row fabricated a verdict: %+v", results[0])
	}
	if results[1] == nil || !results[1].NeedsFactCheck || results[1].Confidence != 0.9 {
		t.Fatalf("survivor row = %+v", results[1])
	}
}
