package classification

import (
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// signalReadiness returns a map indicating whether each signal type's infrastructure is ready.
// Separated from EvaluateAllSignalsWithContext to keep cyclomatic complexity under the linter limit.
func (c *Classifier) signalReadiness() map[string]bool {
	return map[string]bool{
		config.SignalTypeKeyword:      c.keywordClassifier != nil,
		config.SignalTypeEmbedding:    c.keywordEmbeddingClassifier != nil,
		config.SignalTypeDomain:       c.IsCategoryEnabled() && c.categoryInference != nil && c.CategoryMapping != nil,
		config.SignalTypeFactCheck:    len(c.Config.FactCheckRules) > 0 && c.IsFactCheckEnabled(),
		config.SignalTypeUserFeedback: len(c.Config.UserFeedbackRules) > 0 && c.IsFeedbackDetectorEnabled(),
		config.SignalTypeReask:        c.reaskClassifier != nil,
		config.SignalTypePreference:   len(c.Config.PreferenceRules) > 0 && c.IsPreferenceClassifierEnabled(),
		config.SignalTypeLanguage:     len(c.Config.LanguageRules) > 0 && c.IsLanguageEnabled(),
		config.SignalTypeContext:      c.contextClassifier != nil,
		config.SignalTypeStructure:    c.structureClassifier != nil,
		config.SignalTypeComplexity:   c.complexityClassifier != nil || c.complexityModelInference != nil,
		config.SignalTypeModality:     len(c.Config.ModalityRules) > 0 && c.Config.ModalityDetector.Enabled,
		config.SignalTypeJailbreak:    len(c.Config.JailbreakRules) > 0 && c.IsJailbreakEnabled(),
		config.SignalTypePII:          len(c.Config.PIIRules) > 0 && c.IsPIIEnabled(),
		config.SignalTypeKB:           len(c.kbClassifiers) > 0,
		config.SignalTypeConversation: len(c.Config.ConversationRules) > 0,
		config.SignalTypeEvent:        c.eventClassifier != nil,
	}
}

// textForSignalFunc returns a function that resolves the correct text for a given signal type,
// using uncompressed text for signals that must not receive compressed input.
func textForSignalFunc(text, uncompressedText string, skipCompressionSignals map[string]bool) func(string) string {
	return func(signalType string) string {
		if uncompressedText != "" && skipCompressionSignals[signalType] {
			return uncompressedText
		}
		return text
	}
}

// Native classifiers cross a C-string boundary; replacement preserves the suffix.
func normalizeSignalText(text string) string {
	return strings.ReplaceAll(text, "\x00", "\uFFFD")
}

func normalizeSignalTexts(texts []string) []string {
	for _, text := range texts {
		if strings.IndexByte(text, 0) < 0 {
			continue
		}
		normalized := append([]string(nil), texts...)
		for i := range normalized {
			normalized[i] = normalizeSignalText(normalized[i])
		}
		return normalized
	}
	return texts
}

// EvaluateAllSignalsWithContext evaluates all signal types with separate text for context counting.
//
// text: (possibly compressed) text for signal evaluation
// contextText: text for context token counting (usually all messages combined)
// nonUserMessages: conversation history for jailbreak/PII with include_history
// forceEvaluateAll: if true, evaluates all configured signals regardless of decision usage
// uncompressedText: original text before prompt compression (empty = no compression happened)
// skipCompressionSignals: signal types that must use uncompressedText instead of text
// imageURL: image URL for multimodal signals ("" when the request carries no image)
func (c *Classifier) EvaluateAllSignalsWithContext(text string, contextText string, currentUserText string, priorUserMessages []string, nonUserMessages []string, hasPriorAssistantReply bool, forceEvaluateAll bool, uncompressedText string, skipCompressionSignals map[string]bool, convFacts ConversationFacts, imageURL string) *SignalResults {
	defer c.enterSignalEvaluationLoadGate()()
	input := signalEvaluationInput{
		text:                   normalizeSignalText(text),
		contextText:            contextText,
		currentUserText:        normalizeSignalText(currentUserText),
		priorUserMessages:      normalizeSignalTexts(priorUserMessages),
		nonUserMessages:        normalizeSignalTexts(nonUserMessages),
		hasPriorAssistantReply: hasPriorAssistantReply,
		forceEvaluateAll:       forceEvaluateAll,
		uncompressedText:       normalizeSignalText(uncompressedText),
		skipCompressionSignals: skipCompressionSignals,
		convFacts:              convFacts,
		imageURL:               imageURL,
	}
	if c.signalBatchCollector == nil {
		return c.evaluateSignalInputs([]signalEvaluationInput{input}, false)[0]
	}
	return <-c.signalBatchCollector.enqueue(input)
}

type signalEvaluationRow struct {
	input         signalEvaluationInput
	usedSignals   map[string]bool
	textForSignal func(string) string
	results       *SignalResults
	mu            sync.Mutex
	imgCache      *requestImageEmbeddingCache
}

func (c *Classifier) evaluateSignalBatch(inputs []signalEvaluationInput) []*SignalResults {
	return c.evaluateSignalInputs(inputs, true)
}

func (c *Classifier) evaluateSignalInputs(inputs []signalEvaluationInput, batchModels bool) []*SignalResults {
	rows := make([]*signalEvaluationRow, len(inputs))
	for i, input := range inputs {
		var usedSignals map[string]bool
		if input.forceEvaluateAll {
			usedSignals = c.getAllSignalTypes()
			logging.Debugf("[Signal Computation] Force evaluate all signals mode enabled")
		} else {
			usedSignals = c.getUsedSignals()
		}
		row := &signalEvaluationRow{
			input:         input,
			usedSignals:   usedSignals,
			textForSignal: textForSignalFunc(input.text, input.uncompressedText, input.skipCompressionSignals),
			results:       newSignalResults(),
		}
		if input.imageURL != "" {
			row.imgCache = newRequestImageEmbeddingCache()
		}
		rows[i] = row
	}

	var wg sync.WaitGroup
	dispatchers := c.buildSignalDispatchers()
	runSignalDispatchers(dispatchers, rows, c.signalReadiness(), batchModels, &wg)
	wg.Wait()

	results := make([]*SignalResults, len(rows))
	for i, row := range rows {
		result := c.applySignalGroups(row.results)
		result = c.applySignalComposers(result)
		result = c.applySignalOutputPolicies(result)
		results[i] = c.applyProjections(result)
	}
	return results
}
