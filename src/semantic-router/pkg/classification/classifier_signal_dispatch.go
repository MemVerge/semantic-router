package classification

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type signalDispatch struct {
	signalType    string
	name          string
	evaluate      func(*signalEvaluationRow)
	evaluateBatch func([]*signalEvaluationRow)
}

// buildSignalDispatchers assembles the wave's dispatcher table. The groups below are split
// by what each evaluator actually consumes - per-signal text, text plus an image, prior
// conversation state, or conversation history - so a new signal family lands next to the
// others reading the same inputs instead of extending one flat table. Dispatchers run one
// goroutine each and write disjoint fields under the row mutex, so group order is not
// observable.
func (c *Classifier) buildSignalDispatchers() []signalDispatch {
	dispatchers := make([]signalDispatch, 0, 17)
	dispatchers = append(dispatchers, c.textSignalDispatchers()...)
	dispatchers = append(dispatchers, c.multimodalSignalDispatchers()...)
	dispatchers = append(dispatchers, c.conversationStateSignalDispatchers()...)
	dispatchers = append(dispatchers, c.historyAwareSignalDispatchers()...)
	return dispatchers
}

// textSignalDispatchers covers the families evaluated from the row's per-signal text alone.
func (c *Classifier) textSignalDispatchers() []signalDispatch {
	return []signalDispatch{
		{
			signalType: config.SignalTypeKeyword,
			name:       "Keyword",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateKeywordSignal(row.results, &row.mu, row.textForSignal(config.SignalTypeKeyword))
			},
		},
		{
			signalType: config.SignalTypeDomain,
			name:       "Domain",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateDomainSignal(row.results, &row.mu, row.textForSignal(config.SignalTypeDomain))
			},
			evaluateBatch: c.evaluateDomainSignalsBatch,
		},
		{
			signalType: config.SignalTypeFactCheck,
			name:       "Fact-check",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateFactCheckSignal(row.results, &row.mu, row.textForSignal(config.SignalTypeFactCheck))
			},
			evaluateBatch: c.evaluateFactCheckSignalsBatch,
		},
		{
			signalType: config.SignalTypePreference,
			name:       "Preference",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluatePreferenceSignal(row.results, &row.mu, row.textForSignal(config.SignalTypePreference))
			},
		},
		{
			signalType: config.SignalTypeLanguage,
			name:       "Language",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateLanguageSignal(row.results, &row.mu, row.textForSignal(config.SignalTypeLanguage))
			},
		},
		{
			signalType: config.SignalTypeStructure,
			name:       "Structure",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateStructureSignal(row.results, &row.mu, row.textForSignal(config.SignalTypeStructure))
			},
		},
		{
			signalType: config.SignalTypeModality,
			name:       "Modality",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateModalitySignal(row.results, &row.mu, row.textForSignal(config.SignalTypeModality))
			},
		},
		{
			signalType: config.SignalTypeKB,
			name:       "KB",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateKBSignals(row.results, &row.mu, row.textForSignal(config.SignalTypeKB))
			},
		},
		{
			signalType: config.SignalTypeEvent,
			name:       "Event",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateEventSignal(row.results, &row.mu, row.textForSignal(config.SignalTypeEvent))
			},
		},
	}
}

// multimodalSignalDispatchers covers the families that also read the row's image attachment
// and share the request-scoped image embedding cache.
func (c *Classifier) multimodalSignalDispatchers() []signalDispatch {
	return []signalDispatch{
		{
			signalType: config.SignalTypeEmbedding,
			name:       "Embedding",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateEmbeddingSignal(row.results, &row.mu, row.textForSignal(config.SignalTypeEmbedding), row.input.imageURL, row.imgCache)
			},
			evaluateBatch: c.evaluateEmbeddingSignalsBatch,
		},
		{
			signalType: config.SignalTypeComplexity,
			name:       "Complexity",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateComplexitySignal(row.results, &row.mu, row.textForSignal(config.SignalTypeComplexity), row.input.imageURL, row.imgCache)
			},
			evaluateBatch: c.evaluateComplexitySignalsBatch,
		},
	}
}

// conversationStateSignalDispatchers covers the families evaluated from state carried
// alongside the current turn rather than from the turn's own text.
func (c *Classifier) conversationStateSignalDispatchers() []signalDispatch {
	return []signalDispatch{
		{
			signalType: config.SignalTypeUserFeedback,
			name:       "User feedback",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateUserFeedbackSignal(
					row.results,
					&row.mu,
					row.textForSignal(config.SignalTypeUserFeedback),
					row.input.hasPriorAssistantReply,
				)
			},
		},
		{
			signalType: config.SignalTypeReask,
			name:       "Reask",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateReaskSignal(row.results, &row.mu, row.input.currentUserText, row.input.priorUserMessages)
			},
		},
		{
			signalType: config.SignalTypeContext,
			name:       "Context",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateContextSignal(row.results, &row.mu, row.input.contextText)
			},
		},
		{
			signalType: config.SignalTypeConversation,
			name:       "Conversation",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateConversationSignal(row.results, &row.mu, row.input.convFacts)
			},
		},
	}
}

// historyAwareSignalDispatchers covers the safety families that score the current text
// against the flattened conversation history.
func (c *Classifier) historyAwareSignalDispatchers() []signalDispatch {
	return []signalDispatch{
		{
			signalType: config.SignalTypeJailbreak,
			name:       "Jailbreak",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateJailbreakSignal(row.results, &row.mu, row.textForSignal(config.SignalTypeJailbreak), historyForHistoryAwareSignals(row.input.priorUserMessages, row.input.nonUserMessages))
			},
		},
		{
			signalType: config.SignalTypePII,
			name:       "PII",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluatePIISignal(row.results, &row.mu, row.textForSignal(config.SignalTypePII), historyForHistoryAwareSignals(row.input.priorUserMessages, row.input.nonUserMessages))
			},
		},
	}
}

func runSignalDispatchers(dispatchers []signalDispatch, rows []*signalEvaluationRow, ready map[string]bool, batchModels bool, wg *sync.WaitGroup) {
	for _, d := range dispatchers {
		eligible := make([]*signalEvaluationRow, 0, len(rows))
		unused := 0
		for _, row := range rows {
			if !isSignalTypeUsed(row.usedSignals, d.signalType) {
				unused++
				continue
			}
			if ready[d.signalType] {
				eligible = append(eligible, row)
			}
		}
		// One line per dispatcher, not per row: a wave carries up to 32 rows and the
		// message identifies only the signal, so logging inside the row loop repeated
		// the same text 32 times and allocated an args slice for each.
		if unused > 0 {
			logging.Debugf("[Signal Computation] %s signal not used in any decision for %d of %d rows, skipping evaluation", d.name, unused, len(rows))
		}
		if len(eligible) == 0 {
			continue
		}
		wg.Add(1)
		go func(dispatch signalDispatch, signalRows []*signalEvaluationRow) {
			defer wg.Done()
			if batchModels && dispatch.evaluateBatch != nil {
				dispatch.evaluateBatch(signalRows)
				return
			}
			for _, row := range signalRows {
				dispatch.evaluate(row)
			}
		}(d, eligible)
	}
}
