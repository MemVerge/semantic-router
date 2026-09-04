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

func (c *Classifier) buildSignalDispatchers() []signalDispatch {
	return []signalDispatch{
		{
			signalType: config.SignalTypeKeyword,
			name:       "Keyword",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateKeywordSignal(row.results, &row.mu, row.textForSignal(config.SignalTypeKeyword))
			},
		},
		{
			signalType: config.SignalTypeEmbedding,
			name:       "Embedding",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateEmbeddingSignal(row.results, &row.mu, row.textForSignal(config.SignalTypeEmbedding), row.input.imageURL, row.imgCache)
			},
			evaluateBatch: c.evaluateEmbeddingSignalsBatch,
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
			signalType: config.SignalTypeContext,
			name:       "Context",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateContextSignal(row.results, &row.mu, row.input.contextText)
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
			signalType: config.SignalTypeComplexity,
			name:       "Complexity",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateComplexitySignal(row.results, &row.mu, row.textForSignal(config.SignalTypeComplexity), row.input.imageURL, row.imgCache)
			},
			evaluateBatch: c.evaluateComplexitySignalsBatch,
		},
		{
			signalType: config.SignalTypeModality,
			name:       "Modality",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateModalitySignal(row.results, &row.mu, row.textForSignal(config.SignalTypeModality))
			},
		},
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
		{
			signalType: config.SignalTypeKB,
			name:       "KB",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateKBSignals(row.results, &row.mu, row.textForSignal(config.SignalTypeKB))
			},
		},
		{
			signalType: config.SignalTypeConversation,
			name:       "Conversation",
			evaluate: func(row *signalEvaluationRow) {
				c.evaluateConversationSignal(row.results, &row.mu, row.input.convFacts)
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

func runSignalDispatchers(dispatchers []signalDispatch, rows []*signalEvaluationRow, ready map[string]bool, batchModels bool, wg *sync.WaitGroup) {
	for _, d := range dispatchers {
		eligible := make([]*signalEvaluationRow, 0, len(rows))
		for _, row := range rows {
			if isSignalTypeUsed(row.usedSignals, d.signalType) && ready[d.signalType] {
				eligible = append(eligible, row)
			} else if !isSignalTypeUsed(row.usedSignals, d.signalType) {
				logging.Debugf("[Signal Computation] %s signal not used in any decision, skipping evaluation", d.name)
			}
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
