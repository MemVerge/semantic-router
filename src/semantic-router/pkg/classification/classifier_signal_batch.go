package classification

import (
	"sync"
	"time"
)

const (
	signalBatchWindow  = 100 * time.Millisecond
	signalBatchMaxSize = 32
)

type signalEvaluationInput struct {
	text                   string
	contextText            string
	currentUserText        string
	priorUserMessages      []string
	nonUserMessages        []string
	hasPriorAssistantReply bool
	forceEvaluateAll       bool
	uncompressedText       string
	skipCompressionSignals map[string]bool
	convFacts              ConversationFacts
	imageURL               string
}

type signalBatchEntry struct {
	input  signalEvaluationInput
	result chan *SignalResults
}

type signalBatchCollector struct {
	mu         sync.Mutex
	generation uint64
	current    []*signalBatchEntry
	timer      *time.Timer
	evaluate   func([]signalEvaluationInput) []*SignalResults
}

func newSignalBatchCollector(evaluate func([]signalEvaluationInput) []*SignalResults) *signalBatchCollector {
	return &signalBatchCollector{evaluate: evaluate}
}

func (c *signalBatchCollector) enqueue(input signalEvaluationInput) <-chan *SignalResults {
	entry := &signalBatchEntry{
		input:  input,
		result: make(chan *SignalResults, 1),
	}

	c.mu.Lock()
	if len(c.current) == 0 {
		c.generation++
		generation := c.generation
		c.timer = time.AfterFunc(signalBatchWindow, func() {
			c.flush(generation)
		})
	}
	c.current = append(c.current, entry)
	var batch []*signalBatchEntry
	if len(c.current) == signalBatchMaxSize {
		batch = c.detachLocked()
	}
	c.mu.Unlock()

	if len(batch) > 0 {
		go c.evaluateBatch(batch)
	}
	return entry.result
}

func (c *signalBatchCollector) flush(generation uint64) {
	c.mu.Lock()
	if generation != c.generation || len(c.current) == 0 {
		c.mu.Unlock()
		return
	}
	batch := c.detachLocked()
	c.mu.Unlock()
	c.evaluateBatch(batch)
}

func (c *signalBatchCollector) detachLocked() []*signalBatchEntry {
	batch := c.current
	c.current = nil
	if c.timer != nil {
		c.timer.Stop()
		c.timer = nil
	}
	return batch
}

func (c *signalBatchCollector) evaluateBatch(batch []*signalBatchEntry) {
	inputs := make([]signalEvaluationInput, len(batch))
	for i, entry := range batch {
		inputs[i] = entry.input
	}
	results := c.evaluate(inputs)
	if len(results) != len(batch) {
		results = make([]*SignalResults, len(batch))
		for i := range results {
			results[i] = newSignalResults()
		}
	}
	for i, entry := range batch {
		entry.result <- results[i]
	}
}

func newSignalResults() *SignalResults {
	return &SignalResults{
		Metrics:           &SignalMetricsCollection{},
		SignalConfidences: make(map[string]float64),
		SignalValues:      make(map[string]float64),
	}
}
