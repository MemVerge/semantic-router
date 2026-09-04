package extproc

import (
	"encoding/base64"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// applyCurrentMessageHeader lets a client name the current user turn
// explicitly instead of having the router infer it from the last "user"
// message in the body.
//
// Clients that build the request body from a memory-augmented conversation
// (MemBox prepends retrieved memories and prior turns into the messages it
// sends) end up with a last user message the router should not treat as the
// user's own words: every signal that reads UserContent — decision
// evaluation, RAG lookup, the semantic-cache key, modality detection — then
// keys off injected context rather than the turn the user typed. The client
// already holds the verbatim turn, so it sends it in
// x-membox-current-message (base64 of the UTF-8 text) and the router uses
// that as UserContent.
//
// Only UserContent is replaced. PriorUserMessages, NonUserMessages and the
// conversation-shape counts still describe the body as sent, because that is
// what the upstream model will see. A missing, empty, undecodable or
// non-UTF-8 header leaves the body-derived extraction untouched, so a client
// that never sends the header, or sends a bad one, gets today's behaviour.
func applyCurrentMessageHeader(fast *FastExtractResult, ctx *RequestContext) {
	if fast == nil || ctx == nil {
		return
	}
	raw, ok := lookupHeaderFold(ctx.Headers, headers.MemBoxCurrentMessage)
	if !ok || strings.TrimSpace(raw) == "" {
		return
	}
	text, err := decodeCurrentMessageHeader(raw)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "current_message_header_ignored", map[string]interface{}{
			"request_id": ctx.RequestID,
			"header":     headers.MemBoxCurrentMessage,
			"reason":     err.Error(),
		})
		return
	}
	if text == "" {
		return
	}
	logging.ComponentDebugEvent("extproc", "current_message_header_applied", map[string]interface{}{
		"request_id":      ctx.RequestID,
		"header_chars":    len(text),
		"body_user_chars": len(fast.UserContent),
		"body_user_count": fast.UserMessageCount,
	})
	fast.UserContent = text
}

// decodeCurrentMessageHeader decodes the base64 (standard alphabet, padding
// optional) header value into the UTF-8 text it carries.
func decodeCurrentMessageHeader(raw string) (string, error) {
	value := strings.TrimSpace(raw)
	decoded, err := base64.StdEncoding.DecodeString(value)
	if err != nil {
		decoded, err = base64.RawStdEncoding.DecodeString(value)
	}
	if err != nil {
		return "", &currentMessageHeaderError{reason: "invalid base64"}
	}
	if !utf8.Valid(decoded) {
		return "", &currentMessageHeaderError{reason: "not valid UTF-8"}
	}
	return string(decoded), nil
}

type currentMessageHeaderError struct{ reason string }

func (e *currentMessageHeaderError) Error() string { return e.reason }

// lookupHeaderFold finds a header by case-insensitive name. HTTP/2 lowercases
// header names on the wire, but ctx.Headers stores keys as received, so an
// HTTP/1.1 client or a test fixture may carry mixed case.
func lookupHeaderFold(h map[string]string, name string) (string, bool) {
	if v, ok := h[name]; ok {
		return v, true
	}
	for k, v := range h {
		if strings.EqualFold(k, name) {
			return v, true
		}
	}
	return "", false
}
