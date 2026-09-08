package extproc

import (
	"encoding/base64"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// currentMessageHeaderEnabled reports whether x-membox-current-message is
// honored. Like x-vsr-skip-processing it is gated by deployment config
// (global.router.current_message_header.enabled, default off): the header
// lets a caller choose the text every request signal evaluates — including
// jailbreak and PII detection — while the body the model receives stays as
// sent, so only deployments whose clients are trusted to fill it turn it on.
func (r *OpenAIRouter) currentMessageHeaderEnabled() bool {
	if r == nil || r.Config == nil {
		return false
	}
	return r.Config.CurrentMessageHeader.IsEnabled()
}

// applyCurrentMessageHeader lets a client name the current user turn
// explicitly instead of having the router infer it from the last "user"
// message in the body.
//
// Clients that build the request body from a memory-augmented conversation
// (MemBox prepends retrieved memories and prior turns into the messages it
// sends) end up with a last user message the router should not treat as the
// user's own words: every signal that reads UserContent — decision
// evaluation, RAG lookup, tool selection, modality detection — then keys off
// injected context rather than the turn the user typed. The client already
// holds the verbatim turn, so it sends it in x-membox-current-message
// (base64 of the UTF-8 text) and the router uses that as UserContent.
//
// UserContent becomes the header text and the body-derived last user
// message is demoted instead of dropped, so it stays reachable by the
// history-aware signals (jailbreak and PII read PriorUserMessages +
// NonUserMessages): on MemBox's request shape it is a runtime-state block, or
// on the remote-visitor assistant the retrieved memories and uploaded
// documents, and it must not vanish from every signal just because it is no
// longer "current". Unlike the body walker, which demotes a superseded user
// message to PriorUserMessages, this demotes to NonUserMessages: the block is
// not the user's words, so it should not enter the reask signal's prior-turn
// comparison. A body whose last user message already equals the header text
// is left alone, so the turn is not counted twice in the context text.
// PriorUserMessages and the conversation-shape counts still describe the body
// as sent, because that is what the upstream model will see; the semantic
// cache keys off the body for the same reason. A missing, empty, undecodable or non-UTF-8 header
// leaves the body-derived extraction untouched, so a client that never sends
// the header, or sends a bad one, gets today's behaviour.
func (r *OpenAIRouter) applyCurrentMessageHeader(fast *FastExtractResult, ctx *RequestContext) {
	if fast == nil || ctx == nil || !r.currentMessageHeaderEnabled() {
		return
	}
	raw := headerValueCI(ctx, headers.MemBoxCurrentMessage)
	if strings.TrimSpace(raw) == "" {
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
		"header_bytes":    len(text),
		"body_user_bytes": len(fast.UserContent),
		"body_user_count": fast.UserMessageCount,
	})
	if fast.UserContent != "" && fast.UserContent != text {
		fast.NonUserMessages = append(fast.NonUserMessages, fast.UserContent)
	}
	fast.UserContent = text
	ctx.CurrentMessageFromHeader = true
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
