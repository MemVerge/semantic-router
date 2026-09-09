package extproc

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// The MemBox shapes, read from agent/resources/assistants/*.yaml: the
// assembler pushes one message per template entry and never merges adjacent
// same-role messages, so the typed turn, the retrieved memories and the
// runtime state arrive as SEPARATE adjacent "user" messages. Body extraction
// therefore lands UserContent on whichever block is LAST — a timestamp, a
// locale and a connector list for the local assistants — while the typed turn
// sits in PriorUserMessages, which is exactly what the header exists to fix.

const (
	memboxTypedTurn    = "<current_user_request>where should I stay?</current_user_request>"
	memboxMemoryBlock  = "<memorybox_context>Potentially relevant context for this turn, not the current user request. trip to Kyoto in March; budget 2k</memorybox_context>"
	memboxRuntimeBlock = `<runtime_capability_state non_actionable="true">current_time: 2026-09-08T16:00-07:00; locale: en-US; connectors: none</runtime_capability_state>`
)

// memboxChatBody marshals role/content pairs into an OpenAI request body, so the
// fixtures can carry the blocks verbatim (the runtime block has quotes).
func memboxChatBody(messages ...[2]string) string {
	msgs := make([]map[string]string, 0, len(messages))
	for _, m := range messages {
		msgs = append(msgs, map[string]string{"role": m[0], "content": m[1]})
	}
	body, err := json.Marshal(map[string]interface{}{"model": "auto", "messages": msgs})
	if err != nil {
		panic(err)
	}
	return string(body)
}

// memboxLocalChatBody is local_agent_chat / default_idea / onboarding_chat:
// transcript, then current_user_request, memorybox_context, runtime state.
var memboxLocalChatBody = memboxChatBody(
	[2]string{"system", "You are a helpful assistant."},
	[2]string{"user", "<current_user_request>What did I plan last week?</current_user_request>"},
	[2]string{"assistant", "You planned a trip."},
	[2]string{"user", memboxTypedTurn},
	[2]string{"user", memboxMemoryBlock},
	[2]string{"user", memboxRuntimeBlock},
)

// memboxRemoteChatBody is remote_agent_chat, the untrusted-visitor path: no
// runtime block, so the LAST user message is the retrieved memories and
// uploaded documents.
var memboxRemoteChatBody = memboxChatBody(
	[2]string{"system", "You are a helpful assistant."},
	[2]string{"user", memboxTypedTurn},
	[2]string{"user", memboxMemoryBlock},
)

// memoryAugmentedBody is the other client shape the header serves: a single
// last user message that wraps the typed turn in retrieved memories.
const memoryAugmentedBody = `{
	"model": "auto",
	"messages": [
		{"role": "system", "content": "You are a helpful assistant."},
		{"role": "user", "content": "What did I plan last week?"},
		{"role": "assistant", "content": "You planned a trip."},
		{"role": "user", "content": "<memories>trip to Kyoto in March; budget 2k</memories>\n\nwhere should I stay?"}
	]
}`

func b64(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }

func ctxWithHeaders(h map[string]string) *RequestContext {
	return &RequestContext{Headers: h, RequestID: "req-1"}
}

func newRouterWithCurrentMessageHeader(enabled bool) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterOptions: config.RouterOptions{
				CurrentMessageHeader: config.CurrentMessageHeaderConfig{Enabled: enabled},
			},
		},
	}
}

func TestApplyCurrentMessageHeader_DisabledByDefaultIgnoresHeader(t *testing.T) {
	for name, router := range map[string]*OpenAIRouter{
		"gate off":   newRouterWithCurrentMessageHeader(false),
		"no config":  {},
		"nil router": nil,
	} {
		t.Run(name, func(t *testing.T) {
			fast, err := extractContentFast([]byte(memoryAugmentedBody))
			require.NoError(t, err)
			want := fast.UserContent
			ctx := ctxWithHeaders(map[string]string{headers.MemBoxCurrentMessage: b64("where should I stay?")})
			router.applyCurrentMessageHeader(fast, ctx)
			assert.Equal(t, want, fast.UserContent)
			assert.False(t, ctx.CurrentMessageFromHeader)
		})
	}
}

// The body walker lands UserContent on the LAST user message; the header
// replaces it and touches nothing else. The context text is the whole body
// as sent (AllBodyText), so the displaced block still counts toward
// long_context without being demoted anywhere.
func TestApplyCurrentMessageHeader_ReplacesUserContentOnly(t *testing.T) {
	cases := map[string]struct {
		body      string
		displaced string
		priorTurn string
	}{
		"membox local chat: runtime block is last":    {memboxLocalChatBody, memboxRuntimeBlock, "What did I plan last week?"},
		"membox remote chat: memories are last":       {memboxRemoteChatBody, memboxMemoryBlock, ""},
		"single wrapped message: the wrapper is last": {memoryAugmentedBody, "<memories>trip to Kyoto in March; budget 2k</memories>\n\nwhere should I stay?", "What did I plan last week?"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			fast, err := extractContentFast([]byte(tc.body))
			require.NoError(t, err)
			require.Equal(t, tc.displaced, fast.UserContent, "fixture: body extraction lands on the last user message")
			before := *fast

			ctx := ctxWithHeaders(map[string]string{headers.MemBoxCurrentMessage: b64("where should I stay?")})
			newRouterWithCurrentMessageHeader(true).applyCurrentMessageHeader(fast, ctx)

			assert.True(t, ctx.CurrentMessageFromHeader)
			assert.Equal(t, "where should I stay?", fast.UserContent)
			assert.Equal(t, before.NonUserMessages, fast.NonUserMessages, "nothing is demoted")
			assert.Equal(t, before.PriorUserMessages, fast.PriorUserMessages, "prior turns describe the body as sent")
			assert.Equal(t, before.AllBodyText, fast.AllBodyText, "the body text is never overridden")
			assert.Equal(t, before.UserMessageCount, fast.UserMessageCount)
			assert.Equal(t, before.AssistantMessageCount, fast.AssistantMessageCount)
			assert.Equal(t, before.LastMessageRole, fast.LastMessageRole)

			input := newRouterWithCurrentMessageHeader(true).prepareSignalEvaluationInput(signalConversationHistoryFromFastExtract(fast))
			assert.Equal(t, "where should I stay?", input.evaluationText, "the classifiers score the header text only")
			assert.Equal(t, "where should I stay?", input.currentUserText)
			assert.Contains(t, input.allMessagesText, tc.displaced, "the displaced block still counts as context")
			if tc.priorTurn != "" {
				assert.Contains(t, input.allMessagesText, tc.priorTurn, "prior user turns count as context")
			}
		})
	}
}

// AllBodyText is the context-count input: one entry per message with text,
// in body order, every role — on both inbound protocols, so a tool result
// counts the same whether it arrives as a tool-role message (OpenAI) or an
// inline tool_result block (Anthropic).
func TestExtractContentFast_AllBodyTextCountsEveryMessageOnBothProtocols(t *testing.T) {
	openAIBody := []byte(`{
		"model": "auto",
		"messages": [
			{"role": "system", "content": "system prompt"},
			{"role": "user", "content": "first question"},
			{"role": "assistant", "content": "calling a tool", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
			{"role": "tool", "tool_call_id": "c1", "content": "tool output"},
			{"role": "user", "content": [{"type": "text", "text": "second question"}]}
		]
	}`)
	fast, err := extractContentFast(openAIBody)
	require.NoError(t, err)
	assert.Equal(t, []string{"system prompt", "first question", "calling a tool", "tool output", "second question"}, fast.AllBodyText)
	assert.Equal(t, "second question", fast.UserContent)
	assert.Equal(t, []string{"first question"}, fast.PriorUserMessages)
	assert.Equal(t, []string{"system prompt", "calling a tool"}, fast.NonUserMessages, "tool output enters no signal slice")

	anthropicBody := []byte(`{
		"model": "claude-sonnet-4-5",
		"max_tokens": 64,
		"system": [{"type": "text", "text": "system prompt"}],
		"messages": [
			{"role": "user", "content": "first question"},
			{"role": "assistant", "content": [{"type": "text", "text": "calling a tool"}, {"type": "tool_use", "id": "tu_1", "name": "lookup", "input": {}}]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "tool output"}, {"type": "text", "text": "second question"}]}
		]
	}`)
	fast, err = extractContentFastAnthropic(anthropicBody)
	require.NoError(t, err)
	assert.Equal(t, []string{"system prompt", "first question", "calling a tool", "tool output second question"}, fast.AllBodyText)
	assert.Equal(t, "second question", fast.UserContent)
}

// The whole-body context count is not behind the header gate: with the gate
// off and no header, evaluation stays on the body's last user message (today's
// routing) while the context text now includes prior user turns, which the
// nonUser+current join used to leave out.
func TestPrepareSignalEvaluationInput_ContextTextIsTheWholeBodyWithoutTheHeader(t *testing.T) {
	fast, err := extractContentFast([]byte(memboxLocalChatBody))
	require.NoError(t, err)
	ctx := ctxWithHeaders(map[string]string{})
	newRouterWithCurrentMessageHeader(false).applyCurrentMessageHeader(fast, ctx)
	assert.False(t, ctx.CurrentMessageFromHeader)

	input := newRouterWithCurrentMessageHeader(false).prepareSignalEvaluationInput(signalConversationHistoryFromFastExtract(fast))
	assert.Equal(t, memboxRuntimeBlock, input.evaluationText, "gate off: the body's last user message is still what gets scored")
	assert.Equal(t, strings.Join(fast.AllBodyText, " "), input.allMessagesText)
	assert.Contains(t, input.allMessagesText, "What did I plan last week?", "prior user turns now count")
	assert.Contains(t, input.allMessagesText, memboxTypedTurn)
	assert.Contains(t, input.allMessagesText, memboxMemoryBlock)
}

func TestApplyCurrentMessageHeader_AbsentKeepsBodyExtraction(t *testing.T) {
	fast, err := extractContentFast([]byte(memoryAugmentedBody))
	require.NoError(t, err)
	want := fast.UserContent

	newRouterWithCurrentMessageHeader(true).applyCurrentMessageHeader(fast, ctxWithHeaders(map[string]string{}))
	assert.Equal(t, want, fast.UserContent)

	newRouterWithCurrentMessageHeader(true).applyCurrentMessageHeader(fast, ctxWithHeaders(nil))
	assert.Equal(t, want, fast.UserContent)
}

func TestApplyCurrentMessageHeader_BadValuesAreIgnored(t *testing.T) {
	cases := map[string]string{
		"empty":         "",
		"whitespace":    "   ",
		"not base64":    "where should I stay?",
		"decodes empty": b64(""),
		"invalid utf-8": base64.StdEncoding.EncodeToString([]byte{0xff, 0xfe, 0x41}),
	}
	for name, value := range cases {
		t.Run(name, func(t *testing.T) {
			fast, err := extractContentFast([]byte(memoryAugmentedBody))
			require.NoError(t, err)
			want := fast.UserContent
			newRouterWithCurrentMessageHeader(true).applyCurrentMessageHeader(fast, ctxWithHeaders(map[string]string{headers.MemBoxCurrentMessage: value}))
			assert.Equal(t, want, fast.UserContent)
		})
	}
}

func TestApplyCurrentMessageHeader_AcceptsUnpaddedAndMixedCaseName(t *testing.T) {
	fast, err := extractContentFast([]byte(memoryAugmentedBody))
	require.NoError(t, err)

	unpadded := base64.RawStdEncoding.EncodeToString([]byte("多轮对话：住哪里？\nsecond line"))
	ctx := ctxWithHeaders(map[string]string{"X-MemBox-Current-Message": unpadded})
	newRouterWithCurrentMessageHeader(true).applyCurrentMessageHeader(fast, ctx)

	assert.Equal(t, "多轮对话：住哪里？\nsecond line", fast.UserContent)
}

func TestExtractFastRequestState_HeaderOverridesBothProtocols(t *testing.T) {
	router := newRouterWithCurrentMessageHeader(true)
	header := map[string]string{headers.MemBoxCurrentMessage: b64("where should I stay?")}

	openAICtx := ctxWithHeaders(header)
	fast, err := router.extractFastRequestState([]byte(memboxLocalChatBody), openAICtx)
	require.NoError(t, err)
	assert.Equal(t, "where should I stay?", fast.UserContent)
	assert.Equal(t, []string{
		"<current_user_request>What did I plan last week?</current_user_request>",
		memboxTypedTurn,
		memboxMemoryBlock,
	}, fast.PriorUserMessages, "the body's user messages, as sent")
	assert.Equal(t, []string{"You are a helpful assistant.", "You planned a trip."}, fast.NonUserMessages, "the displaced block is not demoted")

	anthropicBody := []byte(`{
		"model": "claude-sonnet-4-5",
		"max_tokens": 64,
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "<memories>trip to Kyoto</memories>\n\nwhere should I stay?"}]}
		]
	}`)
	anthropicCtx := ctxWithHeaders(header)
	anthropicCtx.ClientProtocol = config.ClientProtocolAnthropic
	fast, err = router.extractFastRequestState(anthropicBody, anthropicCtx)
	require.NoError(t, err)
	assert.Equal(t, "where should I stay?", fast.UserContent)
}

// The header is addressed to the router alone; it must be removed before the
// request reaches the upstream provider on every path that continues, not
// only the routed one.
func TestHandleRequestHeaders_StripsCurrentMessageHeaderOnEveryContinuePath(t *testing.T) {
	withHeaderNamed := func(name, method, path string, extra ...*core.HeaderValue) *ext_proc.ProcessingRequest_RequestHeaders {
		rh := newRequestHeaders(method, path)
		rh.RequestHeaders.Headers.Headers = append(rh.RequestHeaders.Headers.Headers,
			&core.HeaderValue{Key: name, Value: b64("where should I stay?")})
		rh.RequestHeaders.Headers.Headers = append(rh.RequestHeaders.Headers.Headers, extra...)
		return rh
	}
	withHeader := func(method, path string, extra ...*core.HeaderValue) *ext_proc.ProcessingRequest_RequestHeaders {
		return withHeaderNamed(headers.MemBoxCurrentMessage, method, path, extra...)
	}

	cases := []struct {
		name   string
		router *OpenAIRouter
		req    *ext_proc.ProcessingRequest_RequestHeaders
		sentAs string
	}{
		{
			name:   "routed chat completions",
			router: &OpenAIRouter{},
			req:    withHeader("POST", "/v1/chat/completions"),
		},
		{
			// The body phase reads the header case-insensitively, so the
			// strip must find it under any casing too or it leaks upstream.
			name:   "mixed-case header name",
			router: &OpenAIRouter{},
			req:    withHeaderNamed("X-MemBox-Current-Message", "POST", "/v1/chat/completions"),
			sentAs: "X-MemBox-Current-Message",
		},
		{
			name:   "anthropic messages",
			router: &OpenAIRouter{},
			req:    withHeader("POST", "/v1/messages"),
		},
		{
			name:   "skip-processing opt-out",
			router: newRouterWithSkipProcessingGate(true),
			req: withHeader("POST", "/v1/chat/completions",
				&core.HeaderValue{Key: headers.VSRSkipProcessing, Value: "true"}),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: make(map[string]string)}
			response, err := tc.router.handleRequestHeaders(tc.req, ctx)
			require.NoError(t, err)
			require.NotNil(t, response.GetRequestHeaders(), "expected a continue-headers response")

			// Captured for the body phase...
			sentAs := tc.sentAs
			if sentAs == "" {
				sentAs = headers.MemBoxCurrentMessage
			}
			assert.Equal(t, b64("where should I stay?"), ctx.Headers[sentAs])
			assert.Equal(t, b64("where should I stay?"), headerValueCI(ctx, headers.MemBoxCurrentMessage))

			// ...and removed from what goes upstream.
			mutation := response.GetRequestHeaders().GetResponse().GetHeaderMutation()
			require.NotNil(t, mutation, "expected a header mutation carrying the strip")
			assert.Contains(t, mutation.GetRemoveHeaders(), headers.MemBoxCurrentMessage)
		})
	}
}

// The strip only allocates when there is something to strip: the
// skip-processing fast path exists to do nothing, and a request without the
// header must keep its plain CONTINUE.
func TestHandleRequestHeaders_SkipProcessingWithoutHeaderStaysAPlainContinue(t *testing.T) {
	rh := newRequestHeaders("POST", "/v1/chat/completions")
	rh.RequestHeaders.Headers.Headers = append(rh.RequestHeaders.Headers.Headers,
		&core.HeaderValue{Key: headers.VSRSkipProcessing, Value: "true"})

	ctx := &RequestContext{Headers: make(map[string]string)}
	response, err := newRouterWithSkipProcessingGate(true).handleRequestHeaders(rh, ctx)
	require.NoError(t, err)
	require.True(t, ctx.SkipProcessing)
	require.NotNil(t, response.GetRequestHeaders())
	assert.Nil(t, response.GetRequestHeaders().GetResponse().GetHeaderMutation(), "nothing to strip, nothing to mutate")
}
