package extproc

import (
	"encoding/base64"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// memoryAugmentedBody is the shape MemBox sends: the last user message is
// the typed turn wrapped in retrieved memories, which is exactly what the
// router must NOT treat as the current user message.
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

func TestApplyCurrentMessageHeader_ReplacesOnlyUserContent(t *testing.T) {
	fast, err := extractContentFast([]byte(memoryAugmentedBody))
	require.NoError(t, err)
	before := *fast

	ctx := ctxWithHeaders(map[string]string{headers.MemBoxCurrentMessage: b64("where should I stay?")})
	newRouterWithCurrentMessageHeader(true).applyCurrentMessageHeader(fast, ctx)

	assert.True(t, ctx.CurrentMessageFromHeader)
	assert.Equal(t, "where should I stay?", fast.UserContent)
	assert.Equal(t, before.PriorUserMessages, fast.PriorUserMessages, "prior turns describe the body as sent")
	assert.Equal(t, before.NonUserMessages, fast.NonUserMessages)
	assert.Equal(t, before.UserMessageCount, fast.UserMessageCount)
	assert.Equal(t, before.AssistantMessageCount, fast.AssistantMessageCount)
	assert.Equal(t, before.LastMessageRole, fast.LastMessageRole)
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
	fast, err := router.extractFastRequestState([]byte(memoryAugmentedBody), openAICtx)
	require.NoError(t, err)
	assert.Equal(t, "where should I stay?", fast.UserContent)
	assert.Equal(t, []string{"What did I plan last week?"}, fast.PriorUserMessages)

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
	withHeader := func(method, path string, extra ...*core.HeaderValue) *ext_proc.ProcessingRequest_RequestHeaders {
		rh := newRequestHeaders(method, path)
		rh.RequestHeaders.Headers.Headers = append(rh.RequestHeaders.Headers.Headers,
			&core.HeaderValue{Key: headers.MemBoxCurrentMessage, Value: b64("where should I stay?")})
		rh.RequestHeaders.Headers.Headers = append(rh.RequestHeaders.Headers.Headers, extra...)
		return rh
	}

	cases := []struct {
		name   string
		router *OpenAIRouter
		req    *ext_proc.ProcessingRequest_RequestHeaders
	}{
		{
			name:   "routed chat completions",
			router: &OpenAIRouter{},
			req:    withHeader("POST", "/v1/chat/completions"),
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
			assert.Equal(t, b64("where should I stay?"), ctx.Headers[headers.MemBoxCurrentMessage])

			// ...and removed from what goes upstream.
			mutation := response.GetRequestHeaders().GetResponse().GetHeaderMutation()
			require.NotNil(t, mutation, "expected a header mutation carrying the strip")
			assert.Contains(t, mutation.GetRemoveHeaders(), headers.MemBoxCurrentMessage)
		})
	}
}
