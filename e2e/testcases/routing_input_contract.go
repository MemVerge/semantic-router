package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("routing-input-contract", pkgtestcases.TestCase{
		Description: "Verify single-user Input routing and context accounting before signal compression",
		Tags:        []string{"kubernetes", "routing", "context"},
		Fn:          testRoutingInputContract,
	})
}

func testRoutingInputContract(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	return runRoutingInputContract(ctx, localPort)
}

func runRoutingInputContract(ctx context.Context, localPort string) error {
	const marker = "__ROUTING_INPUT_PROBE__"
	// Exceed the former 30 KiB header limit. Multiple sentences exercise actual
	// compression; the trailing marker also catches truncation of the Input.
	const longInputBytes = 30*1024 + 1
	filler := strings.Repeat("The orchard has green leaves. ", 1100)
	longInput := filler[:longInputBytes-len(marker)-1] + " " + marker
	cases := []struct {
		name     string
		input    string
		decision string
		context  string
	}{
		{"short", "Please summarize this input. " + marker, "input_short_decision", "input_short"},
		{"compressed", longInput, "input_long_decision", "input_long"},
	}
	for _, tc := range cases {
		// The existing helper sends exactly one user message and requests the
		// debug response headers from the deployed ext_proc path.
		response, err := sendLocalChatCompletion(ctx, localPort, "MoM", tc.input, 60*time.Second)
		if err != nil {
			return fmt.Errorf("%s Input: %w", tc.name, err)
		}
		if response.StatusCode != http.StatusOK {
			return fmt.Errorf("%s Input: %s", tc.name, formatUnexpectedChatCompletionStatus(response))
		}

		// Context uses the router's byte-based ceil(len/4) estimate on the full
		// Input. The long case must retain 7681 tokens despite the profile's
		// 128-token classification compression budget.
		expected := []struct {
			header string
			value  string
		}{
			{"x-vsr-selected-decision", tc.decision},
			{"x-vsr-selected-model", "base-model"},
			{"x-vsr-matched-keywords", "input_probe"},
			{"x-vsr-matched-context", tc.context},
			{"x-vsr-context-token-count", strconv.Itoa((len(tc.input) + 3) / 4)},
		}
		for _, want := range expected {
			if got := response.Headers.Get(want.header); got != want.value {
				return fmt.Errorf("%s Input: %s = %q, want %q", tc.name, want.header, got, want.value)
			}
		}
	}
	return nil
}
