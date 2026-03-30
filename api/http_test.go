package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

type mockGenerator struct {
	response string
	err      error
	prompt   string
}

func (m *mockGenerator) Generate(prompt string) (string, error) {
	m.prompt = prompt
	if m.err != nil {
		return "", m.err
	}
	return m.response, nil
}

func TestHTTPServerGenerate(t *testing.T) {
	tests := []struct {
		name             string
		body             string
		mock             *mockGenerator
		expectedStatus   int
		expectedPrompt   string
		expectResponse   bool
		expectedErrorMsg string
	}{
		{
			name:           "success",
			body:           `{"prompt":"Explain Scrum roles"}`,
			mock:           &mockGenerator{response: "Scrum Master, Product Owner, Developers"},
			expectedStatus: http.StatusOK,
			expectedPrompt: "Explain Scrum roles",
			expectResponse: true,
		},
		{
			name:             "invalid json",
			body:             "{",
			mock:             &mockGenerator{},
			expectedStatus:   http.StatusBadRequest,
			expectedErrorMsg: "invalid json",
		},
		{
			name:             "empty prompt",
			body:             `{"prompt":"   "}`,
			mock:             &mockGenerator{},
			expectedStatus:   http.StatusBadRequest,
			expectedErrorMsg: "prompt is required",
		},
		{
			name:             "llm error",
			body:             `{"prompt":"Explain Scrum roles"}`,
			mock:             &mockGenerator{err: errors.New("inference unavailable")},
			expectedStatus:   http.StatusInternalServerError,
			expectedErrorMsg: "inference unavailable",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := NewHTTPServer(tc.mock)
			req := httptest.NewRequest(http.MethodPost, "/generate", bytes.NewBufferString(tc.body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()

			server.Handler().ServeHTTP(rr, req)

			if rr.Code != tc.expectedStatus {
				t.Fatalf("expected status %d, got %d", tc.expectedStatus, rr.Code)
			}

			if tc.expectedPrompt != "" && tc.mock.prompt != tc.expectedPrompt {
				t.Fatalf("expected prompt %q, got %q", tc.expectedPrompt, tc.mock.prompt)
			}

			var payload map[string]string
			if err := json.Unmarshal(rr.Body.Bytes(), &payload); err != nil {
				t.Fatalf("failed to decode response: %v", err)
			}

			if tc.expectResponse && payload["response"] == "" {
				t.Fatalf("expected non-empty response field")
			}

			if tc.expectedErrorMsg != "" && payload["error"] != tc.expectedErrorMsg {
				t.Fatalf("expected error %q, got %q", tc.expectedErrorMsg, payload["error"])
			}
		})
	}
}
