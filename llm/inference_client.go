package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type HTTPDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

type InferenceHTTPClient struct {
	baseURL    string
	httpClient HTTPDoer
}

func NewInferenceHTTPClient(baseURL string, httpClient HTTPDoer) *InferenceHTTPClient {
	return &InferenceHTTPClient{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: httpClient,
	}
}

type inferenceGenerateRequest struct {
	Message string `json:"message"`
	Context string `json:"context"`
}

type inferenceGenerateResponse struct {
	Response string `json:"response"`
}

func (c *InferenceHTTPClient) Generate(prompt string) (string, error) {
	return c.GenerateWithContext(context.Background(), prompt)
}

func (c *InferenceHTTPClient) GenerateWithContext(ctx context.Context, prompt string) (string, error) {
	requestBody := inferenceGenerateRequest{
		Message: prompt,
		Context: "",
	}

	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/generate", bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("call inference server: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("inference server returned %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	var parsed inferenceGenerateResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	if strings.TrimSpace(parsed.Response) == "" {
		return "", fmt.Errorf("inference server returned empty response")
	}

	return parsed.Response, nil
}
