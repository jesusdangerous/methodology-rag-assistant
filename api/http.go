package api

import (
	"encoding/json"
	"net/http"
	"strings"

	"methodology-rag-assistant/llm"
)

type HTTPServer struct {
	generator llm.Generator
}

func NewHTTPServer(generator llm.Generator) *HTTPServer {
	return &HTTPServer{generator: generator}
}

func (s *HTTPServer) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /generate", s.handleGenerate)
	return mux
}

type generateRequest struct {
	Prompt string `json:"prompt"`
}

type generateResponse struct {
	Response string `json:"response"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func (s *HTTPServer) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req generateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, errorResponse{Error: "invalid json"})
		return
	}

	req.Prompt = strings.TrimSpace(req.Prompt)
	if req.Prompt == "" {
		writeJSON(w, http.StatusBadRequest, errorResponse{Error: "prompt is required"})
		return
	}

	generated, err := s.generator.Generate(req.Prompt)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, errorResponse{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, generateResponse{Response: generated})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
