package main

import (
	"fmt"
	"net/http"
	"os"
	"strings"

	"methodology-rag-assistant/api"
	"methodology-rag-assistant/config"
	"methodology-rag-assistant/llm"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: go run . \"your prompt\" OR go run . serve")
	}

	cfg, err := config.Load()
	if err != nil {
		return err
	}

	httpClient := &http.Client{Timeout: cfg.InferenceTimeout}
	inferenceClient := llm.NewInferenceHTTPClient(cfg.InferenceBaseURL, httpClient)

	if args[0] == "serve" {
		server := api.NewHTTPServer(inferenceClient)
		fmt.Fprintf(os.Stdout, "HTTP server listening on %s\n", cfg.HTTPAddr)
		return http.ListenAndServe(cfg.HTTPAddr, server.Handler())
	}

	cli := api.NewCLI(inferenceClient, os.Stdout)

	prompt := strings.Join(args, " ")
	return cli.Run(prompt)
}
