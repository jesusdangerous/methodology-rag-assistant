package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

type Config struct {
	InferenceBaseURL string
	HTTPAddr         string
	InferenceTimeout time.Duration
}

func Load() (Config, error) {
	if err := loadDotEnv(); err != nil {
		return Config{}, err
	}

	baseURL := strings.TrimSpace(os.Getenv("INFERENCE_BASE_URL"))
	if baseURL == "" {
		return Config{}, fmt.Errorf("INFERENCE_BASE_URL is required")
	}

	timeoutSeconds := 600
	if raw := strings.TrimSpace(os.Getenv("INFERENCE_TIMEOUT_SECONDS")); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed <= 0 {
			return Config{}, fmt.Errorf("invalid INFERENCE_TIMEOUT_SECONDS: %q", raw)
		}
		timeoutSeconds = parsed
	}

	return Config{
		InferenceBaseURL: strings.TrimRight(baseURL, "/"),
		HTTPAddr:         envOrDefault("APP_HTTP_ADDR", ":8090"),
		InferenceTimeout: time.Duration(timeoutSeconds) * time.Second,
	}, nil
}

func envOrDefault(key, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}

func loadDotEnv() error {
	err := godotenv.Load(".env")
	if err == nil {
		return nil
	}

	if os.IsNotExist(err) {
		return nil
	}

	return fmt.Errorf("failed to load .env: %w", err)
}
