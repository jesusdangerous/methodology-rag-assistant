package api

import (
	"fmt"
	"io"
	"strings"

	"methodology-rag-assistant/llm"
)

type CLI struct {
	generator llm.Generator
	out       io.Writer
}

func NewCLI(generator llm.Generator, out io.Writer) *CLI {
	return &CLI{generator: generator, out: out}
}

func (c *CLI) Run(prompt string) error {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return fmt.Errorf("prompt is required")
	}

	response, err := c.generator.Generate(prompt)
	if err != nil {
		return err
	}

	_, err = fmt.Fprintln(c.out, response)
	return err
}
