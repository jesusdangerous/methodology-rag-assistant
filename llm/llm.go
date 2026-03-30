package llm

type Generator interface {
	Generate(prompt string) (string, error)
}
