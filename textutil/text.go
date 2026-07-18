package textutil

import "strings"

// TitleWords normalizes separators and title-cases words for display.
func TitleWords(value string) string {
	value = strings.ReplaceAll(value, "_", " ")
	value = strings.ReplaceAll(value, "-", " ")
	words := strings.Fields(value)
	for i, word := range words {
		lower := strings.ToLower(word)
		if lower == "" {
			continue
		}
		runes := []rune(lower)
		words[i] = strings.ToUpper(string(runes[0:1])) + string(runes[1:])
	}
	return strings.Join(words, " ")
}
