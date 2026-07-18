package textutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExtractKeywordsDeterministic(t *testing.T) {
	t.Parallel()

	keywords := ExtractKeywords([]string{
		"Go routines and channels make concurrency practical.",
		"Channels help coordinate goroutines in Go.",
	}, KeywordOptions{Limit: 3, MinCount: 1})

	assert.Equal(t, []string{"channels", "concurrency", "coordinate"}, keywords)
}

func TestExtractKeywordsPerCallStopWords(t *testing.T) {
	t.Parallel()

	before := len(DefaultStopWordsCopy())
	keywords := ExtractKeywords([]string{
		"Channels help coordinate channels in Go.",
		"Goroutines coordinate work through channels.",
	}, KeywordOptions{
		Limit:     3,
		MinCount:  1,
		StopWords: map[string]struct{}{"channels": {}},
	})

	assert.Equal(t, []string{"coordinate", "goroutines", "help"}, keywords)
	assert.Equal(t, before, len(DefaultStopWordsCopy()), "per-call stop words must not mutate default stop words")
	assert.False(t, IsStopWord("channels"))
}

func TestMostFrequentValueWithMinimum(t *testing.T) {
	t.Parallel()

	assert.Equal(t, "docs", MostFrequentValue([]string{"docs", "api", "docs"}, 2, "fallback"))
	assert.Equal(t, "fallback", MostFrequentValue([]string{"docs", "api"}, 2, "fallback"))
}

func TestDetectTheme(t *testing.T) {
	t.Parallel()

	themes := map[string][]string{
		"engineering": {"go", "chan", "channel"},
		"docs":        {"readme", "guide"},
	}

	detected := DetectTheme("This project uses Go channels heavily.", themes, "fallback")
	assert.Equal(t, "engineering", detected)

	noMatch := DetectTheme("No related language appears", themes, "fallback")
	assert.Equal(t, "fallback", noMatch)
}

func TestDefaultStopWordsCopyReturnsCopy(t *testing.T) {
	t.Parallel()

	assert.True(t, IsStopWord("the"))
	assert.False(t, IsStopWord("channels"))

	words := DefaultStopWordsCopy()
	words["channels"] = struct{}{}
	delete(words, "the")

	assert.True(t, IsStopWord("the"))
	assert.False(t, IsStopWord("channels"))
	assert.Contains(t, DefaultStopWordsCopy(), "the")
}

func TestDefaultStopWordsCompatibilityMapDoesNotDrivePackageBehavior(t *testing.T) {
	assert.Contains(t, DefaultStopWords, "the")

	oldChannels, hadChannels := DefaultStopWords["channels"]
	oldThe, hadThe := DefaultStopWords["the"]
	t.Cleanup(func() {
		if hadChannels {
			DefaultStopWords["channels"] = oldChannels
		} else {
			delete(DefaultStopWords, "channels")
		}
		if hadThe {
			DefaultStopWords["the"] = oldThe
		} else {
			delete(DefaultStopWords, "the")
		}
	})

	DefaultStopWords["channels"] = struct{}{}
	delete(DefaultStopWords, "the")

	assert.True(t, IsStopWord("the"))
	assert.False(t, IsStopWord("channels"))
}

func TestMostFrequentValueTieBreak(t *testing.T) {
	t.Parallel()
	// "apple" and "banana" each appear twice; alphabetically smaller wins.
	assert.Equal(t, "apple", MostFrequentValue([]string{"banana", "apple", "banana", "apple"}, 1, "fallback"))
}

func TestTitleWords(t *testing.T) {
	t.Parallel()

	assert.Equal(t, "Hello World Foo", TitleWords("hello_world-foo"))
}

func TestNormalizeKeywordToken(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name, in, want string
	}{
		{"empty", "", ""},
		{"all punctuation", "..,;:", ""},
		{"leading punctuation", "((channels", "channels"},
		{"trailing punctuation", "channels))", "channels"},
		{"both ends", "(channels).", "channels"},
		{"interior punctuation kept", "foo.bar", "foo.bar"},
		{"multibyte adjacent", "(café)", "café"},
		{"multibyte only", "café", "café"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.want, NormalizeKeywordToken(tc.in))
		})
	}
}
