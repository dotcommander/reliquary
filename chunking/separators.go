package chunking

// SeparatorProfile names an ordered list of separators for recursive text
// splitting. The package provides pure profile data only; callers own the
// splitting algorithm that consumes it.
type SeparatorProfile struct {
	ID         string
	Separators []string
}

const (
	SeparatorProfileDefaultTextID = "default_text"
	SeparatorProfileCJKThaiID     = "cjk_thai"
)

// DefaultTextSeparatorProfile returns separators for whitespace-delimited
// prose, ordered from broad to narrow.
func DefaultTextSeparatorProfile() SeparatorProfile {
	return SeparatorProfile{
		ID:         SeparatorProfileDefaultTextID,
		Separators: []string{"\n\n", "\n", " ", ""},
	}
}

// CJKThaiSeparatorProfile returns separators for scripts that commonly lack
// whitespace word boundaries, including fullwidth and ideographic punctuation
// plus a zero-width space.
func CJKThaiSeparatorProfile() SeparatorProfile {
	return SeparatorProfile{
		ID: SeparatorProfileCJKThaiID,
		Separators: []string{
			"\n\n",
			"\n",
			" ",
			".",
			",",
			"\u200b",
			"\uff0c",
			"\u3001",
			"\uff0e",
			"\u3002",
			"",
		},
	}
}

// SeparatorStrings returns a detached separator slice.
func (profile SeparatorProfile) SeparatorStrings() []string {
	if len(profile.Separators) == 0 {
		return nil
	}
	out := make([]string, len(profile.Separators))
	copy(out, profile.Separators)
	return out
}
