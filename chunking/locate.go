package chunking

import "github.com/dotcommander/reliquary/textutil"

// Locate returns the byte offsets [start, end) of fragment in content.
// It tries exact substring match from cursor, then exact match from 0,
// then whitespace-normalized match. ok is false if no match is found.
//
// The locate machinery lives in textutil; this delegates with
// ExactFirst ordering (exact matches preferred over normalized) so the
// chunking pipeline does not duplicate the implementation.
func Locate(content, fragment string, cursor int) (int, int, bool) {
	return textutil.FragmentRange(content, fragment, cursor, textutil.ExactFirst)
}
