package judgment

// ValidateChoiceRequest rejects malformed tasks and option sets with ErrInvalidRequest.
// It applies the same request rules as ValidateChoice without requiring a result.
func ValidateChoiceRequest(request ChoiceRequest) error {
	return validateChoiceRequest(request)
}

// ValidateScoreRequest rejects blank tasks or targets with ErrInvalidRequest.
// It applies the same request rules as ValidateScore without requiring a result.
func ValidateScoreRequest(request ScoreRequest) error {
	return validateScoreRequest(request)
}

// ValidateNoulRequest rejects blank questions with ErrInvalidRequest.
// It applies the same request rules as ValidateNoul without requiring a result.
func ValidateNoulRequest(request NoulRequest) error {
	return validateNoulRequest(request)
}
