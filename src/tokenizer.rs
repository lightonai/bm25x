/// A simple tokenizer: lowercase, split on non-alphanumeric, optional stopword removal.
pub struct Tokenizer {
    stopwords: Option<std::collections::HashSet<String>>,
}

const ENGLISH_STOPWORDS: &[&str] = &[
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "its",
    "itself",
    "them",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "am",
    "been",
    "being",
    "do",
    "does",
    "did",
    "doing",
    "would",
    "should",
    "could",
    "ought",
    "might",
    "shall",
    "can",
    "need",
    "dare",
    "had",
    "has",
    "have",
    "having",
    "about",
    "above",
    "after",
    "again",
    "against",
    "below",
    "between",
    "during",
    "from",
    "further",
    "here",
    "once",
    "only",
    "out",
    "over",
    "same",
    "so",
    "than",
    "too",
    "under",
    "until",
    "up",
    "very",
    "own",
    "just",
    "don",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "s",
    "t",
    "ve",
];

impl Tokenizer {
    pub fn new(use_stopwords: bool) -> Self {
        let stopwords = if use_stopwords {
            Some(ENGLISH_STOPWORDS.iter().map(|s| s.to_string()).collect())
        } else {
            None
        };
        Tokenizer { stopwords }
    }

    /// Tokenize a string into a list of tokens.
    pub fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut tokens = Vec::new();
        let bytes = text.as_bytes();
        let mut start = None;

        for (i, &b) in bytes.iter().enumerate() {
            let is_alnum = b.is_ascii_alphanumeric();
            if is_alnum {
                if start.is_none() {
                    start = Some(i);
                }
            } else if let Some(s) = start {
                let token = &text[s..i];
                start = None;
                if self.should_keep(token) {
                    tokens.push(token);
                }
            }
        }
        // Handle last token
        if let Some(s) = start {
            let token = &text[s..];
            if self.should_keep(token) {
                tokens.push(token);
            }
        }

        tokens
    }

    /// Tokenize and return owned lowercase tokens.
    pub fn tokenize_owned(&self, text: &str) -> Vec<String> {
        let lower = text.to_ascii_lowercase();
        self.tokenize(&lower)
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[inline]
    fn should_keep(&self, token: &str) -> bool {
        if token.is_empty() {
            return false;
        }
        if let Some(ref sw) = self.stopwords {
            !sw.contains(token)
        } else {
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenize() {
        let tok = Tokenizer::new(false);
        let text = "hello world";
        let tokens = tok.tokenize(text);
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_stopwords() {
        let tok = Tokenizer::new(true);
        let text = "the quick brown fox";
        let tokens = tok.tokenize(text);
        assert_eq!(tokens, vec!["quick", "brown", "fox"]);
    }

    #[test]
    fn test_punctuation() {
        let tok = Tokenizer::new(false);
        let text = "hello, world! how's it going?";
        let tokens = tok.tokenize(text);
        assert_eq!(tokens, vec!["hello", "world", "how", "s", "it", "going"]);
    }

    #[test]
    fn test_owned_lowercase() {
        let tok = Tokenizer::new(false);
        let tokens = tok.tokenize_owned("Hello WORLD");
        assert_eq!(tokens, vec!["hello", "world"]);
    }
}
