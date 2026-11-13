use crate::regexp::Regexp;

pub fn any_regex_match(f: &str, res: &[Regexp]) -> bool {
    res.iter().any(|re| regex_matched(f, re))
}

pub fn regex_matched(f: &str, re: &Regexp) -> bool {
    re.find_string_nonempty(f)
}

/// joinRegexOr combines multiple patterns into a single regexp.
pub fn join_regex_or(patterns: &[Regexp]) -> Regexp {
    let mut sb = String::from("(?:");
    for (i, p) in patterns.iter().enumerate() {
        sb.push_str(p.as_str());
        if i != patterns.len() - 1 {
            sb.push('|');
        }
    }
    sb.push(')');
    Regexp::must_compile(&sb)
}
