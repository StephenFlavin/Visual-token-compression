//! Minimal emulation of the lipgloss/termenv styling used when printing
//! verbose findings.

use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Profile {
    Ascii,
    Ansi256,
    TrueColor,
}

fn profile() -> Profile {
    static P: OnceLock<Profile> = OnceLock::new();
    *P.get_or_init(|| {
        let env = |k: &str| std::env::var(k).unwrap_or_default();
        let forced = !env("CLICOLOR_FORCE").is_empty() && env("CLICOLOR_FORCE") != "0";
        if !env("NO_COLOR").is_empty() || (env("CLICOLOR") == "0" && !forced) {
            return Profile::Ascii;
        }
        let tty = unsafe { libc::isatty(1) } == 1;
        if !tty && !forced {
            return Profile::Ascii;
        }
        let term = env("TERM");
        let colorterm = env("COLORTERM").to_lowercase();
        if colorterm == "truecolor" || colorterm == "24bit" {
            return Profile::TrueColor;
        }
        if term == "dumb" && !forced {
            return Profile::Ascii;
        }
        if term.contains("256color") {
            return Profile::Ansi256;
        }
        Profile::Ansi256
    })
}

fn ansi256(r: u8, g: u8, b: u8) -> u8 {
    let v = |x: u8| -> u8 {
        if x < 48 {
            0
        } else if x < 115 {
            1
        } else {
            (x - 35) / 40
        }
    };
    16 + 36 * v(r) + 6 * v(g) + v(b)
}

/// Render text with optional bold+italic and a foreground color, mirroring
/// lipgloss v0.5 behaviour (multi-line strings are padded to equal width).
pub fn render(s: &str, bold: bool, italic: bool, fg: Option<(u8, u8, u8)>) -> String {
    let p = profile();
    let mut styles: Vec<String> = Vec::new();
    if bold {
        styles.push("1".to_string());
    }
    if italic {
        styles.push("3".to_string());
    }
    if let Some((r, g, b)) = fg {
        styles.push(match p {
            Profile::Ascii => String::new(),
            Profile::Ansi256 => format!("38;5;{}", ansi256(r, g, b)),
            Profile::TrueColor => format!("38;2;{};{};{}", r, g, b),
        });
    }
    let seq = styles.join(";");
    let styled = |l: &str| -> String {
        if styles.is_empty() || seq.is_empty() {
            l.to_string()
        } else {
            format!("\x1b[{}m{}\x1b[0m", seq, l)
        }
    };
    let lines: Vec<&str> = s.split('\n').collect();
    let mut out: Vec<String> = lines.iter().map(|l| styled(l)).collect();
    if lines.len() > 1 {
        let widths: Vec<usize> = lines.iter().map(|l| unicode_width::UnicodeWidthStr::width(*l)).collect();
        let max = *widths.iter().max().unwrap_or(&0);
        for (i, o) in out.iter_mut().enumerate() {
            if widths[i] < max {
                o.push_str(&" ".repeat(max - widths[i]));
            }
        }
    }
    out.join("\n")
}
