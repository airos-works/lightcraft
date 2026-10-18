//! Haiku keyboard messages to egui keys and modifiers. Platform-neutral tables.
//!
//! A `B_KEY_DOWN` carries the physical key code (`key`, the same on every keymap), the
//! character the key types without modifiers (`raw_char`, first byte only) and the bytes it
//! types now. Keys that don't type (arrows, F-keys, Enter…) are told by their key code; the
//! others by `raw_char`, so Command+Z is Z on any Latin keymap, with the US position as the
//! physical key for the rest.

use egui::{Key, Modifiers};

/// Haiku's modifier bits (`InterfaceDefs.h`).
pub const B_SHIFT_KEY: u32 = 0x0001;
pub const B_COMMAND_KEY: u32 = 0x0002;
pub const B_CONTROL_KEY: u32 = 0x0004;
pub const B_OPTION_KEY: u32 = 0x0040;

/// egui's modifiers for Haiku's. Command (the left Alt key by default) and Control are both
/// egui's Ctrl/command, so Word's Ctrl shortcuts work with either, as Haiku users press
/// Command+C where others press Ctrl+C. Option (the Windows key) is egui's Alt.
pub fn modifiers(bits: u32) -> Modifiers {
    let ctrl = bits & (B_COMMAND_KEY | B_CONTROL_KEY) != 0;
    Modifiers { alt: bits & B_OPTION_KEY != 0, ctrl, shift: bits & B_SHIFT_KEY != 0, mac_cmd: false, command: ctrl }
}

/// The key a Haiku key code names when it doesn't type a character (or always types the same).
fn by_code(code: i32) -> Option<Key> {
    Some(match code {
        0x01 => Key::Escape,
        0x02 => Key::F1,
        0x03 => Key::F2,
        0x04 => Key::F3,
        0x05 => Key::F4,
        0x06 => Key::F5,
        0x07 => Key::F6,
        0x08 => Key::F7,
        0x09 => Key::F8,
        0x0a => Key::F9,
        0x0b => Key::F10,
        0x0c => Key::F11,
        0x0d => Key::F12,
        0x1e => Key::Backspace,
        0x1f => Key::Insert,
        0x20 => Key::Home,
        0x21 => Key::PageUp,
        0x26 => Key::Tab,
        0x34 => Key::Delete,
        0x35 => Key::End,
        0x36 => Key::PageDown,
        0x47 | 0x5b => Key::Enter,
        0x57 => Key::ArrowUp,
        0x5e => Key::Space,
        0x61 => Key::ArrowLeft,
        0x62 => Key::ArrowDown,
        0x63 => Key::ArrowRight,
        _ => return None,
    })
}

/// Numeric keypad keys: what they do follows Num Lock, so they're told by what they type.
fn is_keypad(code: i32) -> bool {
    matches!(code, 0x23 | 0x24 | 0x25 | 0x37..=0x3a | 0x48..=0x4a | 0x58..=0x5a | 0x64 | 0x65)
}

/// The key for a character Haiku's keymap gives (`raw_char`, or the typed byte on the keypad).
fn by_char(c: u8) -> Option<Key> {
    Some(match c {
        b'a'..=b'z' | b'A'..=b'Z' => return Key::from_name(&(c.to_ascii_uppercase() as char).to_string()),
        b'0' => Key::Num0,
        b'1' => Key::Num1,
        b'2' => Key::Num2,
        b'3' => Key::Num3,
        b'4' => Key::Num4,
        b'5' => Key::Num5,
        b'6' => Key::Num6,
        b'7' => Key::Num7,
        b'8' => Key::Num8,
        b'9' => Key::Num9,
        b'-' => Key::Minus,
        b'=' => Key::Equals,
        b'+' => Key::Plus,
        b'[' => Key::OpenBracket,
        b']' => Key::CloseBracket,
        b'{' => Key::OpenCurlyBracket,
        b'}' => Key::CloseCurlyBracket,
        b'\\' => Key::Backslash,
        b'|' => Key::Pipe,
        b';' => Key::Semicolon,
        b':' => Key::Colon,
        b'\'' => Key::Quote,
        b',' => Key::Comma,
        b'.' => Key::Period,
        b'/' => Key::Slash,
        b'?' => Key::Questionmark,
        b'!' => Key::Exclamationmark,
        b'`' => Key::Backtick,
        b' ' => Key::Space,
        // Haiku's codes for keys that don't type (B_LEFT_ARROW…), as the keypad sends them
        // with Num Lock off.
        0x01 => Key::Home,
        0x04 => Key::End,
        0x05 => Key::Insert,
        0x08 => Key::Backspace,
        0x09 => Key::Tab,
        0x0a => Key::Enter,
        0x0b => Key::PageUp,
        0x0c => Key::PageDown,
        0x1b => Key::Escape,
        0x1c => Key::ArrowLeft,
        0x1d => Key::ArrowRight,
        0x1e => Key::ArrowUp,
        0x1f => Key::ArrowDown,
        0x7f => Key::Delete,
        _ => return None,
    })
}

/// The key on the US layout at a key code (letters and digits), for egui's physical key.
fn physical(code: i32) -> Option<Key> {
    const ROW_Q: &str = "QWERTYUIOP";
    const ROW_A: &str = "ASDFGHJKL";
    const ROW_Z: &str = "ZXCVBNM";
    const DIGITS: &str = "1234567890";
    let pick = |row: &str, first: i32| -> Option<Key> {
        let i = usize::try_from(code - first).ok()?;
        Key::from_name(row.get(i..i + 1)?)
    };
    match code {
        0x12..=0x1b => pick(DIGITS, 0x12),
        0x27..=0x30 => pick(ROW_Q, 0x27),
        0x3c..=0x44 => pick(ROW_A, 0x3c),
        0x4c..=0x52 => pick(ROW_Z, 0x4c),
        _ => by_code(code),
    }
}

/// The egui key (logical, physical) of a key message.
pub fn key(code: i32, raw_char: i32, bytes: &[u8]) -> (Option<Key>, Option<Key>) {
    let physical = physical(code);
    if is_keypad(code) {
        return (bytes.first().copied().and_then(by_char), physical);
    }
    if let Some(k) = by_code(code) {
        return (Some(k), physical);
    }
    let logical = u8::try_from(raw_char).ok().filter(|c| c.is_ascii()).and_then(by_char).or(physical);
    (logical, physical)
}

/// The text a key message types: printable characters only, and nothing while Command or
/// Control is held (those are shortcuts).
pub fn text(bits: u32, bytes: &str) -> Option<&str> {
    if bits & (B_COMMAND_KEY | B_CONTROL_KEY) != 0 || bytes.is_empty() {
        return None;
    }
    // Haiku's non-typing keys send control characters; the function keys B_FUNCTION_KEY (0x10).
    bytes.chars().all(|c| !c.is_control() && !('\u{f700}'..='\u{f8ff}').contains(&c)).then_some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letters_follow_the_keymap_and_keep_their_us_position() {
        // AZERTY: the key at Q's position types 'a'.
        assert_eq!(key(0x27, b'a' as i32, b"a"), (Some(Key::A), Some(Key::Q)));
        assert_eq!(key(0x4e, b'c' as i32, b"\x03"), (Some(Key::C), Some(Key::C)));
        assert_eq!(key(0x12, b'1' as i32, b"1"), (Some(Key::Num1), Some(Key::Num1)));
    }

    #[test]
    fn keys_that_dont_type_come_from_the_key_code() {
        assert_eq!(key(0x61, 0x1c, b"\x1c").0, Some(Key::ArrowLeft));
        assert_eq!(key(0x47, 0x0a, b"\n").0, Some(Key::Enter));
        assert_eq!(key(0x02, 0x10, b"\x10").0, Some(Key::F1));
        assert_eq!(key(0x1e, 0x08, b"\x08").0, Some(Key::Backspace));
        // Ctrl+A types 0x01 (B_HOME's code), still A.
        assert_eq!(key(0x3c, b'a' as i32, b"\x01").0, Some(Key::A));
    }

    #[test]
    fn the_keypad_follows_num_lock() {
        assert_eq!(key(0x37, 0x01, b"7").0, Some(Key::Num7));
        assert_eq!(key(0x37, 0x01, b"\x01").0, Some(Key::Home));
    }

    #[test]
    fn command_and_control_are_shortcuts_and_type_nothing() {
        let m = modifiers(B_COMMAND_KEY);
        assert!(m.command && m.ctrl && !m.alt);
        assert!(modifiers(B_CONTROL_KEY).command);
        assert!(modifiers(B_OPTION_KEY).alt);
        assert_eq!(text(B_COMMAND_KEY, "c"), None);
        assert_eq!(text(0, "é"), Some("é"));
        assert_eq!(text(B_SHIFT_KEY | B_OPTION_KEY, "Å"), Some("Å"));
        assert_eq!(text(0, "\x1c"), None);
        assert_eq!(text(0, "\x10"), None);
    }
}
