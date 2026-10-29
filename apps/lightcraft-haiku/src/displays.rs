//! The profile of the screen showing the window (`Services::screen_profile`, used while Settings ▸
//! Display has no monitor profile chosen), as PhotoCraft's host reads its displays' profiles.
//!
//! Neither Haiku nor air/OS keeps a colour profile for a screen: app_server has no colour
//! management and the Screen preferences none either. air/OS's display layout knows every
//! monitor (its connector, its EDID name, vendor and serial number, and where it is on the
//! desktop); plain Haiku has one screen and `BScreen::GetMonitorInfo`. So the profile of a
//! monitor is a file the user puts in `~/config/settings/ColorProfiles/` (made at the first
//! read, so it can be found), named after the monitor; the first of these that exists is used
//! (letter case and the `.icc`/`.icm` extension don't matter, `/` in a name becomes `-`):
//!
//! 1. `<monitor name> <serial number>.icc`, for one of two identical monitors;
//! 2. `<monitor name>.icc` (the name in the Screen preferences, e.g. `DELL U2720Q.icc`);
//! 3. `<connector>.icc` (e.g. `DP-2.icc`), for a monitor without EDID;
//! 4. `default.icc`, for every monitor without one of its own.
//!
//! A symbolic link to a profile elsewhere works too. PhotoCraft reads the same folder, and
//! Settings ▸ Display ▸ Choose Monitor Profile starts there.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// The folder monitor profiles are put in: `~/config/settings/ColorProfiles`.
pub fn profiles_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty()).unwrap_or_else(|| "/boot/home".into());
    Some(PathBuf::from(home).join("config/settings/ColorProfiles"))
}

/// Larger files are no monitor profile.
const MAX_PROFILE_BYTES: u64 = 4 << 20;

/// One display as the host lists it (`lc_host_displays`).
#[derive(Clone, Debug, PartialEq)]
pub struct Monitor {
    pub id: u32,
    /// Left, top, right, bottom (exclusive), in screen coordinates as window frames use them.
    pub frame: [f64; 4],
    pub primary: bool,
    pub connector: String,
    pub name: String,
    pub vendor: String,
    pub serial: String,
}

/// The host's lines: id, left, top, right, bottom, primary, connector, monitor name, vendor,
/// serial number (tab-separated). Malformed lines are skipped.
pub fn parse(text: &str) -> Vec<Monitor> {
    text.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            let num = |i: usize| f.get(i).and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| v.is_finite());
            let text = |i: usize| f.get(i).map_or(String::new(), |v| v.trim().to_string());
            let frame = [num(1)?, num(2)?, num(3)?, num(4)?];
            (frame[2] > frame[0] && frame[3] > frame[1]).then(|| Monitor {
                id: f.first().and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(0),
                frame,
                primary: num(5) == Some(1.0),
                connector: text(6),
                name: text(7),
                vendor: text(8),
                serial: text(9),
            })
        })
        .collect()
}

/// The file names (without extension) that give `m` its profile, in order of preference.
pub fn profile_names(m: &Monitor) -> Vec<String> {
    let clean = |s: &str| s.replace('/', "-").trim().to_string();
    let mut names = Vec::new();
    if !m.name.is_empty() && !m.serial.is_empty() {
        names.push(clean(&format!("{} {}", m.name, m.serial)));
    }
    if !m.name.is_empty() {
        names.push(clean(&m.name));
    }
    if !m.connector.is_empty() {
        names.push(clean(&m.connector));
    }
    names.push("default".into());
    names
}

/// The profile file in `dir` for `m`, if there is one.
pub fn profile_file(dir: &Path, m: &Monitor) -> Option<PathBuf> {
    let files: Vec<(String, PathBuf)> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            let ext = path.extension()?.to_str()?.to_ascii_lowercase();
            (ext == "icc" || ext == "icm").then(|| Some((path.file_stem()?.to_str()?.to_lowercase(), path.clone())))?
        })
        .collect();
    profile_names(m).iter().find_map(|name| {
        let name = name.to_lowercase();
        // Sorted, so the choice doesn't depend on the directory's order (`x.icc` before `x.icm`).
        let mut found: Vec<&PathBuf> = files.iter().filter(|(stem, _)| *stem == name).map(|(_, p)| p).collect();
        found.sort();
        found.first().map(|p| (*p).clone())
    })
}

/// The monitor showing most of `window` (left, top, right, bottom in screen coordinates); the
/// primary one when the window is on none of them.
pub fn monitor_for(monitors: &[Monitor], window: [f64; 4]) -> Option<&Monitor> {
    let overlap = |m: &Monitor| {
        let w = (window[2].min(m.frame[2]) - window[0].max(m.frame[0])).max(0.0);
        let h = (window[3].min(m.frame[3]) - window[1].max(m.frame[1])).max(0.0);
        w * h
    };
    let best = monitors.iter().map(|m| (overlap(m), m)).filter(|(a, _)| *a > 0.0).max_by(|a, b| a.0.total_cmp(&b.0)).map(|(_, m)| m);
    best.or_else(|| monitors.iter().find(|m| m.primary)).or_else(|| monitors.first())
}

/// Reads the profile of the screen showing the window, keeping the last file's bytes while it
/// is unchanged (the app asks about once a second).
#[derive(Default)]
pub struct ScreenProfile {
    dir: Option<PathBuf>,
    made_dir: bool,
    last: Option<(PathBuf, Option<SystemTime>, u64, Vec<u8>)>,
}

impl ScreenProfile {
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self { dir, made_dir: false, last: None }
    }

    /// The profile for the window at `window` among the host's `displays` lines, if its monitor
    /// has one.
    pub fn read(&mut self, displays: &str, window: [f64; 4]) -> Option<Vec<u8>> {
        let dir = self.dir.clone()?;
        if !self.made_dir {
            // Made once, so the folder is there to put profiles in.
            self.made_dir = true;
            if let Err(e) = std::fs::create_dir_all(&dir) {
                log::warn!(target: "lightcraft_haiku", "{}: {e}", dir.display());
            }
        }
        let monitors = parse(displays);
        let file = profile_file(&dir, monitor_for(&monitors, window)?)?;
        let meta = std::fs::metadata(&file).ok().filter(|m| m.len() <= MAX_PROFILE_BYTES)?;
        let (modified, len) = (meta.modified().ok(), meta.len());
        if let Some((path, m, l, bytes)) = &self.last
            && *path == file
            && *m == modified
            && *l == len
        {
            return Some(bytes.clone());
        }
        match std::fs::read(&file) {
            Ok(bytes) => {
                log::info!(target: "lightcraft_haiku", "screen profile: {}", file.display());
                self.last = Some((file, modified, len, bytes.clone()));
                Some(bytes)
            }
            Err(e) => {
                log::warn!(target: "lightcraft_haiku", "monitor profile {}: {e}", file.display());
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO: &str = "1\t0\t0\t2560\t1440\t1\tDP-1\tDELL U2720Q\tDEL\tABC123\n\
                       2\t2560\t0\t4480\t1080\t0\tHDMI-1\t\t\t\n\
                       bad line\n";

    /// A fresh folder holding `files` (their names as their contents), removed when dropped.
    struct Dir(PathBuf);
    impl Dir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn dir(files: &[&str]) -> Dir {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("lightcraft-haiku-profiles-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for f in files {
            std::fs::write(d.join(f), f.as_bytes()).unwrap();
        }
        Dir(d)
    }

    #[test]
    fn parses_the_hosts_lines_and_skips_malformed_ones() {
        let m = parse(TWO);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].frame, [0.0, 0.0, 2560.0, 1440.0]);
        assert!(m[0].primary && !m[1].primary);
        assert_eq!((m[0].name.as_str(), m[0].serial.as_str(), m[1].connector.as_str()), ("DELL U2720Q", "ABC123", "HDMI-1"));
    }

    #[test]
    fn profile_names_go_from_most_to_least_specific() {
        let m = parse(TWO);
        assert_eq!(profile_names(&m[0]), ["DELL U2720Q ABC123", "DELL U2720Q", "DP-1", "default"]);
        assert_eq!(profile_names(&m[1]), ["HDMI-1", "default"]);
    }

    #[test]
    fn the_monitors_own_profile_wins_over_the_default_and_case_is_ignored() {
        let d = dir(&["dell u2720q.ICC", "default.icm"]);
        let m = parse(TWO);
        assert_eq!(profile_file(d.path(), &m[0]).unwrap().file_name().unwrap(), "dell u2720q.ICC");
        assert_eq!(profile_file(d.path(), &m[1]).unwrap().file_name().unwrap(), "default.icm");
        let empty = dir(&["notes.txt"]);
        assert_eq!(profile_file(empty.path(), &m[0]), None);
    }

    #[test]
    fn the_window_takes_the_profile_of_the_monitor_showing_most_of_it() {
        let m = parse(TWO);
        assert_eq!(monitor_for(&m, [100.0, 100.0, 900.0, 700.0]).unwrap().id, 1);
        assert_eq!(monitor_for(&m, [2400.0, 100.0, 3400.0, 700.0]).unwrap().id, 2);
        // off every screen: the primary one
        assert_eq!(monitor_for(&m, [-900.0, -900.0, -100.0, -100.0]).unwrap().id, 1);
        assert!(monitor_for(&[], [0.0, 0.0, 1.0, 1.0]).is_none());

        let d = dir(&["HDMI-1.icc"]);
        let mut reader = ScreenProfile::new(Some(d.path().to_path_buf()));
        assert_eq!(reader.read(TWO, [100.0, 100.0, 900.0, 700.0]), None);
        assert_eq!(reader.read(TWO, [2600.0, 100.0, 3400.0, 700.0]).as_deref(), Some(&b"HDMI-1.icc"[..]));
        assert_eq!(ScreenProfile::new(None).read(TWO, [2600.0, 100.0, 3400.0, 700.0]), None);
    }

    #[test]
    fn the_folder_is_made_at_the_first_read() {
        let d = dir(&[]);
        let sub = d.path().join("ColorProfiles");
        let mut reader = ScreenProfile::new(Some(sub.clone()));
        assert_eq!(reader.read("", [0.0, 0.0, 1.0, 1.0]), None);
        assert!(sub.is_dir());
    }
}
