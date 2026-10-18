//! Keep a failed window-device startup from repeating on the next ordinary launch.
use std::path::PathBuf;

pub struct WindowRecovery {
    marker: Option<PathBuf>,
    writable: bool,
}
impl WindowRecovery {
    pub fn new(dir: Option<PathBuf>, writable: bool) -> Self {
        Self { marker: dir.map(|d| d.join("window-init.marker")), writable }
    }
    /// Memory-only runs honor a previous failure while leaving its file untouched.
    pub fn previous_failure(&self) -> Option<String> {
        let reason = std::fs::read_to_string(self.marker.as_ref()?).ok()?;
        self.clear();
        Some(reason.trim().to_owned())
    }
    pub fn record(&self, reason: &str) {
        if !self.writable {
            return;
        }
        if let Some(path) = &self.marker {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Err(e) = std::fs::write(path, reason) {
                log::warn!("Could not record native GPU startup: {e}");
            }
        }
    }
    pub fn clear(&self) {
        if self.writable
            && let Some(path) = &self.marker
        {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_start_is_reported_once_but_memory_runs_preserve_it() {
        let dir = std::env::temp_dir().join(format!("lightcraft-native-recovery-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let memory = WindowRecovery::new(Some(dir.clone()), false);
        memory.record("must not create a settings folder");
        assert!(!dir.exists());
        let persistent = WindowRecovery::new(Some(dir.clone()), true);
        assert_eq!(persistent.previous_failure(), None);
        persistent.record("Vulkan device creation");
        assert_eq!(memory.previous_failure().as_deref(), Some("Vulkan device creation"));
        assert_eq!(persistent.previous_failure().as_deref(), Some("Vulkan device creation"));
        assert_eq!(persistent.previous_failure(), None);
        persistent.record("a completed startup");
        persistent.clear();
        assert_eq!(persistent.previous_failure(), None);
        let _ = std::fs::remove_dir_all(dir);
    }
}
