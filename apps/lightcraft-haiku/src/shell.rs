//! The window's event loop on the interface thread: Haiku's input becomes egui's, LightcraftApp runs
//! its frame, and egui's output goes back to the window and the system (title, cursor,
//! clipboard, links) and, through the software rasteriser, to the screen.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use egui::{Event, Key, Modifiers, PointerButton, Pos2, Theme, ViewportCommand, ViewportEvent, ViewportId, ViewportInfo, pos2, vec2};
use lightcraft_ui_egui::LightcraftApp;

use crate::host::{self, RawEvent, WindowCommand, kind, wheel_phase};
use crate::keys;
use crate::raster::Rasterizer;
use crate::services;

pub struct Shell {
    pub app: LightcraftApp,
    ctx: egui::Context,
    raster: Rasterizer,
    gpu: Option<crate::gpu::Gpu>,
    gpu_health: Option<crate::gpu::Health>,
    writer: crate::prefs::PrefsWriter,
    window_recovery: crate::window_recovery::WindowRecovery,
    /// The desktop app's GPU-lost marker (`gpu_crash`, issue #1337), until it is written; `None`
    /// for `--memory` and `LIGHTCRAFT_NO_PREFS`. See [`Self::record_gpu_lost`].
    gpu_lost: Option<std::path::PathBuf>,
    /// The window's GPU device failed during this session.
    window_lost: bool,
    start: Instant,
    /// egui's points to device pixels, as of the last frame, and the system's own: the interface
    /// scale (the system font against 12 pt) times the desktop's density.
    ppp: f32,
    native_ppp: f32,
    ui_scale: f32,
    /// Device pixels per logical pixel ([`host::device_scale`]), and when it was last asked.
    dev: f32,
    dev_checked: Instant,
    buttons: i32,
    modifiers: Modifiers,
    focused: bool,
    events: Vec<Event>,
    /// A touchpad gesture's end held back for the next frame (its modifiers), see [`Self::wheel`].
    wheel_end: Option<Modifiers>,
    close_requested: bool,
    quit: bool,
    /// When egui wants the next frame (from this thread's frames).
    repaint_at: Option<Instant>,
    /// When other threads asked for one (`request_repaint` from a worker).
    woken_for: Arc<Mutex<Option<Instant>>>,
    screenshots: Vec<egui::ScreenshotCallback>,
    title: String,
}

/// egui's cursor as one of Haiku's `BCursorID`s.
fn cursor_id(icon: egui::CursorIcon) -> i32 {
    use egui::CursorIcon as C;
    match icon {
        C::None => 12,
        C::ContextMenu => 3,
        C::Help => 9,
        C::PointingHand => 6,
        C::Progress | C::Wait => 14,
        C::Cell | C::Crosshair => 5,
        C::Text => 2,
        C::VerticalText => 10,
        C::Alias => 29,
        C::Copy => 4,
        C::Move | C::AllScroll => 11,
        C::NoDrop | C::NotAllowed => 13,
        C::Grab => 7,
        C::Grabbing => 8,
        C::ResizeHorizontal | C::ResizeColumn => 24,
        C::ResizeVertical | C::ResizeRow => 23,
        C::ResizeNeSw => 25,
        C::ResizeNwSe => 26,
        C::ResizeEast => 16,
        C::ResizeSouthEast => 21,
        C::ResizeSouth => 17,
        C::ResizeSouthWest => 22,
        C::ResizeWest => 18,
        C::ResizeNorthWest => 20,
        C::ResizeNorth => 15,
        C::ResizeNorthEast => 19,
        C::ZoomIn => 27,
        C::ZoomOut => 28,
        _ => 1,
    }
}

fn button(bit: i32) -> Option<PointerButton> {
    Some(match bit {
        1 => PointerButton::Primary,
        2 => PointerButton::Secondary,
        4 => PointerButton::Middle,
        8 => PointerButton::Extra1,
        16 => PointerButton::Extra2,
        _ => return None,
    })
}

impl Shell {
    pub fn new(
        mut app: LightcraftApp,
        writer: crate::prefs::PrefsWriter,
        ctx: egui::Context,
        ui_scale: f32,
        window_recovery: crate::window_recovery::WindowRecovery,
        gpu_lost: Option<std::path::PathBuf>,
        safe_start: bool,
    ) -> Result<Self, String> {
        let mode = std::env::var("LIGHTCRAFT_RENDERER").unwrap_or_else(|_| "auto".into());
        if !matches!(mode.as_str(), "auto" | "cpu" | "vulkan") {
            return Err("LIGHTCRAFT_RENDERER must be auto, cpu or vulkan".into());
        }
        let gpu = if mode == "cpu" || safe_start {
            None
        } else {
            window_recovery.record("Vulkan device creation");
            let result = crate::guard("GPU startup", || crate::gpu::Gpu::new(&crate::gpu::Native));
            window_recovery.clear();
            match result {
                Ok(gpu) => Some(gpu),
                Err(e) if mode == "vulkan" => return Err(e),
                Err(e) => {
                    log::warn!(target: "lightcraft_haiku", "{e}; using the software window renderer");
                    None
                }
            }
        };
        let gpu_health = gpu.as_ref().map(|g| g.health.clone());
        app.perf.host_renderer = Some(gpu.as_ref().map_or("CPU bitmap", |g| match g.present_mode() {
            crate::present::Mode::Direct => "Vulkan direct to screen",
            crate::present::Mode::Readback => "Vulkan changed-tile readback",
            crate::present::Mode::Full => "Vulkan bitmap readback",
        }));
        log::info!(target: "lightcraft_haiku", "air/OS window renderer: {}", app.perf.host_renderer.unwrap_or("unknown"));
        let dev = host::device_scale();
        let native_ppp = ui_scale * dev;
        let woken_for: Arc<Mutex<Option<Instant>>> = Arc::default();
        let wake = woken_for.clone();
        ctx.set_request_repaint_callback(move |info| {
            let at = Instant::now() + info.delay;
            if let Ok(mut w) = wake.lock() {
                *w = Some(w.map_or(at, |old| old.min(at)));
            }
            host::wake();
        });
        Ok(Self {
            app,
            gpu,
            gpu_health,
            ctx,
            raster: Rasterizer::new(),
            writer,
            window_recovery,
            gpu_lost,
            window_lost: false,
            start: Instant::now(),
            ppp: native_ppp,
            native_ppp,
            ui_scale,
            dev,
            dev_checked: Instant::now(),
            buttons: 0,
            modifiers: Modifiers::default(),
            focused: true,
            events: Vec::new(),
            wheel_end: None,
            close_requested: false,
            quit: false,
            repaint_at: Some(Instant::now()),
            woken_for,
            screenshots: Vec::new(),
            title: String::new(),
        })
    }

    /// Runs until the app quits.
    pub fn run(&mut self) {
        while !self.quit {
            host::wait(self.timeout());
            let happened = self.drain();
            let due = self.take_due();
            if happened || due {
                self.frame();
            }
            self.present_chores();
        }
    }

    /// What the presenter does between frames: the last frame again where the window moved on
    /// the screen, and app_server's copy catching up with frames that went straight to the screen.
    fn present_chores(&mut self) {
        let dev = self.dev;
        let Some(gpu) = &mut self.gpu else { return };
        let native = crate::gpu::Native;
        let result = if gpu.wants_present() {
            crate::guard("GPU present", || gpu.present_again(dev, &native))
        } else if gpu.sync_due().is_some_and(|at| at <= Instant::now()) {
            crate::guard("GPU sync", || gpu.sync(dev, &native))
        } else {
            return;
        };
        if let Err(e) = result {
            log::error!(target: "lightcraft_haiku", "{e}");
            if let Some(health) = &self.gpu_health {
                health.mark(e);
            }
            self.ctx.request_repaint();
        }
    }

    /// When the next frame is due, if one is.
    fn next_due(&self) -> Option<Instant> {
        let woken = self.woken_for.lock().ok().and_then(|w| *w);
        let sync = self.gpu.as_ref().and_then(|g| g.sync_due());
        [self.repaint_at, woken, sync].into_iter().flatten().min()
    }

    fn timeout(&self) -> Option<Duration> {
        self.next_due().map(|at| at.saturating_duration_since(Instant::now()))
    }

    /// Whether a frame is due now; clears what made it due.
    fn take_due(&mut self) -> bool {
        let now = Instant::now();
        let due = self.next_due().is_some_and(|at| at <= now);
        if due {
            self.repaint_at = None;
            if let Ok(mut w) = self.woken_for.lock()
                && w.is_some_and(|at| at <= now)
            {
                *w = None;
            }
        }
        due
    }

    /// A point in the window (logical pixels) in egui's points.
    fn pos(&self, x: f32, y: f32) -> Pos2 {
        pos2(x * self.dev / self.ppp, y * self.dev / self.ppp)
    }

    /// Asks for the desktop's density again (the monitors may have changed).
    fn check_density(&mut self) {
        self.dev_checked = Instant::now();
        let dev = host::device_scale();
        if (dev - self.dev).abs() > f32::EPSILON {
            log::info!(target: "lightcraft_haiku", "display density {} -> {dev}", self.dev);
            self.dev = dev;
            self.native_ppp = self.ui_scale * dev;
        }
    }

    fn set_modifiers(&mut self, bits: u32) {
        let m = keys::modifiers(bits);
        if m != self.modifiers {
            self.modifiers = m;
            self.events.push(Event::ModifiersChanged(m));
        }
    }

    /// Turns the queued native events into egui's, starting with those that came while a file
    /// panel was open. Whether there were any.
    fn drain(&mut self) -> bool {
        let mut any = false;
        if let Some(modifiers) = self.wheel_end.take() {
            any = true;
            self.events.push(Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::Vec2::ZERO,
                phase: egui::TouchPhase::End,
                source: egui::MouseWheelSource::Trackpad,
                modifiers,
            });
        }
        for (ev, text) in services::take_deferred() {
            any = true;
            self.event(ev, text);
        }
        while let Some((ev, text)) = host::next_event() {
            any = true;
            self.event(ev, text);
        }
        any
    }

    fn event(&mut self, ev: RawEvent, text: String) {
        match ev.kind {
            kind::MOUSE_MOVED | kind::MOUSE_DOWN | kind::MOUSE_UP => {
                self.set_modifiers(ev.modifiers);
                let pos = self.pos(ev.x, ev.y);
                self.events.push(Event::PointerMoved(pos));
                let changed = ev.buttons ^ self.buttons;
                for bit in [1, 2, 4, 8, 16] {
                    if changed & bit != 0
                        && let Some(b) = button(bit)
                    {
                        self.events.push(Event::PointerButton { pos, button: b, pressed: ev.buttons & bit != 0, modifiers: self.modifiers });
                    }
                }
                self.buttons = ev.buttons;
            }
            kind::POINTER_GONE => self.events.push(Event::PointerGone),
            kind::WHEEL => {
                self.set_modifiers(ev.modifiers);
                self.wheel(&ev);
            }
            kind::KEY_DOWN | kind::KEY_UP => self.key(ev.kind == kind::KEY_DOWN, ev.key, ev.raw_char, ev.modifiers, &text),
            kind::MODIFIERS => self.set_modifiers(host::modifiers()),
            kind::FOCUS => {
                self.focused = ev.x != 0.0;
                self.events.push(Event::WindowFocused(self.focused));
                if !self.focused {
                    // Keys released elsewhere never come back up.
                    self.set_modifiers(0);
                }
            }
            kind::CLOSE | kind::QUIT_REQUESTED => self.close_requested = true,
            // Tracker's Open, Open With and drops on the icon, the files and folders of a later
            // launch (one copy runs: B_SINGLE_LAUNCH; already absolute) and files dropped on the
            // window: import through the same full library path as the desktop app's launch
            // arguments. A later launch without any only brings the window to the front.
            kind::OPEN_FILES | kind::DROPPED_FILES => {
                let paths = text.lines().filter(|p| !p.is_empty()).map(str::to_owned).collect::<Vec<_>>();
                let (paths, notices) = crate::desktop::launch_imports(&paths);
                self.app.notices.extend(notices);
                if let Some(problem) = self.app.library_problem.as_mut() {
                    problem.pending_import.extend(paths);
                } else if !paths.is_empty()
                    && let Err(e) = lightcraft_ui_egui::import::start_paths(&mut self.app, paths)
                {
                    self.app.notices.push(e);
                }
                host::window_command(WindowCommand::Activate);
            }
            kind::DIALOG_ANSWER => services::answer(ev.id, ev.x != 0.0, &text),
            kind::COLORS => self.check_density(),
            // The next frame picks up the new size, position or state.
            kind::RESIZED | kind::MOVED | kind::MINIMIZED => {}
            // A panel's answer after its caller stopped waiting has nobody to go to.
            _ => {}
        }
    }

    /// A wheel turn, or a step of an air/OS touchpad's scroll gesture: its precise deltas (wheel
    /// steps, unrounded) become points at egui's line scroll speed, with the gesture's phases,
    /// so the grid and panels follow the fingers 1:1 and egui's scroll areas treat it as a
    /// trackpad. A wheel's whole steps stay lines.
    fn wheel(&mut self, ev: &RawEvent) {
        use egui::{MouseWheelSource, MouseWheelUnit, TouchPhase};
        let modifiers = self.modifiers;
        // `repeat`: precise deltas (`be:wheel_precise_delta_*`), `clicks`: `be:wheel_phase`.
        if ev.repeat == 0 {
            // A mouse wheel (B_MOUSE_POINTING_DEVICE), or a system without touchpad gestures.
            self.events.push(Event::MouseWheel {
                unit: MouseWheelUnit::Line,
                delta: vec2(-ev.dx, -ev.dy),
                phase: TouchPhase::Move,
                source: MouseWheelSource::Wheel,
                modifiers,
            });
            return;
        }
        let event = |unit, delta, phase| Event::MouseWheel { unit, delta, phase, source: MouseWheelSource::Trackpad, modifiers };
        let line = self.ctx.options(|o| o.input_options.line_scroll_speed);
        let delta = vec2(-ev.dx, -ev.dy) * line;
        let delta = if delta.is_finite() { delta } else { egui::Vec2::ZERO };
        match ev.clicks {
            wheel_phase::BEGAN => {
                self.events.push(event(MouseWheelUnit::Point, egui::Vec2::ZERO, TouchPhase::Start));
            }
            wheel_phase::ENDED => {
                // egui drops a gesture's unapplied deltas at its end: one that ends in the same
                // frame as its last movement ends in the next.
                let moved =
                    self.events.iter().any(|e| matches!(e, Event::MouseWheel { source: MouseWheelSource::Trackpad, phase: TouchPhase::Move, .. }));
                if moved {
                    self.wheel_end = Some(modifiers);
                } else {
                    self.events.push(event(MouseWheelUnit::Point, egui::Vec2::ZERO, TouchPhase::End));
                }
                return;
            }
            _ => {}
        }
        if delta != egui::Vec2::ZERO {
            self.events.push(event(MouseWheelUnit::Point, delta, TouchPhase::Move));
        }
    }

    fn key(&mut self, pressed: bool, code: i32, raw_char: i32, bits: u32, text: &str) {
        self.set_modifiers(bits);
        let (logical, physical) = keys::key(code, raw_char, text.as_bytes());
        log::debug!(target: "lightcraft_haiku", "key {} code {code:#x} raw {raw_char:#x} modifiers {bits:#x} {text:?}: {logical:?} / {physical:?}", if pressed { "down" } else { "up" });
        let m = self.modifiers;
        if let Some(active) = logical.or(physical) {
            if pressed {
                // As egui's own integrations do: the clipboard shortcuts become clipboard events.
                if m.command && active == Key::X || m.shift && active == Key::Delete {
                    self.events.push(Event::Cut);
                    return;
                }
                if m.command && (active == Key::C || active == Key::Insert) {
                    self.events.push(Event::Copy);
                    return;
                }
                if m.command && active == Key::V || m.shift && active == Key::Insert {
                    if let Some(t) = services::clipboard_text().map(|t| t.replace("\r\n", "\n")) {
                        self.events.push(Event::Paste(t));
                    }
                    return;
                }
            }
            self.events.push(Event::Key { key: active, physical_key: physical, pressed, repeat: false, modifiers: m });
        }
        if pressed && let Some(t) = keys::text(bits, text) {
            self.events.push(Event::Text(t.to_string()));
        }
    }

    fn viewport_info(&self, close: bool) -> ViewportInfo {
        let w = host::window_info();
        let k = self.dev / self.ppp;
        let rect = |r: [f32; 4]| egui::Rect::from_min_max(pos2(r[0], r[1]) * k, pos2(r[2], r[3]) * k);
        ViewportInfo {
            title: Some(self.title.clone()),
            events: if close { vec![ViewportEvent::Close] } else { Vec::new() },
            native_pixels_per_point: Some(self.native_ppp),
            monitor_size: (w.screen[0] > 0.0).then(|| vec2(w.screen[0], w.screen[1]) * k),
            inner_rect: Some(rect(w.inner)),
            outer_rect: Some(rect(w.outer)),
            minimized: Some(w.minimized),
            maximized: Some(false),
            fullscreen: Some(w.fullscreen),
            focused: Some(w.active),
            ..Default::default()
        }
    }

    /// The desktop app's per-frame logic (`apps/lightcraft`): the app's own, and the window's
    /// close button, including the editor's unsaved-changes handling.
    fn logic(&mut self, ctx: &egui::Context) {
        self.app.logic(ctx);
        self.writer.tick(&mut self.app, ctx);
    }

    fn frame(&mut self) {
        // Window device faults switch presentation to software. Photo compute has
        // a separate device and its own failure recovery.
        if let Some(fault) = self.gpu_health.as_ref().and_then(|h| h.fault()) {
            log::warn!(target: "lightcraft_haiku", "{fault}; switching native host to CPU");
            self.app.notices.push(format!("{fault}. The window now renders on the CPU."));
            self.window_recovery.record(&fault);
            self.window_lost = true;
            self.record_gpu_lost();
            self.gpu = None;
            self.gpu_health = None;
            self.app.perf.host_renderer = Some("CPU bitmap");
            self.ctx.request_repaint();
        }

        if self.dev_checked.elapsed() > Duration::from_secs(2) {
            self.check_density();
        }
        let (lw, lh) = host::view_size();
        let (w, h) = ((lw * self.dev).round(), (lh * self.dev).round());
        if w < 1.0 || h < 1.0 {
            self.repaint_at = Some(Instant::now() + Duration::from_millis(50));
            return;
        }
        let close = std::mem::take(&mut self.close_requested);
        let raw = egui::RawInput {
            viewport_id: ViewportId::ROOT,
            viewports: std::iter::once((ViewportId::ROOT, self.viewport_info(close))).collect(),
            screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, vec2(w, h) / self.ppp)),
            max_texture_side: Some(self.gpu.as_ref().map_or(8192, |g| g.max_texture_side())),
            time: Some(self.start.elapsed().as_secs_f64()),
            predicted_dt: 1.0 / 60.0,
            events: std::mem::take(&mut self.events),
            focused: self.focused,
            system_theme: Some(if host::dark() { Theme::Dark } else { Theme::Light }),
            ..Default::default()
        };
        let mut raw = raw;
        let ctx = self.ctx.clone();
        self.app.raw_input_hook(&ctx, &mut raw);
        let t0 = Instant::now();
        let mut out = ctx.run_ui(raw, |ui| {
            let ctx = ui.ctx().clone();
            self.logic(&ctx);
            self.app.ui(ui);
        });

        // The system side of egui's output.
        host::set_cursor(cursor_id(out.platform_output.cursor_icon));
        for cmd in &out.platform_output.commands {
            log::debug!(target: "lightcraft_haiku", "output {cmd:?}");
            match cmd {
                egui::OutputCommand::CopyText(t) => {
                    if !host::clipboard_set(t, None) {
                        log::warn!(target: "lightcraft_haiku", "clipboard: couldn't copy");
                    }
                }
                egui::OutputCommand::CopyImage(img) => copy_image(img),
                // Haiku has no PRIMARY selection to put it in
                egui::OutputCommand::TextSelectionSettled(_) => {}
                egui::OutputCommand::OpenUrl(u) => {
                    if !host::open_url(&u.url) {
                        log::warn!(target: "lightcraft_haiku", "couldn't open {}", u.url);
                    }
                }
            }
        }
        let mut delay = Duration::MAX;
        if let Some(vp) = out.viewport_output.get(&ViewportId::ROOT) {
            delay = vp.repaint_delay;
            if let Some(t) = &vp.builder.title
                && *t != self.title
            {
                self.title.clone_from(t);
                host::set_title(t);
            }
            // Honor the editor's unsaved-changes prompt before accepting a native close.
            let requested = close || vp.commands.iter().any(|c| matches!(c, ViewportCommand::Close));
            if requested && !vp.commands.iter().any(|c| matches!(c, ViewportCommand::CancelClose)) {
                self.quit = true;
            }
            for cmd in &vp.commands {
                self.command(cmd);
            }
        }

        // Draw.
        let ppp = out.pixels_per_point;
        self.raster.set_textures(&out.textures_delta);
        let minimized = host::window_info().minimized;
        if (!minimized || self.gpu.is_some() || !self.screenshots.is_empty()) && !self.gpu_health.as_ref().is_some_and(|h| h.fault().is_some()) {
            let t1 = Instant::now();
            let prims = ctx.tessellate(out.shapes, ppp);
            let t2 = Instant::now();
            let clear = ctx.global_style().visuals.panel_fill;
            let (pw, ph) = (w as usize, h as usize);
            let dev = self.dev;
            // The presenter shows GPU frames itself; the software rasteriser's go below.
            let dirty = if let Some(gpu) = &mut self.gpu {
                match crate::guard("GPU frame", || {
                    gpu.render([pw as u32, ph as u32], prims, &out.textures_delta, ppp, clear, dev, &crate::gpu::Native)
                }) {
                    Ok(_) => None,
                    Err(e) => {
                        log::error!(target: "lightcraft_haiku", "{e}");
                        if let Some(health) = &self.gpu_health {
                            health.mark(e);
                        }
                        ctx.request_repaint();
                        None
                    }
                }
            } else {
                self.raster.render([pw, ph], &prims, ppp, clear)
            };
            let t3 = Instant::now();
            self.app.perf.host_render_ms = Some((t3 - t2).as_secs_f64() * 1e3);
            let faulted = self.gpu_health.as_ref().is_some_and(|h| h.fault().is_some());
            if faulted {
                ctx.request_repaint();
                delay = Duration::ZERO;
            }
            if let Some(dirty) = dirty.filter(|_| !faulted && !minimized && self.gpu.is_none()) {
                host::present(self.raster.pixels(), pw, ph, dirty, self.dev);
            }
            log::debug!(target: "lightcraft_haiku",
                "frame {pw}x{ph}: ui {:.1} ms, tessellate {:.1} ms, raster {:.1} ms, present {:.1} ms, dirty {dirty:?}",
                (t1 - t0).as_secs_f64() * 1e3,
                (t2 - t1).as_secs_f64() * 1e3,
                (t3 - t2).as_secs_f64() * 1e3,
                t3.elapsed().as_secs_f64() * 1e3
            );
        }
        self.raster.free_textures(&out.textures_delta);
        // Applied: epaint checks (in debug builds) that no texture update is dropped unhandled.
        out.textures_delta.clear();
        if !self.gpu_health.as_ref().is_some_and(|h| h.fault().is_some()) {
            for callback in std::mem::take(&mut self.screenshots) {
                // egui 0.37 answers a screenshot request through its one-shot callback
                let image = match self.gpu.as_mut() {
                    Some(gpu) => gpu.image().unwrap_or_else(|e| {
                        log::warn!(target: "lightcraft_haiku", "screenshot: {e}");
                        egui::ColorImage::default()
                    }),
                    None => self.raster.image(),
                };
                callback.complete(Arc::new(image));
                delay = Duration::ZERO;
            }
        } else {
            delay = Duration::ZERO;
        }
        if (ppp - self.ppp).abs() > f32::EPSILON {
            self.ppp = ppp;
            delay = Duration::ZERO;
        }
        if !self.events.is_empty() || self.wheel_end.is_some() {
            delay = Duration::ZERO;
        }
        self.app.perf.host_frame_ms = Some(t0.elapsed().as_secs_f64() * 1e3);
        self.repaint_at = (delay < Duration::from_secs(3600)).then(|| Instant::now() + delay);
    }

    fn command(&mut self, cmd: &ViewportCommand) {
        // Points to logical pixels.
        let ppp = self.ppp / self.dev;
        match cmd {
            ViewportCommand::Title(t) => {
                if *t != self.title {
                    self.title.clone_from(t);
                    host::set_title(t);
                }
            }
            ViewportCommand::InnerSize(s) => host::window_command(WindowCommand::ResizeTo(s.x * ppp, s.y * ppp)),
            ViewportCommand::MinInnerSize(s) => host::window_command(WindowCommand::MinSize(s.x * ppp, s.y * ppp)),
            ViewportCommand::OuterPosition(p) => host::window_command(WindowCommand::MoveTo(p.x * ppp, p.y * ppp)),
            ViewportCommand::Minimized(m) => host::window_command(WindowCommand::Minimize(*m)),
            ViewportCommand::Maximized(true) => host::window_command(WindowCommand::Zoom),
            ViewportCommand::Fullscreen(on) => host::window_command(WindowCommand::Fullscreen(*on)),
            ViewportCommand::Focus => host::window_command(WindowCommand::Activate),
            ViewportCommand::Screenshot(callback) => self.screenshots.push(callback.clone()),
            ViewportCommand::RequestCut => self.events.push(Event::Cut),
            ViewportCommand::RequestCopy => self.events.push(Event::Copy),
            ViewportCommand::RequestPaste => {
                if let Some(t) = services::clipboard_text() {
                    self.events.push(Event::Paste(t.replace("\r\n", "\n")));
                }
            }
            _ => {}
        }
    }

    /// Where the desktop app stops after the graphics driver reset the GPU (its window renderer
    /// panics on the lost device) and records it with `record_gpu_lost`, the window here falls
    /// back to the CPU and goes on. A reset takes the photo pipeline's device too: once both the
    /// window's device failed and the photo pipeline stopped after a GPU failure, the same marker
    /// is written, so the next launch renders photos on the CPU and says why, as the desktop
    /// app's does (`gpu_lost_check_at` removes it then). The photo pipeline only notices at its
    /// next render, so this is asked again at quit.
    fn record_gpu_lost(&mut self) {
        if !self.window_lost {
            return;
        }
        if let Some(reason) = crate::gpu_crash::gpu_lost_reason()
            && let Some(marker) = self.gpu_lost.take()
        {
            log::warn!(target: "lightcraft_haiku", "the GPU was lost ({reason}); the next launch renders photos on the CPU");
            crate::gpu_crash::record_gpu_lost(Some(&marker), &reason);
        }
    }

    /// What quitting does besides closing the window (`on_exit` in the desktop app).
    pub fn finish(&mut self) {
        self.record_gpu_lost();
        self.app.ui.window = self.writer.window_at_exit(self.app.ui.window);
        if let Err(e) = self.writer.save(&self.app) {
            log::error!("{e}");
        }
        if let Err(e) = self.app.session.close_library() {
            log::error!("Saving the library failed: {e}");
        }
        if !self.app.shutdown(Duration::from_secs(5)) {
            log::warn!("Background work was still finishing at quit");
        }
    }
}

/// egui's image copy, as PNG.
fn copy_image(img: &egui::ColorImage) {
    let [w, h] = img.size;
    let rgba: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_srgba_unmultiplied()).collect();
    let mut png = Vec::new();
    let encoded = image::RgbaImage::from_raw(w as u32, h as u32, rgba)
        .ok_or_else(|| "bad size".to_string())
        .and_then(|i| i.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).map_err(|e| e.to_string()));
    match encoded {
        Ok(()) => {
            if !host::clipboard_set_picture("image/png", &png) {
                log::warn!(target: "lightcraft_haiku", "clipboard: couldn't copy the picture");
            }
        }
        Err(e) => log::warn!(target: "lightcraft_haiku", "clipboard picture: {e}"),
    }
}
