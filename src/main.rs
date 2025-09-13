use bytemuck::cast_slice;
use crossbeam_channel::bounded;
use signal_hook::consts::SIGUSR2;
use signal_hook::iterator::Signals;
use std::io::Read;
use std::process::Command;
use std::sync::atomic::AtomicU32;
use std::thread;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

static PID: AtomicU32 = AtomicU32::new(0);

fn get_focused_window_app_id(sway: &mut swayipc::Connection) -> Option<String> {
    fn get_focused(node: &swayipc::Node) -> Option<String> {
        if node.focused {
            return node.app_id.clone();
        }
        for child in &node.nodes {
            if let Some(found) = get_focused(&child) {
                return Some(found.into());
            }
        }
        None
    }
    get_focused(&sway.get_tree().expect("error getting sway tree"))
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .expect("Please specify path to model as argument 1");
    let print_text = std::env::var("PRINT_TEXT").unwrap_or_default() == "1";
    let keep_audio = std::env::var("KEEP_AUDIO").unwrap_or_default() == "1";

    let mut params = WhisperContextParameters::default();
    params.flash_attn = true;
    let ctx = WhisperContext::new_with_params(&model_path, params).expect("failed to load model");

    let mut state = ctx.create_state().expect("failed to create state");
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    params.set_n_threads(8);
    params.set_no_context(true);
    params.set_no_timestamps(true);
    params.set_print_progress(false);
    params.set_print_timestamps(false);
    params.set_single_segment(true);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);

    let mut sway = swayipc::Connection::new()?;

    // USR2 signal will toggle recording
    let (send_toggle, recv_toggle) = bounded(100);
    let mut signals = Signals::new([SIGUSR2])?;
    thread::spawn(move || {
        for _ in signals.forever() {
            let _ = send_toggle.send(());
        }
    });

    let mut command = None;
    loop {
        let _ = recv_toggle.recv()?;
        match command.take() {
            None => {
                command = Some(thread::spawn(|| {
                    let mut child = Command::new("pw-record")
                        .args(["--format=f32", "--rate=16000", "--channels=1", "/tmp/a.au"])
                        .spawn()
                        .expect("failed to spawn pw-record");
                    PID.store(child.id(), std::sync::atomic::Ordering::SeqCst);
                    child.wait().expect("pw-record did not exit cleanly");
                }));
            }
            Some(child) => {
                unsafe {
                    libc::kill(
                        PID.load(std::sync::atomic::Ordering::SeqCst) as i32,
                        libc::SIGINT,
                    );
                }
                child.join().expect("pw-record to finish");

                let mut file = std::fs::File::open("/tmp/a.au")?;
                let mut output = Vec::new();
                file.read_to_end(&mut output)?;
                if !keep_audio {
                    std::fs::remove_file("/tmp/a.au")?;
                }
                state
                    .full(params.clone(), cast_slice(&output))
                    .expect("failed to run model");

                let use_paste_mode =
                    get_focused_window_app_id(&mut sway).unwrap_or_default() == "firefox";

                for segment in state.as_iter() {
                    let text = format!("{segment}");
                    let text = text.trim();
                    if print_text {
                        println!("{text}");
                    }
                    if use_paste_mode {
                        Command::new("wl-copy")
                            .arg(text)
                            .status()
                            .expect("failed to execute wl-copy");
                        Command::new("ydotool")
                            .args(["key", "29:1", "47:1", "47:0", "29:0"])
                            .status()
                            .expect("failed to execute ydotool");
                    } else {
                        Command::new("ydotool")
                            .args(["type", "-d=8", "-H=6"])
                            .arg(text)
                            .status()
                            .expect("failed to execute ydotool");
                    }
                }
            }
        }
    }
}
