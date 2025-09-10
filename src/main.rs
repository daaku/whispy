use bytemuck::cast_slice;
use crossbeam_channel::bounded;
use signal_hook::consts::SIGUSR2;
use signal_hook::iterator::Signals;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicU32;
use std::thread;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

static PID: AtomicU32 = AtomicU32::new(0);

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .expect("Please specify path to model as argument 1");

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
                        .args(["--format=f32", "--rate=16000", "--channels=1", "-"])
                        .stdout(Stdio::piped())
                        .spawn()
                        .expect("failed to spawn pw-record");
                    PID.store(child.id(), std::sync::atomic::Ordering::SeqCst);
                    let mut stdout = Vec::new();
                    if let Some(mut out) = child.stdout.take() {
                        out.read_to_end(&mut stdout)
                            .expect("failed to read pw-record output");
                    }
                    child.wait().expect("pw-record did not exit cleanly");
                    stdout
                }));
            }
            Some(child) => {
                unsafe {
                    libc::kill(
                        PID.load(std::sync::atomic::Ordering::SeqCst) as i32,
                        libc::SIGINT,
                    );
                }
                let output = child.join().expect("pw-record to finish");

                let mut file = std::fs::File::create("audio.pcm")?;
                file.write_all(&output)?;

                state
                    .full(params.clone(), cast_slice(&output))
                    .expect("failed to run model");

                for segment in state.as_iter() {
                    println!("{}", segment);
                }
            }
        }
    }
}
