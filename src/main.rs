use bytemuck::cast_slice;
use std::io;
use std::{fs::File, io::Read};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn main() -> io::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .expect("Please specify path to model as argument 1");
    let raw_pcm_path = std::env::args()
        .nth(2)
        .expect("Please specify path to wav file as argument 2");

    let mut file = File::open(raw_pcm_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .expect("failed to load model");
    let mut state = ctx.create_state().expect("failed to create state");

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    params.set_n_threads(1);
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_token_timestamps(false);

    state
        .full(params, cast_slice(&buffer))
        .expect("failed to run model");

    for segment in state.as_iter() {
        println!("{}", segment);
    }
    Ok(())
}
