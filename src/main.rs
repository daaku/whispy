// This example is not going to build in this folder.
// You need to copy this code into your project and add the dependencies whisper_rs and hound in your cargo.toml

use bytemuck::cast_slice;
use std::io::{self, Write};
use std::{fs::File, io::Read};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Loads a context and model, processes an audio file, and prints the resulting transcript to stdout.
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

    let mut context_param = WhisperContextParameters::default();

    // Enable DTW token level timestamp for known model by using model preset
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset {
        model_preset: whisper_rs::DtwModelPreset::BaseEn,
    };

    // Enable DTW token level timestamp for unknown model by providing custom aheads
    // see details https://github.com/ggerganov/whisper.cpp/pull/1485#discussion_r1519681143
    // values corresponds to ggml-base.en.bin, result will be the same as with DtwModelPreset::BaseEn
    let custom_aheads = [
        (3, 1),
        (4, 2),
        (4, 3),
        (4, 7),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 6),
    ]
    .map(|(n_text_layer, n_head)| whisper_rs::DtwAhead {
        n_text_layer,
        n_head,
    });
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::Custom {
        aheads: &custom_aheads,
    };

    // Load a context and model
    let ctx =
        WhisperContext::new_with_params(&model_path, context_param).expect("failed to load model");
    // Create a state
    let mut state = ctx.create_state().expect("failed to create state");

    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

    // Edit params as needed.
    // Set the number of threads to use to 1.
    params.set_n_threads(1);
    // Enable translation.
    params.set_translate(true);
    // Set the language to translate to to English.
    params.set_language(Some("en"));
    // Disable anything that prints to stdout.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Enable token level timestamps
    params.set_token_timestamps(true);

    // Run the model.
    state
        .full(params, cast_slice(&buffer))
        .expect("failed to run model");

    // Create a file to write the transcript to.
    let mut file = File::create("transcript.txt").expect("failed to create file");

    // Iterate through the segments of the transcript.
    for segment in state.as_iter() {
        // Get the transcribed text and timestamps for the current segment.
        let start_timestamp = segment.start_timestamp();
        let end_timestamp = segment.end_timestamp();

        let first_token_dtw_ts = segment.get_token(0).map_or(-1, |t| t.token_data().t_dtw);
        // Print the segment to stdout.
        println!(
            "[{} - {} ({})]: {}",
            start_timestamp, end_timestamp, first_token_dtw_ts, segment
        );

        // Format the segment information as a string.
        let line = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, segment);

        // Write the segment information to the file.
        file.write_all(line.as_bytes())
            .expect("failed to write to file");
    }
    Ok(())
}
