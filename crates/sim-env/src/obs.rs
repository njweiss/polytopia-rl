//! Observation encoding (docs/MILESTONE_9 "Observation"). Reads only a `PlayerView`.

use sim_core::view::PlayerView;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObsSpec {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub globals: usize,
    pub channel_names: Vec<String>,
    pub global_names: Vec<String>,
}

pub fn obs_spec(size: u8) -> ObsSpec {
    let _ = size;
    unimplemented!("milestone 9")
}

/// Channel-major `(C, H, W)`, length `channels * height * width`; overwrites all of `out`.
pub fn encode_board(view: &PlayerView, out: &mut [f32]) {
    let _ = (view, out);
    unimplemented!("milestone 9")
}

/// Length `globals`; overwrites all of `out`.
pub fn encode_globals(view: &PlayerView, out: &mut [f32]) {
    let _ = (view, out);
    unimplemented!("milestone 9")
}
