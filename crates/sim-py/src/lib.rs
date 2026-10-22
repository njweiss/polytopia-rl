//! Python module `polytopia_sim` (docs/MILESTONE_9 "Python layer"): pure type conversion around
//! `sim_env::env::Env`. No game logic lives here.

use numpy::{PyArray1, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(name = "Env", unsendable)]
pub struct PyEnv {}

#[pymethods]
impl PyEnv {
    #[new]
    #[pyo3(signature = (
        size = 11,
        turn_cap = 30,
        full_observability = false,
        opponent = "rush",
        opponent_epsilon = 0.0,
        agent_seat = None,
        terminal_weight = 1.0,
        shaping_weight = 0.0,
        gamma = 1.0,
        potential_scale = 1000.0,
        max_agent_actions = 5000,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        size: u8,
        turn_cap: u32,
        full_observability: bool,
        opponent: &str,
        opponent_epsilon: f32,
        agent_seat: Option<Bound<'_, PyAny>>,
        terminal_weight: f32,
        shaping_weight: f32,
        gamma: f32,
        potential_scale: f32,
        max_agent_actions: u32,
    ) -> PyResult<PyEnv> {
        let _ = (
            size,
            turn_cap,
            full_observability,
            opponent,
            opponent_epsilon,
            agent_seat,
            terminal_weight,
            shaping_weight,
            gamma,
            potential_scale,
            max_agent_actions,
        );
        unimplemented!("milestone 9")
    }

    fn reset(&mut self, seed: u64) -> PyResult<()> {
        let _ = seed;
        unimplemented!("milestone 9")
    }

    /// `(board float32[C,H,W], globals float32[G])`.
    #[allow(clippy::type_complexity)]
    fn observation<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray1<f32>>)> {
        let _ = py;
        unimplemented!("milestone 9")
    }

    fn actor_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let _ = py;
        unimplemented!("milestone 9")
    }

    fn kind_mask<'py>(
        &self,
        py: Python<'py>,
        actor: usize,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let _ = (py, actor);
        unimplemented!("milestone 9")
    }

    fn arg_mask<'py>(
        &self,
        py: Python<'py>,
        actor: usize,
        kind: usize,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let _ = (py, actor, kind);
        unimplemented!("milestone 9")
    }

    fn legal_count(&self) -> PyResult<usize> {
        unimplemented!("milestone 9")
    }

    /// `(reward, rewards float32[2], terminated, truncated)`.
    fn step_legal<'py>(
        &mut self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<(f32, Bound<'py, PyArray1<f32>>, bool, bool)> {
        let _ = (py, index);
        unimplemented!("milestone 9")
    }

    fn step_factored<'py>(
        &mut self,
        py: Python<'py>,
        actor: usize,
        kind: usize,
        arg: usize,
    ) -> PyResult<(f32, Bound<'py, PyArray1<f32>>, bool, bool)> {
        let _ = (py, actor, kind, arg);
        unimplemented!("milestone 9")
    }

    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let _ = py;
        unimplemented!("milestone 9")
    }

    fn replay_json(&self) -> PyResult<String> {
        unimplemented!("milestone 9")
    }

    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let _ = py;
        unimplemented!("milestone 9")
    }
}

#[pymodule]
fn polytopia_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnv>()?;
    Ok(())
}
