//! Python module `polytopia_sim` (docs/MILESTONE_9 "Python layer"): pure type conversion around
//! `sim_env::env::Env`. No game logic lives here.

use bots::BotKind;
use numpy::{PyArray1, PyArray3, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sim_core::data::{ALL_REWARDS, ALL_SITE_ACTIONS, ALL_TECHS, ALL_UNITS};
use sim_core::map::MapParams;
use sim_core::state::{Config, Outcome, Ruleset};
use sim_env::actions::{arg_len, num_actors, ActionType, Factored};
use sim_env::env::{AgentSeat, Env, EnvConfig, EnvError, Opponent, RewardConfig, StepOut};
use sim_env::hash::{obs_hash, rules_hash};
use sim_env::obs::{
    encode_board_with_legal, encode_globals, obs_spec, snake_name, CHANNELS, GLOBALS,
};

fn py_err(e: EnvError) -> PyErr {
    match e {
        EnvError::NotReset | EnvError::Finished => PyRuntimeError::new_err(format!("{e:?}")),
        EnvError::Illegal | EnvError::BadIndex | EnvError::Decode(_) => {
            PyValueError::new_err(format!("{e:?}"))
        }
    }
}

fn kind_of(kind: usize) -> PyResult<ActionType> {
    ActionType::ALL
        .get(kind)
        .copied()
        .ok_or_else(|| PyValueError::new_err(format!("no action type {kind}")))
}

type StepResult<'py> = (f32, Bound<'py, PyArray1<f32>>, bool, bool);

fn step_result(py: Python<'_>, out: StepOut) -> StepResult<'_> {
    (
        out.reward,
        PyArray1::from_slice(py, &out.rewards),
        out.terminated,
        out.truncated,
    )
}

fn snake_names<T: std::fmt::Debug + Copy>(xs: &[T]) -> Vec<String> {
    xs.iter().map(|&x| snake_name(x)).collect()
}

#[pyclass(name = "Env", unsendable)]
pub struct PyEnv {
    env: Env,
    size: u8,
}

impl PyEnv {
    /// The env, or `RuntimeError` (as for `NotReset` from a step) before the first `reset`.
    fn env(&self) -> PyResult<&Env> {
        if !self.env.is_reset() {
            return Err(py_err(EnvError::NotReset));
        }
        Ok(&self.env)
    }

    fn check_actor(&self, actor: usize) -> PyResult<()> {
        if actor >= num_actors(self.size) {
            return Err(PyValueError::new_err(format!("no actor {actor}")));
        }
        Ok(())
    }
}

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
        ruleset = "v1",
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
        ruleset: &str,
    ) -> PyResult<PyEnv> {
        if size < MapParams::MIN_SIZE {
            return Err(PyValueError::new_err(format!(
                "map size {size} is below {}",
                MapParams::MIN_SIZE
            )));
        }
        let ruleset = match ruleset {
            "v1" => Ruleset::V1,
            "v2" => Ruleset::V2,
            other => return Err(PyValueError::new_err(format!("unknown ruleset {other:?}"))),
        };
        if ruleset != Ruleset::V1 {
            return Err(PyValueError::new_err(format!(
                "ruleset {ruleset:?} is not supported until milestone 20 (the observation is v1 only)"
            )));
        }
        let bot = |kind| Opponent::Bot {
            kind,
            epsilon: opponent_epsilon,
        };
        let opponent = match (opponent, BotKind::from_name(opponent)) {
            ("none", _) => Opponent::TwoSeat,
            (_, Some(kind)) => bot(kind),
            (other, None) => {
                return Err(PyValueError::new_err(format!("unknown opponent {other:?}")))
            }
        };
        let agent_seat = match agent_seat {
            None => AgentSeat::Fixed(0),
            Some(s) => {
                if let Ok(name) = s.extract::<String>() {
                    if name != "seed_parity" {
                        return Err(PyValueError::new_err(format!("unknown seat {name:?}")));
                    }
                    AgentSeat::SeedParity
                } else {
                    match s.extract::<u8>()? {
                        p @ (0 | 1) => AgentSeat::Fixed(p),
                        p => return Err(PyValueError::new_err(format!("no seat {p}"))),
                    }
                }
            }
        };
        let cfg = EnvConfig {
            map: MapParams {
                size,
                ..MapParams::default()
            },
            config: Config {
                full_observability,
                turn_cap,
                ruleset,
                ..Config::default()
            },
            opponent,
            agent_seat,
            reward: RewardConfig {
                terminal_weight,
                shaping_weight,
                gamma,
                potential_scale,
            },
            max_agent_actions,
        };
        Ok(PyEnv {
            env: Env::new(cfg),
            size,
        })
    }

    fn reset(&mut self, seed: u64) -> PyResult<()> {
        self.env.reset(seed);
        Ok(())
    }

    /// `(board float32[C,H,W], globals float32[G])`.
    #[allow(clippy::type_complexity)]
    fn observation<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray1<f32>>)> {
        let n = self.size as usize;
        let env = self.env()?;
        let view = env.view();
        let mut board = vec![0.0f32; CHANNELS * n * n];
        encode_board_with_legal(&view, env.legal_actions(), &mut board);
        let mut globals = vec![0.0f32; GLOBALS];
        encode_globals(&view, &mut globals);
        let board = PyArray1::from_vec(py, board).reshape([CHANNELS, n, n])?;
        Ok((board, PyArray1::from_vec(py, globals)))
    }

    fn actor_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        Ok(PyArray1::from_slice(py, self.env()?.masks().actor()))
    }

    fn kind_mask<'py>(
        &self,
        py: Python<'py>,
        actor: usize,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        self.check_actor(actor)?;
        Ok(PyArray1::from_slice(py, &self.env()?.masks().kind(actor)))
    }

    fn arg_mask<'py>(
        &self,
        py: Python<'py>,
        actor: usize,
        kind: usize,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        self.check_actor(actor)?;
        let kind = kind_of(kind)?;
        Ok(PyArray1::from_slice(
            py,
            self.env()?.masks().arg(actor, kind),
        ))
    }

    fn legal_count(&self) -> PyResult<usize> {
        Ok(self.env()?.legal_actions().len())
    }

    /// `(reward, rewards float32[2], terminated, truncated)`.
    fn step_legal<'py>(&mut self, py: Python<'py>, index: usize) -> PyResult<StepResult<'py>> {
        let out = self.env.step_legal(index).map_err(py_err)?;
        Ok(step_result(py, out))
    }

    fn step_factored<'py>(
        &mut self,
        py: Python<'py>,
        actor: usize,
        kind: usize,
        arg: usize,
    ) -> PyResult<StepResult<'py>> {
        let kind = kind_of(kind)?;
        let out = self
            .env
            .step_factored(&Factored { actor, kind, arg })
            .map_err(py_err)?;
        Ok(step_result(py, out))
    }

    /// `(actor, kind, arg)` of `sim_env::env::Env::bot_action`; `ValueError` for an unknown bot
    /// name, `RuntimeError` once finished.
    fn bot_action(&mut self, name: &str) -> PyResult<(usize, usize, usize)> {
        let kind = BotKind::from_name(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown bot {name:?}")))?;
        let f = self.env.bot_action(kind).map_err(py_err)?;
        Ok((f.actor, f.kind as usize, f.arg))
    }

    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let env = self.env()?;
        let view = env.view();
        let outcome = match env.state().outcome {
            None => None,
            Some(Outcome::Win(p)) => Some(format!("win_{p}")),
            Some(Outcome::Draw) => Some("draw".to_string()),
        };
        let d = PyDict::new(py);
        d.set_item("seed", env.seed())?;
        d.set_item("turn", view.turn())?;
        d.set_item("current_player", view.current())?;
        d.set_item("observer", env.observer())?;
        d.set_item("agent_seat", env.agent_seat())?;
        d.set_item("scores", (view.score(0), view.score(1)))?;
        d.set_item("outcome", outcome)?;
        d.set_item("agent_actions", env.agent_actions())?;
        Ok(d)
    }

    fn replay_json(&self) -> PyResult<String> {
        Ok(self.env()?.replay_json())
    }

    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let spec = obs_spec(self.size);
        let d = PyDict::new(py);
        d.set_item("height", spec.height)?;
        d.set_item("width", spec.width)?;
        d.set_item("channels", spec.channels)?;
        d.set_item("globals", spec.globals)?;
        d.set_item("channel_names", spec.channel_names)?;
        d.set_item("global_names", spec.global_names)?;
        d.set_item("action_types", snake_names(&ActionType::ALL))?;
        d.set_item("num_actors", num_actors(self.size))?;
        let arg_lens: Vec<usize> = ActionType::ALL
            .iter()
            .map(|&k| arg_len(self.size, k))
            .collect();
        d.set_item("arg_lens", arg_lens)?;
        d.set_item("unit_types", snake_names(&ALL_UNITS))?;
        d.set_item("rewards", snake_names(&ALL_REWARDS))?;
        d.set_item("techs", snake_names(&ALL_TECHS))?;
        d.set_item("site_actions", snake_names(&ALL_SITE_ACTIONS))?;
        d.set_item("obs_hash", obs_hash(self.size))?;
        d.set_item("rules_hash", rules_hash())?;
        Ok(d)
    }
}

#[pymodule]
fn polytopia_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnv>()?;
    Ok(())
}
