//! The `polytopia` CLI with hand-parsed arguments (docs/MILESTONE_7 "CLI").

use std::io::Write;

use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::state::{Config, Outcome, State};

use crate::html::html_page;
use crate::policy::random_action;
use crate::render::{action_label, render, View};
use crate::replay::{record_game, replay, replay_each, ReplayLog};

const USAGE: &str = "usage:
  polytopia play-random --seed N --out FILE [--turn-cap K] [--max-actions M] [--full-observability]
  polytopia verify FILE
  polytopia view FILE [--step I | --all] [--player P]
  polytopia html FILE OUT.html
";

/// XORed into the seed for the policy's stream so it differs from the map's.
const POLICY_SEED_XOR: u64 = 0x9e37_79b9_7f4a_7c15;

/// A command's failure: exit 1 (a verify/replay or file failure) or exit 2 (usage).
enum Fail {
    Run(String),
    Usage(String),
}

fn usage<T>(msg: &str) -> Result<T, Fail> {
    Err(Fail::Usage(msg.to_string()))
}

pub fn run_cli(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let result = match args.first().map(String::as_str) {
        Some("play-random") => play_random(&args[1..], out),
        Some("verify") => verify(&args[1..], out),
        Some("view") => view(&args[1..], out),
        Some("html") => html(&args[1..]),
        Some(other) => usage(&format!("unknown subcommand {other:?}")),
        None => usage("missing subcommand"),
    };
    match result {
        Ok(()) => 0,
        Err(Fail::Run(msg)) => {
            let _ = writeln!(err, "error: {msg}");
            1
        }
        Err(Fail::Usage(msg)) => {
            let _ = write!(err, "error: {msg}\n{USAGE}");
            2
        }
    }
}

/// The value following option `name`, parsed.
fn value<T: std::str::FromStr>(it: &mut std::slice::Iter<String>, name: &str) -> Result<T, Fail> {
    match it.next().map(|v| v.parse()) {
        Some(Ok(v)) => Ok(v),
        Some(Err(_)) => usage(&format!("invalid value for {name}")),
        None => usage(&format!("missing value for {name}")),
    }
}

fn outcome_text(o: Option<Outcome>) -> String {
    match o {
        Some(Outcome::Win(p)) => format!("Win({p})"),
        Some(Outcome::Draw) => "Draw".to_string(),
        None => "unfinished".to_string(),
    }
}

fn io_fail(e: std::io::Error) -> Fail {
    Fail::Run(e.to_string())
}

fn load(path: &str) -> Result<ReplayLog, Fail> {
    let text = std::fs::read_to_string(path).map_err(|e| Fail::Run(format!("{path}: {e}")))?;
    ReplayLog::from_json(&text).map_err(|e| Fail::Run(format!("{path}: {e:?}")))
}

fn play_random(args: &[String], out: &mut dyn Write) -> Result<(), Fail> {
    let mut seed: Option<u64> = None;
    let mut path: Option<String> = None;
    let mut turn_cap: u32 = 50;
    let mut max_actions: usize = 5000;
    let mut full_observability = false;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--seed" => seed = Some(value(&mut it, "--seed")?),
            "--out" => path = Some(value(&mut it, "--out")?),
            "--turn-cap" => turn_cap = value(&mut it, "--turn-cap")?,
            "--max-actions" => max_actions = value(&mut it, "--max-actions")?,
            "--full-observability" => full_observability = true,
            other => return usage(&format!("unknown option {other:?}")),
        }
    }
    let Some(seed) = seed else {
        return usage("play-random needs --seed");
    };
    let Some(path) = path else {
        return usage("play-random needs --out");
    };
    if turn_cap == 0 {
        return usage("--turn-cap must be at least 1");
    }
    let config = Config {
        full_observability,
        turn_cap,
        ..Config::default()
    };
    let mut rng = Rng::new(seed ^ POLICY_SEED_XOR);
    let log = record_game(
        seed,
        &MapParams::default(),
        &config,
        max_actions,
        &mut |s: &State| random_action(s, &mut rng),
    );
    std::fs::write(&path, log.to_json()).map_err(|e| Fail::Run(format!("{path}: {e}")))?;
    writeln!(
        out,
        "wrote {path}: {} actions, outcome {}",
        log.actions.len(),
        outcome_text(log.outcome)
    )
    .map_err(io_fail)
}

fn verify(args: &[String], out: &mut dyn Write) -> Result<(), Fail> {
    let [path] = args else {
        return usage("verify takes exactly one FILE");
    };
    let log = load(path)?;
    replay(&log).map_err(|e| Fail::Run(format!("{path}: {e:?}")))?;
    writeln!(
        out,
        "OK: {} actions, outcome {}",
        log.actions.len(),
        outcome_text(log.outcome)
    )
    .map_err(io_fail)
}

fn view(args: &[String], out: &mut dyn Write) -> Result<(), Fail> {
    let Some((path, rest)) = args.split_first() else {
        return usage("view needs a FILE");
    };
    let mut step: Option<usize> = None;
    let mut all = false;
    let mut view = View::Omniscient;
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--step" => step = Some(value(&mut it, "--step")?),
            "--all" => all = true,
            "--player" => match value::<u8>(&mut it, "--player")? {
                p @ (0 | 1) => view = View::Player(p),
                _ => return usage("--player must be 0 or 1"),
            },
            other => return usage(&format!("unknown option {other:?}")),
        }
    }
    if all && step.is_some() {
        return usage("--step and --all are exclusive");
    }
    let log = load(path)?;
    let mut states = Vec::with_capacity(log.actions.len() + 1);
    replay_each(&log, &mut |_, s| states.push(s.clone()))
        .map_err(|e| Fail::Run(format!("{path}: {e:?}")))?;
    let n = log.actions.len();
    let text = if all {
        let mut text = String::new();
        for (i, s) in states.iter().enumerate() {
            let label = match i {
                0 => "start".to_string(),
                _ => action_label(&log.actions[i - 1]),
            };
            text.push_str(&format!("== step {i}: {label} ==\n"));
            text.push_str(&render(s, view));
        }
        text
    } else {
        let i = step.unwrap_or(n);
        if i > n {
            return usage(&format!("--step {i} is past the last step {n}"));
        }
        render(&states[i], view)
    };
    out.write_all(text.as_bytes()).map_err(io_fail)
}

fn html(args: &[String]) -> Result<(), Fail> {
    let [path, page_path] = args else {
        return usage("html takes FILE and OUT.html");
    };
    let log = load(path)?;
    let page = html_page(&log).map_err(|e| Fail::Run(format!("{path}: {e:?}")))?;
    std::fs::write(page_path, page).map_err(|e| Fail::Run(format!("{page_path}: {e}")))
}
