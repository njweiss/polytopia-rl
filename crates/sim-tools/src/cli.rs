//! The `polytopia` CLI with hand-parsed arguments (docs/MILESTONE_7 "CLI").

use std::io::Write;

use bots::{bot_rng, make_bot, run_match, BotKind};
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::state::{Config, Outcome, State};
use sim_core::view::PlayerView;

use crate::html::html_page;
use crate::policy::random_action;
use crate::render::{action_label, render, View};
use crate::replay::{record_game, replay, replay_each, ReplayLog};

const USAGE: &str = "usage:
  polytopia play-random --seed N --out FILE [--turn-cap K] [--max-actions M] [--full-observability]
  polytopia verify FILE
  polytopia view FILE [--step I | --all] [--player P]
  polytopia html FILE OUT.html
  polytopia play --bots A,B --seed N --out FILE [--turn-cap K] [--max-actions M]
  polytopia tournament --bots LIST [--games N] [--turn-cap K] [--seed S] [--gate-min R]
                       [--save-replays DIR --save-count C]
  (bots: random, rush, economy)
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
        Some("play") => play(&args[1..], out),
        Some("tournament") => tournament(&args[1..], out),
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
    let mut turn_cap = Config::default().turn_cap;
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
    let n = log.actions.len();
    let target = step.unwrap_or(n);
    let mut text = String::new();
    replay_each(&log, &mut |i, s| {
        if all {
            let label = match i {
                0 => "start".to_string(),
                _ => action_label(&log.actions[i - 1]),
            };
            text.push_str(&format!("== step {i}: {label} ==\n"));
            text.push_str(&render(s, view));
        } else if i == target {
            text = render(s, view);
        }
    })
    .map_err(|e| Fail::Run(format!("{path}: {e:?}")))?;
    if !all && target > n {
        return usage(&format!("--step {target} is past the last step {n}"));
    }
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

/// A comma list of bot kinds.
fn bot_list(list: &str) -> Result<Vec<BotKind>, Fail> {
    list.split(',')
        .map(|name| match name {
            "random" => Ok(BotKind::Random),
            "rush" => Ok(BotKind::Rush),
            "economy" => Ok(BotKind::Economy),
            other => usage(&format!("unknown bot {other:?}")),
        })
        .collect()
}

/// A replay log of one bot game with `play_game`'s rng derivation and epsilon 0.
fn record_bot_game(
    seed: u64,
    kinds: [BotKind; 2],
    config: &Config,
    max_actions: usize,
) -> ReplayLog {
    let players = [make_bot(kinds[0], 0.0), make_bot(kinds[1], 0.0)];
    let mut rngs = [bot_rng(seed, 0), bot_rng(seed, 1)];
    record_game(
        seed,
        &MapParams::default(),
        config,
        max_actions,
        &mut |s: &State| {
            let cur = s.current as usize;
            players[cur].choose(&PlayerView::new(s, s.current), &mut rngs[cur])
        },
    )
}

/// `play`: one bot game recorded as a replay log (docs/MILESTONE_8 "CLI additions").
fn play(args: &[String], out: &mut dyn Write) -> Result<(), Fail> {
    let mut kinds: Option<Vec<BotKind>> = None;
    let mut seed: Option<u64> = None;
    let mut path: Option<String> = None;
    let mut turn_cap: u32 = 50;
    let mut max_actions: usize = 5000;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bots" => kinds = Some(bot_list(&value::<String>(&mut it, "--bots")?)?),
            "--seed" => seed = Some(value(&mut it, "--seed")?),
            "--out" => path = Some(value(&mut it, "--out")?),
            "--turn-cap" => turn_cap = value(&mut it, "--turn-cap")?,
            "--max-actions" => max_actions = value(&mut it, "--max-actions")?,
            other => return usage(&format!("unknown option {other:?}")),
        }
    }
    let Some(kinds) = kinds else {
        return usage("play needs --bots");
    };
    let [a, b] = kinds[..] else {
        return usage("--bots takes exactly two bots");
    };
    let Some(seed) = seed else {
        return usage("play needs --seed");
    };
    let Some(path) = path else {
        return usage("play needs --out");
    };
    if turn_cap == 0 {
        return usage("--turn-cap must be at least 1");
    }
    let config = Config {
        turn_cap,
        ..Config::default()
    };
    let log = record_bot_game(seed, [a, b], &config, max_actions);
    std::fs::write(&path, log.to_json()).map_err(|e| Fail::Run(format!("{path}: {e}")))?;
    writeln!(
        out,
        "wrote {path}: {} actions, outcome {}",
        log.actions.len(),
        outcome_text(log.outcome)
    )
    .map_err(io_fail)
}

/// `tournament`: every pair of the listed bots through `run_match` (docs/MILESTONE_8 "CLI
/// additions").
fn tournament(args: &[String], out: &mut dyn Write) -> Result<(), Fail> {
    let mut kinds: Option<Vec<BotKind>> = None;
    let mut games: u64 = 100;
    let mut turn_cap: u32 = 50;
    let mut seed: u64 = 0;
    let mut gate_min: Option<f64> = None;
    let mut save_dir: Option<String> = None;
    let mut save_count: Option<usize> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bots" => kinds = Some(bot_list(&value::<String>(&mut it, "--bots")?)?),
            "--games" => games = value(&mut it, "--games")?,
            "--turn-cap" => turn_cap = value(&mut it, "--turn-cap")?,
            "--seed" => seed = value(&mut it, "--seed")?,
            "--gate-min" => gate_min = Some(value(&mut it, "--gate-min")?),
            "--save-replays" => save_dir = Some(value(&mut it, "--save-replays")?),
            "--save-count" => save_count = Some(value(&mut it, "--save-count")?),
            other => return usage(&format!("unknown option {other:?}")),
        }
    }
    let Some(kinds) = kinds else {
        return usage("tournament needs --bots");
    };
    if kinds.len() < 2 {
        return usage("--bots needs at least two bots");
    }
    if turn_cap == 0 {
        return usage("--turn-cap must be at least 1");
    }
    let save = match (save_dir, save_count) {
        (Some(dir), Some(count)) => Some((dir, count)),
        (None, None) => None,
        _ => return usage("--save-replays and --save-count go together"),
    };
    let config = Config {
        turn_cap,
        ..Config::default()
    };
    let map = MapParams::default();
    let mut failures = Vec::new();
    writeln!(
        out,
        "tournament: {games} seeds per pair ({} games), turn cap {turn_cap}",
        2 * games
    )
    .map_err(io_fail)?;
    for (i, &x) in kinds.iter().enumerate() {
        for &y in &kinds[i + 1..] {
            let m = run_match(x, y, seed..seed + games, &map, &config);
            let (xn, yn) = (x.name(), y.name());
            writeln!(
                out,
                "{xn} vs {yn}: {xn} {} win, {yn} {} win, {} draw | by capture {}/{} | mean score diff {:+.1}",
                m.a_wins, m.b_wins, m.draws, m.a_capture_wins, m.b_capture_wins, m.mean_score_diff
            )
            .map_err(io_fail)?;
            for c in &m.checkpoints {
                writeln!(
                    out,
                    "  turn {}: {xn} cities {:.1} units {:.1} stars {:.1} score {:.1} | {yn} cities {:.1} units {:.1} stars {:.1} score {:.1}",
                    c.turn,
                    c.a.cities, c.a.units, c.a.stars, c.a.score,
                    c.b.cities, c.b.units, c.b.stars, c.b.score
                )
                .map_err(io_fail)?;
            }
            if let Some(min) = gate_min {
                for (bot, rate) in [(x, m.win_rate_a()), (y, m.win_rate_b())] {
                    let other = if bot == x { y } else { x };
                    let gated =
                        matches!(bot, BotKind::Rush | BotKind::Economy) && other == BotKind::Random;
                    if gated && rate < min {
                        failures.push(format!(
                            "{} won {rate:.3} against {} (min {min})",
                            bot.name(),
                            other.name()
                        ));
                    }
                }
            }
            if let Some((dir, count)) = &save {
                // run_match's game order: each seed with x as player 0, then swapped.
                let order = (seed..seed + games).flat_map(|s| [(s, [x, y]), (s, [y, x])]);
                for (k, (s, bots)) in order.take(*count).enumerate() {
                    let log = record_bot_game(s, bots, &config, usize::MAX);
                    let path = format!("{dir}/{xn}-vs-{yn}-{k}.json");
                    std::fs::write(&path, log.to_json())
                        .map_err(|e| Fail::Run(format!("{path}: {e}")))?;
                }
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Fail::Run(format!("GATE FAILED: {}", failures.join("; "))))
    }
}
