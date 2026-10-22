fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = sim_tools::cli::run_cli(&args, &mut std::io::stdout(), &mut std::io::stderr());
    std::process::exit(code);
}
