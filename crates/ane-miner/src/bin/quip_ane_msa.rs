use clap::Parser;
use quip_miner_ane::{worker_main, AneSampler, ANE_MSA_IDENTITY};
use quip_solver_core::{run, CommonArgs, OpenError};

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, hide = true,
        conflicts_with_all = ["quip_coordinator", "capabilities", "check", "solve"])]
    ane_worker: Option<u32>,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    if let Some(parent_pid) = cli.ane_worker {
        return worker_main(parent_pid);
    }
    run(ANE_MSA_IDENTITY, &cli.common, || {
        let executable = std::env::current_exe()
            .map_err(|error| OpenError(format!("locate ANE worker executable: {error}")))?;
        AneSampler::open(executable)
    })
}
