use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use quip_solver_core::{IsingGraph, SampleParams};

use crate::graph::{prepare, LANES};
use crate::msa::{initial_spins, schedule, validate_params, ThresholdRows};
use crate::native::{AneProgram, BLOCK_SWEEPS};
use crate::AneError;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunStats {
    pub(crate) programs: u32,
    pub(crate) dispatches: u64,
    pub(crate) setup_us: u64,
    pub(crate) staging_us: u64,
    pub(crate) dispatch_us: u64,
    pub(crate) anneal_us: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunOutput {
    pub(crate) spins: Vec<Vec<i8>>,
    pub(crate) stats: RunStats,
}

fn elapsed_us(duration: Duration) -> Result<u64, AneError> {
    u64::try_from(duration.as_micros())
        .map_err(|_| AneError::Runtime("solver timing exceeds u64 microseconds".into()))
}

pub(crate) fn solve_in_process(
    graph: &IsingGraph,
    params: &SampleParams,
) -> Result<RunOutput, AneError> {
    solve_with_block(graph, params, BLOCK_SWEEPS)
}

fn solve_with_block(
    graph: &IsingGraph,
    params: &SampleParams,
    block_sweeps: usize,
) -> Result<RunOutput, AneError> {
    let setup_started = Instant::now();
    validate_params(params)?;
    let prepared = prepare(graph)?;
    let rungs = schedule(graph, params)?;
    let mut state = initial_spins(prepared.node_count, params.seed);
    state.resize(prepared.input_channels * LANES, 0);
    let mut stats = RunStats::default();

    if prepared.node_count == 0 || rungs.is_empty() {
        stats.setup_us = elapsed_us(setup_started.elapsed())?;
        return Ok(RunOutput {
            spins: read_major_spins(&state, prepared.node_count, params.num_reads),
            stats,
        });
    }

    let mut program = AneProgram::compile(&prepared, block_sweeps)?;
    stats.programs = 1;
    let order = prepared.storage_order();
    let mut packed = vec![1; prepared.input_channels * LANES];
    for (row, &node) in order.iter().enumerate() {
        packed[row * LANES..(row + 1) * LANES]
            .copy_from_slice(&state[node * LANES..(node + 1) * LANES]);
    }
    program.reset(&packed)?;
    stats.setup_us = elapsed_us(setup_started.elapsed())?;
    let anneal_started = Instant::now();
    let mut rows = ThresholdRows::new(params.seed);
    let mut block = vec![255; prepared.input_channels * LANES * block_sweeps];
    let mut slot = 0;
    for (rung_index, rung) in rungs.iter().enumerate() {
        rows.begin_rung(rung.beta);
        for sweep in 0..rung.sweeps {
            let thresholds = rows.expand(prepared.node_count, rung_index, sweep);
            for (row, &node) in order.iter().enumerate() {
                let start = (slot * prepared.input_channels + row) * LANES;
                block[start..start + LANES]
                    .copy_from_slice(&thresholds[node * LANES..(node + 1) * LANES]);
            }
            slot += 1;
            if slot == block_sweeps {
                let times = program.advance(&block)?;
                stats.dispatches += 1;
                stats.staging_us += times.staging_us;
                stats.dispatch_us += times.dispatch_us;
                block.fill(255);
                slot = 0;
            }
        }
    }
    if slot != 0 {
        let times = program.advance(&block)?;
        stats.dispatches += 1;
        stats.staging_us += times.staging_us;
        stats.dispatch_us += times.dispatch_us;
    }
    program.read(&mut packed)?;
    for (row, &node) in order.iter().enumerate() {
        state[node * LANES..(node + 1) * LANES]
            .copy_from_slice(&packed[row * LANES..(row + 1) * LANES]);
    }
    stats.anneal_us = elapsed_us(anneal_started.elapsed())?;
    let spins = read_major_spins(&state, prepared.node_count, params.num_reads);
    program.close()?;
    Ok(RunOutput { spins, stats })
}

fn read_major_spins(state: &[i8], nodes: usize, reads: usize) -> Vec<Vec<i8>> {
    (0..reads)
        .map(|read| (0..nodes).map(|node| state[node * LANES + read]).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::{solve_in_process, solve_with_block, RunOutput, RunStats};
    use crate::graph::{prepare, LANES};
    use crate::msa::{initial_spins, schedule, ThresholdRows};
    use crate::native::{AneProgram, BLOCK_SWEEPS};
    use quip_solver_core::{IsingGraph, SampleParams};

    fn params(num_reads: usize, num_sweeps: usize) -> SampleParams {
        SampleParams {
            num_reads,
            num_sweeps,
            sweeps_per_beta: 4,
            beta_range: Some((0.2, 2.0)),
            seed: 0x1234_5678,
        }
    }

    fn oracle_color(
        prepared: &crate::graph::PreparedGraph,
        color: usize,
        before: &[i8],
        thresholds: &[u8],
    ) -> Vec<i8> {
        let mut expected = before.to_vec();
        for tile in prepared.tiles.iter().filter(|tile| tile.color == color) {
            for &node in &tile.nodes {
                for read in 0..LANES {
                    let spin = before[node * LANES + read];
                    let field_term = prepared.fields[node];
                    let mut degree = usize::from(field_term != 0);
                    let mut satisfied = usize::from(i16::from(field_term) * i16::from(spin) < 0);
                    for &(neighbor, coupling) in &prepared.neighbors[node] {
                        degree += 1;
                        let product = i16::from(coupling)
                            * i16::from(spin)
                            * i16::from(before[neighbor * LANES + read]);
                        satisfied += usize::from(product < 0);
                    }
                    let threshold = usize::from(thresholds[node * LANES + read]);
                    expected[node * LANES + read] = if satisfied <= (degree + threshold) / 2 {
                        -spin
                    } else {
                        spin
                    };
                }
            }
        }
        expected
    }

    fn oracle_solve(graph: &IsingGraph, params: &SampleParams) -> Vec<Vec<i8>> {
        let prepared = prepare(graph).unwrap();
        let rungs = schedule(graph, params).unwrap();
        let mut state = initial_spins(prepared.node_count, params.seed);
        state.resize(prepared.input_channels * LANES, 0);
        let mut rows = ThresholdRows::new(params.seed);
        for (rung_index, rung) in rungs.iter().enumerate() {
            rows.begin_rung(rung.beta);
            for sweep in 0..rung.sweeps {
                let thresholds = rows.expand(prepared.node_count, rung_index, sweep);
                for color in 0..prepared.color_count {
                    state = oracle_color(&prepared, color, &state, &thresholds);
                }
            }
        }
        (0..params.num_reads)
            .map(|read| {
                (0..prepared.node_count)
                    .map(|node| state[node * LANES + read])
                    .collect()
            })
            .collect()
    }

    fn complete_graph(n: usize) -> IsingGraph {
        let edges: Vec<_> = (0..n)
            .flat_map(|u| ((u + 1)..n).map(move |v| (u, v)))
            .collect();
        let couplings = edges
            .iter()
            .map(|&(u, v)| if (u + v) % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let fields = (0..n).map(|node| (node % 3) as f64 - 1.0).collect();
        IsingGraph::new(fields, couplings, edges)
    }

    fn degree_twenty_fixture(n: usize) -> IsingGraph {
        assert!(n >= 128 && n.is_multiple_of(4));
        let offsets = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13];
        let mut edges = Vec::with_capacity(n * 10);
        let mut couplings = Vec::with_capacity(n * 10);
        for node in 0..n {
            for offset in offsets {
                let next = (node + offset) % n;
                let (lo, hi) = (node.min(next), node.max(next));
                edges.push((node, next));
                couplings.push(if (lo * 17 + hi * 13) % 7 < 3 {
                    -1.0
                } else {
                    1.0
                });
            }
        }
        let fields = (0..n).map(|node| (node % 3) as f64 - 1.0).collect();
        IsingGraph::new(fields, couplings, edges)
    }

    fn run_oracle_case(graph: &IsingGraph, parameters: &SampleParams) -> (RunStats, usize) {
        let output = solve_in_process(graph, parameters).unwrap();
        let expected = oracle_solve(graph, parameters);
        let mismatches = output
            .spins
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
            .filter(|(a, b)| a != b)
            .count();
        (output.stats, mismatches)
    }

    fn run_capacity(graph: IsingGraph) {
        let params = params(128, 4);
        let prepared = prepare(&graph).unwrap();
        let shapes: Vec<_> = prepared
            .tiles
            .iter()
            .map(|tile| (prepared.input_channels, tile.output_channels))
            .collect();
        let started = Instant::now();
        let output = solve_in_process(&graph, &params).unwrap();
        let production_wall_us = u64::try_from(started.elapsed().as_micros()).unwrap();
        let expected = oracle_solve(&graph, &params);
        let final_mismatches = output
            .spins
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
            .filter(|(actual, expected)| actual != expected)
            .count();
        assert_eq!(final_mismatches, 0);
        eprintln!(
            "nodes={} reads=128 sweeps=4 colors={} tiles={} shapes={shapes:?} production_dispatches={} final_mismatches={final_mismatches} production_setup_us={} production_staging_us={} production_dispatch_us={} production_anneal_us={} production_wall_us={production_wall_us}",
            prepared.node_count,
            prepared.color_count,
            prepared.tiles.len(),
            output.stats.dispatches,
            output.stats.setup_us,
            output.stats.staging_us,
            output.stats.dispatch_us,
            output.stats.anneal_us,
        );
    }

    #[test]
    fn zero_sweeps_and_empty_graph_do_not_compile_or_dispatch() {
        let zero = params(31, 0);
        let graph = IsingGraph::new(vec![-1.0, 1.0], vec![1.0], vec![(0, 1)]);
        let output = solve_in_process(&graph, &zero).unwrap();
        assert_eq!(output.stats.programs, 0);
        assert_eq!(output.stats.dispatches, 0);
        assert_eq!(output.spins, oracle_solve(&graph, &zero));

        let empty = IsingGraph::new(Vec::new(), Vec::new(), Vec::new());
        let output = solve_in_process(&empty, &params(33, 16)).unwrap();
        assert_eq!(output.stats.programs, 0);
        assert_eq!(output.stats.dispatches, 0);
        assert_eq!(output.spins, vec![Vec::<i8>::new(); 33]);
    }

    #[test]
    fn run_payloads_reject_unknown_fields() {
        let stats = RunStats::default();
        let encoded = serde_json::to_string(&stats).unwrap();
        assert!(serde_json::from_str::<RunStats>(&encoded).is_ok());
        assert!(serde_json::from_str::<RunStats>(
            r#"{"programs":0,"dispatches":0,"setup_us":0,"staging_us":0,"dispatch_us":0,"anneal_us":0,"extra":0}"#
        )
        .is_err());

        let output = RunOutput {
            spins: vec![vec![1]],
            stats,
        };
        let encoded = serde_json::to_string(&output).unwrap();
        assert!(serde_json::from_str::<RunOutput>(&encoded).is_ok());
        assert!(serde_json::from_str::<RunOutput>(
            r#"{"spins":[],"stats":{"programs":0,"dispatches":0,"setup_us":0,"staging_us":0,"dispatch_us":0,"anneal_us":0},"extra":0}"#
        )
        .is_err());
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_two_sweeps_one_dispatch_exact_oracle() {
        let graph = IsingGraph::new(
            vec![-1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0],
            vec![1.0, -1.0, 1.0],
            vec![(0, 1), (1, 2), (4, 5)],
        );
        let parameters = params(128, 2);
        let actual = solve_in_process(&graph, &parameters).unwrap();
        assert_eq!(actual.spins, oracle_solve(&graph, &parameters));
        assert_eq!(
            actual.stats.dispatches, 1,
            "two ordered sweeps must complete in one ANE dispatch"
        );
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_colors_observe_prior_updates() {
        let graph = IsingGraph::new(vec![0.0; 2], vec![1.0], vec![(0, 1)]);
        let prepared = prepare(&graph).unwrap();
        let mut program = AneProgram::compile(&prepared, BLOCK_SWEEPS).unwrap();
        let mut state = vec![1; prepared.input_channels * LANES];
        program.reset(&state).unwrap();
        let thresholds = vec![0; prepared.input_channels * LANES * BLOCK_SWEEPS];
        program.advance(&thresholds).unwrap();
        program.read(&mut state).unwrap();
        assert!(state[..LANES].iter().all(|&spin| spin == -1));
        assert!(state[LANES..2 * LANES].iter().all(|&spin| spin == 1));
        program.close().unwrap();
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_irregular_colors() {
        let graphs = vec![
            complete_graph(1),
            complete_graph(3),
            complete_graph(21),
            IsingGraph::new(
                vec![-1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0],
                vec![1.0, -1.0, 1.0],
                vec![(0, 1), (1, 2), (4, 5)],
            ),
        ];
        for graph in &graphs {
            let (stats, mismatches) = run_oracle_case(graph, &params(128, 16));
            assert_eq!(mismatches, 0);
            eprintln!(
                "nodes={} dispatches={} mismatches={mismatches}",
                graph.h.len(),
                stats.dispatches
            );
        }
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_read_counts() {
        let graph = IsingGraph::new(vec![-1.0, 1.0], vec![1.0], vec![(0, 1)]);
        for reads in [1, 31, 32, 33, 127, 128] {
            let params = params(reads, 16);
            let output = solve_in_process(&graph, &params).unwrap();
            let expected = oracle_solve(&graph, &params);
            let mismatches = output
                .spins
                .iter()
                .flatten()
                .zip(expected.iter().flatten())
                .filter(|(actual, expected)| actual != expected)
                .count();
            assert_eq!(mismatches, 0);
            assert_eq!(
                output.stats.dispatches,
                16usize.div_ceil(BLOCK_SWEEPS) as u64
            );
            eprintln!(
                "reads={reads} dispatches={} mismatches={mismatches}",
                output.stats.dispatches
            );
        }
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_tails_and_beta_boundaries() {
        let graph = complete_graph(7);
        for sweeps in [0, 1, 3, 5, 17] {
            for per_beta in [1, 3, 4] {
                let mut parameters = params(33, sweeps);
                parameters.sweeps_per_beta = per_beta;
                parameters.beta_range = Some((0.07, 1.73));
                let actual = solve_in_process(&graph, &parameters).unwrap();
                assert_eq!(
                    actual.spins,
                    oracle_solve(&graph, &parameters),
                    "sweeps={sweeps} per_beta={per_beta}"
                );
                assert_eq!(
                    actual.stats.dispatches,
                    sweeps.div_ceil(BLOCK_SWEEPS) as u64
                );
            }
        }
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_full_output_4577() {
        let n = 4577;
        let edges: Vec<_> = (0..n - 1).map(|node| (node, node + 1)).collect();
        run_capacity(IsingGraph::new(vec![0.0; n], vec![1.0; edges.len()], edges));
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_color_block_shapes() {
        for nodes in [15, 16, 21] {
            for block in [1, 2, 4, 8] {
                eprintln!("probe nodes={nodes} block={block}");
                let graph = complete_graph(nodes);
                let parameters = params(128, block * 2 + 1);
                let actual = solve_with_block(&graph, &parameters, block).unwrap();
                assert_eq!(actual.spins, oracle_solve(&graph, &parameters));
                assert_eq!(actual.stats.dispatches, 3);
            }
        }
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE; compares fused block throughput"]
    fn hardware_block_throughput() {
        let graph = degree_twenty_fixture(6016);
        let parameters = params(128, 128);
        let expected = oracle_solve(&graph, &parameters);
        for blocks in [[2, 4, 8], [8, 4, 2], [4, 2, 8]] {
            for block in blocks {
                let started = Instant::now();
                let actual = solve_with_block(&graph, &parameters, block).unwrap();
                assert_eq!(actual.spins, expected);
                eprintln!("nodes=6016 sweeps=128 block={block} dispatches={} setup_us={} staging_us={} dispatch_us={} anneal_us={} wall_us={}", actual.stats.dispatches, actual.stats.setup_us, actual.stats.staging_us, actual.stats.dispatch_us, actual.stats.anneal_us, started.elapsed().as_micros());
            }
        }
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_capacity_6016() {
        run_capacity(degree_twenty_fixture(6_016));
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_capacity_8192() {
        run_capacity(degree_twenty_fixture(8_192));
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_capacity_16384() {
        run_capacity(degree_twenty_fixture(16_384));
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_isolated_16384() {
        run_capacity(IsingGraph::new(vec![0.0; 16_384], Vec::new(), Vec::new()));
    }
}
