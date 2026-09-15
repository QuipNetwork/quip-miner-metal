use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use quip_solver_core::{IsingGraph, SampleParams};

use crate::graph::{prepare, PreparedGraph, LANES};
use crate::msa::{initial_spins, schedule, validate_params, ThresholdRows};
use crate::native::{AneProgram, NeighborUpdate};
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

    let mut programs: Vec<AneProgram> = Vec::with_capacity(prepared.tiles.len());
    for tile in &prepared.tiles {
        let weights = prepared.tile_weights(tile)?;
        let mut fields = vec![0; tile.output_channels];
        for (row, &node) in tile.nodes.iter().enumerate() {
            fields[row] = prepared.fields[node];
        }
        let mut program = AneProgram::compile(
            prepared.input_channels,
            tile.output_channels,
            &weights,
            &fields,
        )?;
        if let Some(first) = programs.first() {
            program.share_input(first)?;
        }
        programs.push(program);
    }
    stats.programs = u32::try_from(programs.len())
        .map_err(|_| AneError::Runtime("ANE program count exceeds u32".into()))?;
    let mut buffers = TileBuffers::new(&prepared);
    stats.setup_us = elapsed_us(setup_started.elapsed())?;
    let anneal_started = Instant::now();
    let mut rows = ThresholdRows::new(params.seed);
    for (rung_index, rung) in rungs.iter().enumerate() {
        rows.begin_rung(rung.beta);
        for sweep in 0..rung.sweeps {
            let thresholds = rows.expand(prepared.node_count, rung_index, sweep);
            for color in 0..prepared.color_count {
                advance_color(
                    &prepared,
                    &mut programs,
                    color,
                    &mut state,
                    &thresholds,
                    &mut stats,
                    &mut buffers,
                )?;
            }
        }
    }
    stats.anneal_us = elapsed_us(anneal_started.elapsed())?;

    let spins = read_major_spins(&state, prepared.node_count, params.num_reads);
    let mut close_error = None;
    for program in programs {
        if let Err(error) = program.close() {
            if close_error.is_none() {
                close_error = Some(error);
            }
        }
    }
    if let Some(error) = close_error {
        return Err(error);
    }
    Ok(RunOutput { spins, stats })
}

fn read_major_spins(state: &[i8], nodes: usize, reads: usize) -> Vec<Vec<i8>> {
    (0..reads)
        .map(|read| (0..nodes).map(|node| state[node * LANES + read]).collect())
        .collect()
}

struct TileBuffers {
    own: Vec<i8>,
    selected: Vec<u8>,
    output: Vec<i8>,
    previous_tile: Option<usize>,
}

impl TileBuffers {
    fn new(prepared: &PreparedGraph) -> Self {
        let count = prepared
            .tiles
            .iter()
            .map(|tile| tile.output_channels * LANES)
            .max()
            .unwrap_or(0);
        Self {
            own: vec![1; count],
            selected: vec![0; count],
            output: vec![0; count],
            previous_tile: None,
        }
    }
}

fn advance_color(
    prepared: &PreparedGraph,
    programs: &mut [AneProgram],
    color: usize,
    state: &mut [i8],
    thresholds: &[u8],
    stats: &mut RunStats,
    buffers: &mut TileBuffers,
) -> Result<(), AneError> {
    if programs.len() != prepared.tiles.len() {
        return Err(AneError::Runtime(
            "ANE program count does not match tiles".into(),
        ));
    }
    if state.len() != prepared.input_channels * LANES
        || thresholds.len() != prepared.node_count * LANES
    {
        return Err(AneError::Runtime(
            "solver state or threshold length mismatch".into(),
        ));
    }

    for (tile_index, (tile, program)) in prepared
        .tiles
        .iter()
        .zip(programs)
        .enumerate()
        .filter(|(_, (tile, _))| tile.color == color)
    {
        let count = tile.output_channels * LANES;
        let own = &mut buffers.own[..count];
        let selected = &mut buffers.selected[..count];
        let output = &mut buffers.output[..count];
        own.fill(1);
        selected.fill(0);
        for (row, &node) in tile.nodes.iter().enumerate() {
            own[row * LANES..(row + 1) * LANES]
                .copy_from_slice(&state[node * LANES..(node + 1) * LANES]);
            selected[row * LANES..(row + 1) * LANES]
                .copy_from_slice(&thresholds[node * LANES..(node + 1) * LANES]);
        }
        let update = match buffers.previous_tile {
            None => NeighborUpdate::Full,
            Some(previous) => NeighborUpdate::Rows(&prepared.tiles[previous].nodes),
        };
        let times = program.evaluate_into(state, update, own, selected, output)?;
        stats.dispatches += 1;
        stats.staging_us += times.staging_us;
        stats.dispatch_us += times.dispatch_us;
        for (row, &node) in tile.nodes.iter().enumerate() {
            state[node * LANES..(node + 1) * LANES]
                .copy_from_slice(&output[row * LANES..(row + 1) * LANES]);
        }
        buffers.previous_tile = Some(tile_index);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::{advance_color, solve_in_process, RunOutput, RunStats, TileBuffers};
    use crate::graph::{prepare, LANES};
    use crate::msa::{initial_spins, schedule, ThresholdRows};
    use crate::native::AneProgram;
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

    fn compile_programs(prepared: &crate::graph::PreparedGraph) -> Vec<AneProgram> {
        let mut programs: Vec<AneProgram> = prepared
            .tiles
            .iter()
            .map(|tile| {
                let weights = prepared.tile_weights(tile).unwrap();
                let mut fields = vec![0; tile.output_channels];
                for (row, &node) in tile.nodes.iter().enumerate() {
                    fields[row] = prepared.fields[node];
                }
                AneProgram::compile(
                    prepared.input_channels,
                    tile.output_channels,
                    &weights,
                    &fields,
                )
                .unwrap()
            })
            .collect();
        if let Some((first, rest)) = programs.split_first_mut() {
            for program in rest {
                program.share_input(first).unwrap();
            }
        }
        programs
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

    fn run_color_oracle_case(graph: &IsingGraph, params: &SampleParams) -> (RunStats, usize) {
        let setup_started = Instant::now();
        let prepared = prepare(graph).unwrap();
        let mut programs = compile_programs(&prepared);
        let mut buffers = TileBuffers::new(&prepared);
        let mut state = initial_spins(prepared.node_count, params.seed);
        state.resize(prepared.input_channels * LANES, 0);
        let mut stats = RunStats {
            programs: u32::try_from(programs.len()).unwrap(),
            ..RunStats::default()
        };
        stats.setup_us = u64::try_from(setup_started.elapsed().as_micros()).unwrap();
        let mut mismatches = 0;
        let anneal_started = Instant::now();
        let mut rows = ThresholdRows::new(params.seed);
        for (rung_index, rung) in schedule(graph, params).unwrap().iter().enumerate() {
            rows.begin_rung(rung.beta);
            for sweep in 0..rung.sweeps {
                let thresholds = rows.expand(prepared.node_count, rung_index, sweep);
                for color in 0..prepared.color_count {
                    let expected = oracle_color(&prepared, color, &state, &thresholds);
                    advance_color(
                        &prepared,
                        &mut programs,
                        color,
                        &mut state,
                        &thresholds,
                        &mut stats,
                        &mut buffers,
                    )
                    .unwrap();
                    mismatches += state
                        .iter()
                        .zip(expected)
                        .filter(|(actual, expected)| **actual != *expected)
                        .count();
                }
            }
        }
        stats.anneal_us = u64::try_from(anneal_started.elapsed().as_micros()).unwrap();
        for program in programs {
            program.close().unwrap();
        }
        (stats, mismatches)
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
        let (validation_stats, color_mismatches) = run_color_oracle_case(&graph, &params);
        assert_eq!(final_mismatches, 0);
        assert_eq!(color_mismatches, 0);
        eprintln!(
            "nodes={} reads=128 sweeps=4 colors={} tiles={} shapes={shapes:?} production_dispatches={} final_mismatches={final_mismatches} color_mismatches={color_mismatches} production_setup_us={} production_staging_us={} production_dispatch_us={} production_anneal_us={} production_wall_us={production_wall_us} validation_setup_us={} validation_anneal_us={}",
            prepared.node_count,
            prepared.color_count,
            prepared.tiles.len(),
            output.stats.dispatches,
            output.stats.setup_us,
            output.stats.staging_us,
            output.stats.dispatch_us,
            output.stats.anneal_us,
            validation_stats.setup_us,
            validation_stats.anneal_us,
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
    fn hardware_colors_observe_prior_updates() {
        let graph = IsingGraph::new(vec![0.0; 2], vec![1.0], vec![(0, 1)]);
        let prepared = prepare(&graph).unwrap();
        let mut programs = compile_programs(&prepared);
        let mut buffers = TileBuffers::new(&prepared);
        let mut state = vec![0; prepared.input_channels * LANES];
        state[..2 * LANES].fill(1);
        let thresholds = vec![0; 2 * LANES];
        let mut stats = RunStats::default();
        advance_color(
            &prepared,
            &mut programs,
            0,
            &mut state,
            &thresholds,
            &mut stats,
            &mut buffers,
        )
        .unwrap();
        advance_color(
            &prepared,
            &mut programs,
            1,
            &mut state,
            &thresholds,
            &mut stats,
            &mut buffers,
        )
        .unwrap();
        assert!(state[..LANES].iter().all(|&spin| spin == -1));
        assert!(state[LANES..2 * LANES].iter().all(|&spin| spin == 1));
        assert_eq!(stats.dispatches, 2);
        for program in programs {
            program.close().unwrap();
        }
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
            let (stats, mismatches) = run_color_oracle_case(graph, &params(128, 16));
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
            assert_eq!(output.stats.dispatches, 32);
            eprintln!(
                "reads={reads} dispatches={} mismatches={mismatches}",
                output.stats.dispatches
            );
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
