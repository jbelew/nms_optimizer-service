use pyo3::prelude::*;
use pyo3_log::init;
use rand::prelude::*;
use serde::{self, Deserialize, Deserializer, Serialize};

fn module_type_from_string_or_null<'de, D>(deserializer: D) -> Result<Option<ModuleType>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: Option<String> = Option::deserialize(deserializer)?;
    match s.as_deref() {
        Some("core") => Ok(Some(ModuleType::Core)),
        Some("bonus") => Ok(Some(ModuleType::Bonus)),
        Some("upgrade") => Ok(Some(ModuleType::Upgrade)),
        Some("cosmetic") => Ok(Some(ModuleType::Cosmetic)),
        Some("reactor") => Ok(Some(ModuleType::Reactor)),
        Some("atlantid") => Ok(Some(ModuleType::Atlantid)),
        Some("") | None => Ok(None),
        _ => Err(serde::de::Error::custom(format!("Expected valid ModuleType string or null, got {:?}", s))),
    }
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AdjacencyType {
    #[serde(rename = "greater")]
    Greater,
    #[serde(rename = "lesser")]
    Lesser,
    #[serde(rename = "no_adjacency")]
    NoAdjacency,
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ModuleType {
    #[serde(rename = "core")]
    Core,
    #[serde(rename = "bonus")]
    Bonus,
    #[serde(rename = "upgrade")]
    Upgrade,
    #[serde(rename = "cosmetic")]
    Cosmetic,
    #[serde(rename = "reactor")]
    Reactor,
    #[serde(rename = "atlantid")]
    Atlantid,
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Cell {
    #[pyo3(get, set)]
    pub module: Option<String>,
    #[pyo3(get, set)]
    pub label: Option<String>,
    #[pyo3(get, set)]
    pub value: i32,
    #[pyo3(get, set)]
    #[serde(rename = "type", deserialize_with = "module_type_from_string_or_null")]
    pub module_type: Option<ModuleType>,
    #[pyo3(get, set)]
    pub total: f64,
    #[pyo3(get, set)]
    pub adjacency_bonus: f64,
    #[pyo3(get, set)]
    pub bonus: f64,
    #[pyo3(get, set)]
    pub active: bool,
    #[pyo3(get, set)]
    pub adjacency: Option<AdjacencyType>,
    #[pyo3(get, set)]
    pub tech: Option<String>,
    #[pyo3(get, set)]
    pub supercharged: bool,
    #[pyo3(get, set)]
    pub sc_eligible: bool,
    #[pyo3(get, set)]
    pub image: Option<String>,
}

#[pymethods]
impl Cell {
    #[new]
    #[pyo3(signature = (value, total, adjacency_bonus, bonus, active, supercharged, sc_eligible, module = None, label = None, module_type = None, adjacency = None, tech = None, image = None))]
    fn new(
        value: i32,
        total: f64,
        adjacency_bonus: f64,
        bonus: f64,
        active: bool,
        supercharged: bool,
        sc_eligible: bool,
        module: Option<String>,
        label: Option<String>,
        module_type: Option<ModuleType>,
        adjacency: Option<AdjacencyType>,
        tech: Option<String>,
        image: Option<String>,
    ) -> Self {
        Cell {
            module,
            label,
            value,
            module_type,
            total,
            adjacency_bonus,
            bonus,
            active,
            adjacency,
            tech,
            supercharged,
            sc_eligible,
            image,
        }
    }
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Grid {
    #[pyo3(get, set)]
    pub width: i32,
    #[pyo3(get, set)]
    pub height: i32,
    #[pyo3(get, set)]
    pub cells: Vec<Vec<Cell>>,
}

#[pymethods]
impl Grid {
    #[new]
    fn new(width: i32, height: i32, cells: Vec<Vec<Cell>>) -> Self {
        Grid {
            width,
            height,
            cells,
        }
    }
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Module {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub label: String,
    #[pyo3(get, set)]
    pub tech: String,
    #[pyo3(get, set)]
    #[serde(rename = "type")]
    pub module_type: ModuleType,
    #[pyo3(get, set)]
    pub bonus: f64,
    #[pyo3(get, set)]
    pub adjacency: AdjacencyType,
    #[pyo3(get, set)]
    pub sc_eligible: bool,
    #[pyo3(get, set)]
    pub image: Option<String>,
}

#[pymethods]
impl Module {
    #[new]
    fn new(
        id: String,
        label: String,
        tech: String,
        module_type: ModuleType,
        bonus: f64,
        adjacency: AdjacencyType,
        sc_eligible: bool,
        image: Option<String>,
    ) -> Self {
        Module {
            id,
            label,
            tech,
            module_type,
            bonus,
            adjacency,
            sc_eligible,
            image,
        }
    }
}

// --- Constants ---
const WEIGHT_FROM_GREATER_BONUS: f64 = 0.06;
const WEIGHT_FROM_LESSER_BONUS: f64 = 0.03;
const WEIGHT_FROM_GREATER_CORE: f64 = 0.07;
const WEIGHT_FROM_LESSER_CORE: f64 = 0.04;
const SUPERCHARGE_MULTIPLIER: f64 = 1.25;

impl Grid {
    fn get_cell(&self, x: i32, y: i32) -> Option<&Cell> {
        if x >= 0 && x < self.width && y >= 0 && y < self.height {
            Some(&self.cells[y as usize][x as usize])
        } else {
            None
        }
    }

    fn place_module(&mut self, x: i32, y: i32, module: &Module) {
        if let Some(cell) = self.cells.get_mut(y as usize).and_then(|row| row.get_mut(x as usize)) {
            cell.module = Some(module.id.clone());
            cell.label = Some(module.label.clone());
            cell.tech = Some(module.tech.clone());
            cell.module_type = Some(module.module_type.clone());
            cell.bonus = module.bonus;
            cell.adjacency = Some(module.adjacency.clone());
            cell.sc_eligible = module.sc_eligible;
            cell.image = module.image.clone();
        }
    }

    fn clear_cell(&mut self, x: i32, y: i32) {
        if let Some(cell) = self.cells.get_mut(y as usize).and_then(|row| row.get_mut(x as usize)) {
            cell.module = None;
            cell.label = None;
            cell.tech = None;
            cell.module_type = None;
            cell.bonus = 0.0;
            cell.adjacency = None;
            cell.sc_eligible = false;
            cell.image = None;
            cell.total = 0.0;
            cell.adjacency_bonus = 0.0;
        }
    }

    fn clear_all_modules_of_tech(&mut self, tech_to_clear: &str) {
        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(cell) = self.cells.get_mut(y as usize).and_then(|row| row.get_mut(x as usize)) {
                    if let Some(cell_tech) = &cell.tech {
                        if cell_tech == tech_to_clear {
                            self.clear_cell(x, y);
                        }
                    }
                }
            }
        }
    }

    fn place_modules_with_supercharged_priority(&mut self, tech_modules: &Vec<Module>, _tech: &str) {
        let mut supercharged_slots: Vec<(i32, i32)> = Vec::new();
        let mut active_slots: Vec<(i32, i32)> = Vec::new();

        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(cell) = self.get_cell(x, y) {
                    if cell.module.is_none() && cell.active {
                        if cell.supercharged {
                            supercharged_slots.push((x, y));
                        } else {
                            active_slots.push((x, y));
                        }
                    }
                }
            }
        }

        let mut core_module: Option<&Module> = None;
        let mut bonus_modules: Vec<&Module> = Vec::new();
        for module in tech_modules {
            if module.module_type == ModuleType::Core {
                core_module = Some(module);
            } else {
                bonus_modules.push(module);
            }
        }

        bonus_modules.sort_by(|a, b| b.bonus.partial_cmp(&a.bonus).unwrap());

        let num_available_positions = supercharged_slots.len() + active_slots.len();
        let mut modules_to_place_candidates: Vec<&Module> = Vec::new();
        if let Some(core) = core_module {
            modules_to_place_candidates.push(core);
        }
        modules_to_place_candidates.extend(bonus_modules);

        let modules_to_place = &modules_to_place_candidates[..num_available_positions.min(modules_to_place_candidates.len())];

        let mut sc_eligible_modules: Vec<&Module> = modules_to_place.iter().filter(|m| m.sc_eligible).cloned().collect();
        let mut non_sc_eligible_modules: Vec<&Module> = modules_to_place.iter().filter(|m| !m.sc_eligible).cloned().collect();

        sc_eligible_modules.sort_by(|a, b| (a.module_type != ModuleType::Core).cmp(&(b.module_type != ModuleType::Core)).then(b.bonus.partial_cmp(&a.bonus).unwrap()));
        non_sc_eligible_modules.sort_by(|a, b| (a.module_type != ModuleType::Core).cmp(&(b.module_type != ModuleType::Core)).then(b.bonus.partial_cmp(&a.bonus).unwrap()));

        let mut placed_module_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut remaining_sc_eligible: Vec<&Module> = Vec::new();

        let mut rng = rand::thread_rng();
        supercharged_slots.shuffle(&mut rng);

        for module in sc_eligible_modules {
            if supercharged_slots.is_empty() {
                remaining_sc_eligible.push(module);
                continue;
            }

            let mut placed_in_sc = false;
            let mut slot_index_to_remove = -1;

            for (i, (x, y)) in supercharged_slots.iter().enumerate() {
                if self.get_cell(*x, *y).unwrap().module.is_none() {
                    self.place_module(*x, *y, module);
                    placed_module_ids.insert(module.id.clone());
                    slot_index_to_remove = i as i32;
                    placed_in_sc = true;
                    break;
                }
            }

            if placed_in_sc {
                if slot_index_to_remove != -1 {
                    supercharged_slots.remove(slot_index_to_remove as usize);
                }
            } else {
                remaining_sc_eligible.push(module);
            }
        }

        let mut modules_for_active_slots = non_sc_eligible_modules;
        modules_for_active_slots.extend(remaining_sc_eligible);
        modules_for_active_slots.sort_by(|a, b| (a.module_type != ModuleType::Core).cmp(&(b.module_type != ModuleType::Core)).then(b.bonus.partial_cmp(&a.bonus).unwrap()));

        active_slots.shuffle(&mut rng);

        for module in modules_for_active_slots {
            if placed_module_ids.contains(&module.id) {
                continue;
            }

            if active_slots.is_empty() {
                break;
            }

            let mut placed_in_active = false;
            let mut slot_index_to_remove = -1;

            for (i, (x, y)) in active_slots.iter().enumerate() {
                if self.get_cell(*x, *y).unwrap().module.is_none() {
                    self.place_module(*x, *y, module);
                    placed_module_ids.insert(module.id.clone());
                    slot_index_to_remove = i as i32;
                    placed_in_active = true;
                    break;
                }
            }

            if placed_in_active {
                if slot_index_to_remove != -1 {
                    active_slots.remove(slot_index_to_remove as usize);
                }
            }
        }
    }

    fn _get_orthogonal_neighbors(&self, x: i32, y: i32) -> Vec<(i32, i32)> {
        let mut neighbors = Vec::new();
        let directions = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        let center_cell = match self.get_cell(x, y) {
            Some(cell) => cell,
            None => return neighbors,
        };

        let center_cell_tech = match &center_cell.tech {
            Some(tech) => tech,
            None => return neighbors,
        };

        if center_cell.module.is_none() {
            return neighbors;
        }

        for (dx, dy) in &directions {
            let nx = x + dx;
            let ny = y + dy;

            if let Some(neighbor_cell) = self.get_cell(nx, ny) {
                if neighbor_cell.module.is_some() {
                    if let Some(neighbor_tech) = &neighbor_cell.tech {
                        if neighbor_tech == center_cell_tech {
                            neighbors.push((nx, ny));
                        }
                    }
                }
            }
        }
        neighbors
    }

    fn _calculate_adjacency_factor(&self, x: i32, y: i32, tech: &str) -> f64 {
        let cell = match self.get_cell(x, y) {
            Some(cell) => cell,
            None => return 0.0,
        };

        if cell.module.is_none() {
            return 0.0;
        }

        let cell_adj_type = match &cell.adjacency {
            Some(adj_type) => adj_type,
            None => return 0.0,
        };

        let mut total_adjacency_boost_factor = 0.0;
        let adjacent_cells = self._get_orthogonal_neighbors(x, y);

        for (nx, ny) in adjacent_cells {
            if let Some(adj_cell) = self.get_cell(nx, ny) {
                let adj_cell_type = match &adj_cell.module_type {
                    Some(t) => t,
                    None => continue,
                };
                let adj_cell_adj_type = match &adj_cell.adjacency {
                    Some(t) => t,
                    None => continue,
                };

                let mut weight_from_this_neighbor = 0.0;

                if *cell_adj_type != AdjacencyType::NoAdjacency && *adj_cell_adj_type != AdjacencyType::NoAdjacency {
                    if *cell_adj_type == AdjacencyType::Lesser && *adj_cell_adj_type == AdjacencyType::Greater {
                        if tech == "pulse" || tech == "photonix" {
                            weight_from_this_neighbor = -0.01;
                        } else {
                            weight_from_this_neighbor = 0.0001;
                        }
                    } else {
                        weight_from_this_neighbor = match adj_cell_type {
                            ModuleType::Core => match adj_cell_adj_type {
                                AdjacencyType::Greater => WEIGHT_FROM_GREATER_CORE,
                                AdjacencyType::Lesser => WEIGHT_FROM_LESSER_CORE,
                                AdjacencyType::NoAdjacency => 0.0,
                            },
                            ModuleType::Bonus | ModuleType::Upgrade | ModuleType::Cosmetic | ModuleType::Reactor | ModuleType::Atlantid => {
                                match adj_cell_adj_type {
                                    AdjacencyType::Greater => WEIGHT_FROM_GREATER_BONUS,
                                    AdjacencyType::Lesser => WEIGHT_FROM_LESSER_BONUS,
                                AdjacencyType::NoAdjacency => 0.0,
                                }
                            }
                        };
                    }
                }
                total_adjacency_boost_factor += weight_from_this_neighbor;
            }
        }
        total_adjacency_boost_factor
    }
}

#[pyfunction]
fn populate_all_module_bonuses(grid: &mut Grid, tech: &str, apply_supercharge_first: bool) -> PyResult<()> {
    let mut tech_module_coords = Vec::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            if let Some(cell) = grid.get_cell(x, y) {
                if cell.module.is_some() {
                    if let Some(cell_tech) = &cell.tech {
                        if cell_tech == tech {
                            tech_module_coords.push((x, y));
                        }
                    }
                }
            }
        }
    }

    for y in 0..grid.height {
        for x in 0..grid.width {
            grid.cells[y as usize][x as usize].total = 0.0;
            grid.cells[y as usize][x as usize].adjacency_bonus = 0.0;
        }
    }

    if tech_module_coords.is_empty() {
        return Ok(());
    }

    let mut module_adj_factors = std::collections::HashMap::new();
    for &(x, y) in &tech_module_coords {
        let adj_factor = grid._calculate_adjacency_factor(x, y, tech);
        module_adj_factors.insert((x, y), adj_factor);
    }

    for &(x, y) in &tech_module_coords {
        let adj_factor = module_adj_factors[&(x, y)];
        let cell = grid.get_cell(x, y).unwrap().clone();

        let base_bonus = cell.bonus;
        let is_supercharged = cell.supercharged;
        let is_sc_eligible = cell.sc_eligible;
        let module_type = match cell.module_type.as_ref() {
            Some(mt) => mt,
            None => continue,
        };

        let total_bonus = if apply_supercharge_first {
            let mut calculation_base = base_bonus;
            if is_supercharged && is_sc_eligible {
                calculation_base *= SUPERCHARGE_MULTIPLIER;
            }

            let adjacency_boost_amount = match module_type {
                ModuleType::Core => adj_factor,
                _ => calculation_base * adj_factor,
            };
            base_bonus + adjacency_boost_amount
        } else {
            let adjacency_boost_amount_on_base = match module_type {
                ModuleType::Core => adj_factor,
                _ => base_bonus * adj_factor,
            };

            let mut total_bonus = base_bonus + adjacency_boost_amount_on_base;

            if is_supercharged && is_sc_eligible {
                total_bonus *= SUPERCHARGE_MULTIPLIER;
            }
            total_bonus
        };
        grid.cells[y as usize][x as usize].total = total_bonus;
        grid.cells[y as usize][x as usize].adjacency_bonus = adj_factor;
    }
    Ok(())
}

#[pyfunction]
fn calculate_grid_score(mut grid: Grid, tech: &str, apply_supercharge_first: bool) -> PyResult<f64> {
    populate_all_module_bonuses(&mut grid, tech, apply_supercharge_first)?;
    let total_grid_score: f64 = grid
        .cells
        .iter()
        .flatten()
        .filter(|c| c.tech.is_some() && c.tech.as_deref().unwrap() == tech)
        .map(|c| c.total)
        .sum();
    Ok((total_grid_score * 10000.0).round() / 10000.0)
}

#[pyfunction]
fn simulated_annealing(
    py: Python<'_>,
    grid_json: String,
    tech_modules: Vec<PyRef<Module>>,
    tech: &str,
    initial_temperature: f64,
    cooling_rate: f64,
    stopping_temperature: f64,
    iterations_per_temp: i32,
    progress_callback: Py<PyAny>,
) -> PyResult<(String, f64)> {
    let mut current_grid: Grid = serde_json::from_str(&grid_json).unwrap();
    let tech_modules_vec: Vec<Module> = tech_modules.iter().map(|m| (**m).clone()).collect();
    current_grid.clear_all_modules_of_tech(tech);
    current_grid.place_modules_with_supercharged_priority(&tech_modules_vec, tech);

    let mut current_score = calculate_grid_score(current_grid.clone(), tech, false).unwrap();
    let mut best_grid = current_grid.clone();
    let mut best_score = current_score;

    let mut temperature = initial_temperature;
    let mut rng = rand::thread_rng();
    let mut iteration_count = 0;
    let total_iterations_estimate = (initial_temperature.log(cooling_rate) - stopping_temperature.log(cooling_rate)).abs() * iterations_per_temp as f64;

    let start_time = std::time::Instant::now();
    let mut time_to_best_score: Option<f64> = None;

    // Send initial 'start' message
    Python::with_gil(|py| {
        if !progress_callback.is_none(py) {
            let progress_data = pyo3::types::PyDict::new(py);
            progress_data.set_item("status", "start").unwrap();
            progress_data.set_item("mode", "Full Run").unwrap(); // Assuming Full Run for now
            progress_data.set_item("best_score", best_score).unwrap();
            progress_data.set_item("temperature", temperature).unwrap();
            log::info!("SA: Starting for {}: Best: {:.4}, Temp: {:.2}",
                tech,
                best_score,
                temperature
            );
            progress_callback.call1(py, (progress_data,)).unwrap();
        }
    });

    while temperature > stopping_temperature {
        for _ in 0..iterations_per_temp {
            iteration_count += 1;
            let mut temp_grid = current_grid.clone();
            let move_type: f64 = rng.r#gen::<f64>();

            let original_cells = if move_type < 0.33 {
                swap_modules(&mut temp_grid, tech, &tech_modules_vec)
            } else if move_type < 0.66 {
                move_module(&mut temp_grid, tech, &tech_modules_vec)
            } else {
                swap_adjacent_modules(&mut temp_grid, tech, &tech_modules_vec)
            };

            if let Some(((_x1, _y1), (_x2, _y2))) = original_cells {
                let new_score = calculate_grid_score(temp_grid.clone(), tech, false).unwrap();
                let delta_e = new_score - current_score;

                if delta_e > 0.0 || rng.r#gen::<f64>() < (delta_e / temperature).exp() {
                    current_grid = temp_grid;
                    current_score = new_score;

                    if current_score > best_score {
                        best_grid = current_grid.clone();
                        best_score = current_score;
                        time_to_best_score = Some(start_time.elapsed().as_secs_f64());

                        // Report new best score immediately
                        Python::with_gil(|py| {
                            if !progress_callback.is_none(py) {
                                let progress_data = pyo3::types::PyDict::new(py);
                                progress_data.set_item("status", "in_progress").unwrap();
                                progress_data.set_item("best_score", best_score).unwrap();
                                progress_data.set_item("current_score", current_score).unwrap();
                                progress_data.set_item("temperature", temperature).unwrap();
                                progress_data.set_item("time", start_time.elapsed().as_secs_f64()).unwrap();
                                log::info!("SA: New best score for {}: {:.4} (Temp: {:.2}, Time: {:.2}s)",
                                    tech,
                                    best_score,
                                    temperature,
                                    start_time.elapsed().as_secs_f64()
                                );
                                progress_callback.call1(py, (progress_data,)).unwrap();
                            }
                        });
                    }
                }
            }
        }
        temperature *= cooling_rate;

        // --- Progress Reporting (periodic) ---
        Python::with_gil(|py| {
            if !progress_callback.is_none(py) {
                let progress_percent = (iteration_count as f64 / total_iterations_estimate) * 100.0;
                let progress_data = pyo3::types::PyDict::new(py);
                progress_data.set_item("progress_percent", progress_percent.min(100.0)).unwrap();
                progress_data.set_item("current_score", current_score).unwrap();
                progress_data.set_item("best_score", best_score).unwrap();
                progress_data.set_item("temperature", temperature).unwrap();
                progress_data.set_item("status", "in_progress").unwrap();
                progress_data.set_item("time", start_time.elapsed().as_secs_f64()).unwrap();
            progress_callback.call1(py, (progress_data,)).unwrap();
            }
        });
    }

    let best_grid_json = serde_json::to_string(&best_grid).unwrap();

    // Send final 'finish' message
    Python::with_gil(|py| {
        if !progress_callback.is_none(py) {
            let progress_data = pyo3::types::PyDict::new(py);
            progress_data.set_item("status", "finish").unwrap();
            progress_data.set_item("best_score", best_score).unwrap();
            progress_data.set_item("time", start_time.elapsed().as_secs_f64()).unwrap();
            progress_data.set_item("time_to_best_score", time_to_best_score.unwrap_or(0.0)).unwrap();
            log::info!("SA: Finished for {}: Best: {:.4}, Time: {:.2}s, Time to best: {:.2}s",
                tech,
                best_score,
                start_time.elapsed().as_secs_f64(),
                time_to_best_score.unwrap_or(0.0)
            );
            progress_callback.call1(py, (progress_data,)).unwrap();
        }
    });

    Ok((best_grid_json, best_score))
}

fn swap_modules(grid: &mut Grid, tech: &str, tech_modules_on_grid: &Vec<Module>) -> Option<((i32, i32), (i32, i32))> {
    let mut module_positions: Vec<(i32, i32)> = Vec::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            if let Some(cell) = grid.get_cell(x, y) {
                if cell.tech.as_deref() == Some(tech) {
                    if let Some(module_id) = &cell.module {
                        if tech_modules_on_grid.iter().any(|m| &m.id == module_id) {
                            module_positions.push((x, y));
                        }
                    }
                }
            }
        }
    }

    if module_positions.len() < 2 {
        return None;
    }

    let mut rng = rand::thread_rng();
    let sample: Vec<_> = module_positions.choose_multiple(&mut rng, 2).collect();
    let pos1 = *sample[0];
    let pos2 = *sample[1];

    let (x1, y1) = pos1;
    let (x2, y2) = pos2;

    let cell1_clone = grid.get_cell(x1, y1).unwrap().clone();
    let cell2_clone = grid.get_cell(x2, y2).unwrap().clone();

    let module1 = tech_modules_on_grid.iter().find(|m| Some(m.id.clone()) == cell1_clone.module).unwrap();
    let module2 = tech_modules_on_grid.iter().find(|m| Some(m.id.clone()) == cell2_clone.module).unwrap();

    grid.place_module(x1, y1, module2);
    grid.place_module(x2, y2, module1);

    Some((pos1, pos2))
}

fn move_module(grid: &mut Grid, tech: &str, tech_modules_on_grid: &Vec<Module>) -> Option<((i32, i32), (i32, i32))> {
    let mut module_positions: Vec<(i32, i32)> = Vec::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            if let Some(cell) = grid.get_cell(x, y) {
                if cell.tech.as_deref() == Some(tech) {
                    if let Some(module_id) = &cell.module {
                        if tech_modules_on_grid.iter().any(|m| &m.id == module_id) {
                            module_positions.push((x, y));
                        }
                    }
                }
            }
        }
    }

    if module_positions.is_empty() {
        return None;
    }

    let mut rng = rand::thread_rng();
    let (x_from, y_from) = *module_positions.choose(&mut rng).unwrap();

    let cell_from_clone = grid.get_cell(x_from, y_from).unwrap().clone();
    let module_to_move = tech_modules_on_grid.iter().find(|m| Some(m.id.clone()) == cell_from_clone.module).unwrap();

    let mut empty_active_positions: Vec<(i32, i32)> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    for y in 0..grid.height {
        for x in 0..grid.width {
            if let Some(cell) = grid.get_cell(x, y) {
                if cell.module.is_none() && cell.active {
                    let pos = (x, y);
                    let mut weight = 1.0;

                    if cell.supercharged && module_to_move.sc_eligible {
                        weight *= 5.0;
                    }

                    for (dx, dy) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx >= 0 && nx < grid.width && ny >= 0 && ny < grid.height {
                            if let Some(neighbor_cell) = grid.get_cell(nx, ny) {
                                if neighbor_cell.tech.as_deref() == Some(tech) {
                                    weight *= 2.0;
                                }
                            }
                        }
                    }
                    empty_active_positions.push(pos);
                    weights.push(weight);
                }
            }
        }
    }

    if empty_active_positions.is_empty() {
        return None;
    }

    let (x_to, y_to) = *empty_active_positions.choose_weighted(&mut rng, |item| *weights.get(empty_active_positions.iter().position(|i| i == item).unwrap()).unwrap()).unwrap();

    grid.place_module(x_to, y_to, module_to_move);
    grid.clear_cell(x_from, y_from);

    Some(((x_from, y_from), (x_to, y_to)))
}

fn swap_adjacent_modules(grid: &mut Grid, tech: &str, _tech_modules_on_grid: &Vec<Module>) -> Option<((i32, i32), (i32, i32))> {
    let mut adjacent_pairs: Vec<((i32, i32), (i32, i32))> = Vec::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            if let Some(cell) = grid.get_cell(x, y) {
                if cell.tech.as_deref() == Some(tech) {
                    // Check right neighbor
                    if x + 1 < grid.width {
                        if let Some(right_neighbor) = grid.get_cell(x + 1, y) {
                            if right_neighbor.tech.as_deref() == Some(tech) {
                                adjacent_pairs.push(((x, y), (x + 1, y)));
                            }
                        }
                    }
                    // Check bottom neighbor
                    if y + 1 < grid.height {
                        if let Some(bottom_neighbor) = grid.get_cell(x, y + 1) {
                            if bottom_neighbor.tech.as_deref() == Some(tech) {
                                adjacent_pairs.push(((x, y), (x, y + 1)));
                            }
                        }
                    }
                }
            }
        }
    }

    if adjacent_pairs.is_empty() {
        return None;
    }

    let mut rng = rand::thread_rng();
    let (pos1, pos2) = *adjacent_pairs.choose(&mut rng).unwrap();

    let (x1, y1) = pos1;
    let (x2, y2) = pos2;

    let cell1_clone = grid.get_cell(x1, y1).unwrap().clone();
    let cell2_clone = grid.get_cell(x2, y2).unwrap().clone();

    let module1_id = cell1_clone.module.unwrap();
    let module2_id = cell2_clone.module.unwrap();

    let module1_label = cell1_clone.label.unwrap();
    let module2_label = cell2_clone.label.unwrap();

    let module1_type = cell1_clone.module_type.unwrap();
    let module2_type = cell2_clone.module_type.unwrap();

    let module1_bonus = cell1_clone.bonus;
    let module2_bonus = cell2_clone.bonus;

    let module1_adjacency = cell1_clone.adjacency.unwrap();
    let module2_adjacency = cell2_clone.adjacency.unwrap();

    let module1_sc_eligible = cell1_clone.sc_eligible;
    let module2_sc_eligible = cell2_clone.sc_eligible;

    let module1_image = cell1_clone.image.clone();
    let module2_image = cell2_clone.image.clone();

    grid.cells[y1 as usize][x1 as usize].module = Some(module2_id);
    grid.cells[y1 as usize][x1 as usize].label = Some(module2_label);
    grid.cells[y1 as usize][x1 as usize].module_type = Some(module2_type);
    grid.cells[y1 as usize][x1 as usize].bonus = module2_bonus;
    grid.cells[y1 as usize][x1 as usize].adjacency = Some(module2_adjacency);
    grid.cells[y1 as usize][x1 as usize].sc_eligible = module2_sc_eligible;
    grid.cells[y1 as usize][x1 as usize].image = module2_image;

    grid.cells[y2 as usize][x2 as usize].module = Some(module1_id);
    grid.cells[y2 as usize][x2 as usize].label = Some(module1_label);
    grid.cells[y2 as usize][x2 as usize].module_type = Some(module1_type);
    grid.cells[y2 as usize][x2 as usize].bonus = module1_bonus;
    grid.cells[y2 as usize][x2 as usize].adjacency = Some(module1_adjacency);
    grid.cells[y2 as usize][x2 as usize].sc_eligible = module1_sc_eligible;
    grid.cells[y2 as usize][x2 as usize].image = module1_image;

    Some((pos1, pos2))
}

#[pymodule]
fn rust_scorer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init();
    m.add_function(wrap_pyfunction!(calculate_grid_score, m)?)?;
    m.add_function(wrap_pyfunction!(populate_all_module_bonuses, m)?)?;
    m.add_function(wrap_pyfunction!(simulated_annealing, m)?)?;
    m.add_class::<Grid>()?;
    m.add_class::<Cell>()?;
    m.add_class::<AdjacencyType>()?;
    m.add_class::<ModuleType>()?;
    m.add_class::<Module>()?;
    Ok(())
}
