use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AdjacencyType {
    #[serde(rename = "greater")]
    Greater,
    #[serde(rename = "lesser")]
    Lesser,
    #[serde(rename = "none")]
    None,
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
    #[serde(rename = "type")]
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

                if *cell_adj_type != AdjacencyType::None && *adj_cell_adj_type != AdjacencyType::None {
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
                                AdjacencyType::None => 0.0,
                            },
                            ModuleType::Bonus | ModuleType::Upgrade | ModuleType::Cosmetic | ModuleType::Reactor | ModuleType::Atlantid => {
                                match adj_cell_adj_type {
                                    AdjacencyType::Greater => WEIGHT_FROM_GREATER_BONUS,
                                    AdjacencyType::Lesser => WEIGHT_FROM_LESSER_BONUS,
                                    AdjacencyType::None => 0.0,
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

    pub fn populate_all_module_bonuses(&mut self, tech: &str, apply_supercharge_first: bool) {
        let mut tech_module_coords = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(cell) = self.get_cell(x, y) {
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

        for &(_, y) in &tech_module_coords {
            for x in 0..self.width {
                self.cells[y as usize][x as usize].total = 0.0;
                self.cells[y as usize][x as usize].adjacency_bonus = 0.0;
            }
        }

        if tech_module_coords.is_empty() {
            return;
        }

        let mut module_adj_factors = std::collections::HashMap::new();
        for &(x, y) in &tech_module_coords {
            let adj_factor = self._calculate_adjacency_factor(x, y, tech);
            module_adj_factors.insert((x, y), adj_factor);
        }

        for &(x, y) in &tech_module_coords {
            let adj_factor = module_adj_factors[&(x, y)];
            let cell = self.get_cell(x, y).unwrap().clone();

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
            self.cells[y as usize][x as usize].total = total_bonus;
            self.cells[y as usize][x as usize].adjacency_bonus = adj_factor;
        }
    }
}

#[pyfunction]
fn calculate_grid_score(mut grid: Grid, tech: &str, apply_supercharge_first: bool) -> PyResult<f64> {
    grid.populate_all_module_bonuses(tech, apply_supercharge_first);
    let total_grid_score: f64 = grid
        .cells
        .iter()
        .flatten()
        .filter(|c| c.tech.is_some() && c.tech.as_deref().unwrap() == tech)
        .map(|c| c.total)
        .sum();
    Ok((total_grid_score * 10000.0).round() / 10000.0)
}

#[pymodule]
fn rust_scorer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_grid_score, m)?)?;
    m.add_class::<Grid>()?;
    m.add_class::<Cell>()?;
    m.add_class::<AdjacencyType>()?;
    m.add_class::<ModuleType>()?;
    Ok(())
}
