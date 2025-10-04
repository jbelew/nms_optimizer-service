#include <emscripten/bind.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <set>

struct Module {
    std::string id;
    std::string label;
    std::string tech;
    std::string type;
    double bonus;
    std::string adjacency;
    bool sc_eligible;
    std::string image;
};

struct Cell {
    bool active = false;
    bool supercharged = false;
    std::optional<std::string> module_id;
    std::optional<std::string> tech;
    int x = 0;
    int y = 0;
    std::string label = "";
    std::string type = "";
    double bonus = 0.0;
    std::optional<std::string> adjacency;
    bool sc_eligible_cell = false;
    std::string image = "";
    std::optional<std::pair<int, int>> module_position;
    double total = 0.0;
    double adjacency_bonus = 0.0;
};

class Grid {
public:
    int width;
    int height;
    std::vector<std::vector<Cell>> cells;

    Grid(int w, int h) : width(w), height(h), cells(h, std::vector<Cell>(w)) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                cells[y][x].x = x;
                cells[y][x].y = y;
            }
        }
    }

    Cell& get_cell(int x, int y) {
        return cells[y][x];
    }

    const Cell& get_cell(int x, int y) const {
        return cells[y][x];
    }

    Grid copy() const {
        Grid new_grid(width, height);
        new_grid.cells = this->cells;
        return new_grid;
    }
};

struct SimulatedAnnealingResult {
    Grid grid;
    double score;

    SimulatedAnnealingResult() : grid(0, 0), score(0.0) {}
    SimulatedAnnealingResult(Grid g, double s) : grid(g), score(s) {}
};

const double WEIGHT_FROM_GREATER_BONUS = 0.06;
const double WEIGHT_FROM_LESSER_BONUS = 0.03;
const double WEIGHT_FROM_GREATER_CORE = 0.07;
const double WEIGHT_FROM_LESSER_CORE = 0.04;
const double SUPERCHARGE_MULTIPLIER = 1.25;

double _calculate_adjacency_factor(const Grid& grid, int x, int y, const std::string& tech);
void populate_all_module_bonuses(Grid& grid, const std::string& tech, bool apply_supercharge_first = false);
double calculate_grid_score(const Grid& grid, const std::string& tech, bool apply_supercharge_first = false);
void clear_all_modules_of_tech(Grid& grid, const std::string& tech);
void place_module(Grid& grid, int x, int y, const Module& module);
void place_modules_with_supercharged_priority(Grid& grid, const std::vector<Module>& tech_modules, const std::string& tech);
std::tuple<std::optional<std::pair<int, int>>, Cell, std::optional<std::pair<int, int>>, Cell> swap_modules(Grid& grid, const std::string& tech, const std::vector<Module>& tech_modules_on_grid);
std::tuple<std::optional<std::pair<int, int>>, Cell, std::optional<std::pair<int, int>>, Cell> move_module(Grid& grid, const std::string& tech, const std::vector<Module>& tech_modules_on_grid);
std::tuple<std::optional<std::pair<int, int>>, Cell, std::optional<std::pair<int, int>>, Cell> swap_adjacent_modules(Grid& grid, const std::string& tech, const std::vector<Module>& tech_modules_on_grid);
double calculate_score_delta(const Grid& grid, const std::vector<std::pair<std::pair<int, int>, Cell>>& modified_cells_info, const std::string& tech);
double get_swap_probability(double temperature, double initial_temperature, double stopping_temperature, double initial_swap_probability, double final_swap_probability);
std::optional<Module> find_module_by_id(const std::string& id, const std::vector<Module>& modules);
std::mt19937& get_rng();

SimulatedAnnealingResult simulated_annealing(
    Grid grid,
    std::string ship,
    std::vector<Module> modules,
    std::string tech,
    std::vector<Module> tech_modules,
    double initial_temperature,
    double cooling_rate,
    double stopping_temperature,
    int iterations_per_temp,
    double initial_swap_probability,
    double final_swap_probability,
    bool start_from_current_grid,
    double max_processing_time,
    int max_steps_without_improvement,
    double reheat_factor
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    Grid current_grid = grid.copy();

    if (!start_from_current_grid) {
        clear_all_modules_of_tech(current_grid, tech);
        place_modules_with_supercharged_priority(current_grid, tech_modules, tech);
    }

    double current_score = calculate_grid_score(current_grid, tech);
    Grid best_grid = current_grid.copy();
    double best_score = current_score;
    double temperature = initial_temperature;

    int steps_without_improvement_count = 0;
    int reheat_count = 0;
    int max_reheats = start_from_current_grid ? 2 : 5;
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    while (temperature > stopping_temperature) {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time;
        if (elapsed.count() > max_processing_time) {
            break;
        }

        if (steps_without_improvement_count >= max_steps_without_improvement) {
            if (reheat_count < max_reheats) {
                reheat_count++;
                temperature = initial_temperature * std::pow(reheat_factor, reheat_count);
                current_grid = best_grid.copy();
                current_score = best_score;
                steps_without_improvement_count = 0;
            } else {
                steps_without_improvement_count = -1;
            }
        }

        double swap_prob = get_swap_probability(temperature, initial_temperature, stopping_temperature, initial_swap_probability, final_swap_probability);
        bool made_improvement_in_batch = false;

        for (int i = 0; i < iterations_per_temp; ++i) {
            std::vector<Module> current_modules_on_grid_defs;
            for(int y = 0; y < current_grid.height; ++y) {
                for (int x = 0; x < current_grid.width; ++x) {
                    auto& cell = current_grid.get_cell(x,y);
                    if(cell.tech.has_value() && cell.tech.value() == tech) {
                        auto mod_info = find_module_by_id(cell.module_id.value(), tech_modules);
                        if(mod_info.has_value()) {
                            current_modules_on_grid_defs.push_back(mod_info.value());
                        }
                    }
                }
            }
            if (current_modules_on_grid_defs.empty()) continue;

            std::vector<std::pair<std::pair<int, int>, Cell>> modified_cells_info;
            double move_type_roll = unif(get_rng());

            if (move_type_roll < swap_prob) {
                auto [pos1, orig1, pos2, orig2] = swap_modules(current_grid, tech, current_modules_on_grid_defs);
                if (pos1.has_value()) modified_cells_info = {{pos1.value(), orig1}, {pos2.value(), orig2}};
            } else if (move_type_roll < swap_prob + (1.0 - swap_prob) / 2.0) {
                auto [pos1, orig1, pos2, orig2] = swap_adjacent_modules(current_grid, tech, current_modules_on_grid_defs);
                 if (pos1.has_value()) modified_cells_info = {{pos1.value(), orig1}, {pos2.value(), orig2}};
            } else {
                auto [pos_from, orig_from, pos_to, orig_to] = move_module(current_grid, tech, current_modules_on_grid_defs);
                if (pos_from.has_value()) modified_cells_info = {{pos_from.value(), orig_from}, {pos_to.value(), orig_to}};
            }

            if (modified_cells_info.empty()) continue;

            double delta_e = calculate_score_delta(current_grid, modified_cells_info, tech);
            double neighbor_score = current_score + delta_e;

            if (delta_e > 0 || unif(get_rng()) < std::exp(delta_e / temperature)) {
                current_score = neighbor_score;
                if (current_score > best_score) {
                    best_grid = current_grid.copy();
                    best_score = current_score;
                    made_improvement_in_batch = true;
                }
            } else {
                for (const auto& info : modified_cells_info) {
                    const auto& pos = info.first;
                    const auto& original_cell_data = info.second;
                    current_grid.cells[pos.second][pos.first] = original_cell_data;
                }
            }
        }

        if (made_improvement_in_batch) {
            steps_without_improvement_count = 0;
        } else if (steps_without_improvement_count != -1) {
            steps_without_improvement_count++;
        }
        temperature *= cooling_rate;
    }

    return SimulatedAnnealingResult(best_grid, best_score);
}

std::vector<Cell> _get_orthogonal_neighbors(const Grid& grid, int x, int y) {
    std::vector<Cell> neighbors;
    std::vector<std::pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    const auto& center_cell = grid.get_cell(x, y);
    auto center_cell_tech = center_cell.tech;

    if (!center_cell_tech.has_value() || !center_cell.module_id.has_value()) {
        return neighbors;
    }

    for (const auto& dir : directions) {
        int nx = x + dir.first;
        int ny = y + dir.second;
        if (nx >= 0 && nx < grid.width && ny >= 0 && ny < grid.height) {
            const auto& neighbor_cell = grid.get_cell(nx, ny);
            if (neighbor_cell.module_id.has_value() && neighbor_cell.tech.has_value() && neighbor_cell.tech.value() == center_cell_tech.value()) {
                neighbors.push_back(neighbor_cell);
            }
        }
    }
    return neighbors;
}

double _calculate_adjacency_factor(const Grid& grid, int x, int y, const std::string& tech) {
    const auto& cell = grid.get_cell(x, y);
    if (!cell.module_id.has_value()) {
        return 0.0;
    }

    auto cell_adj_type_opt = cell.adjacency;
    double total_adjacency_boost_factor = 0.0;
    auto adjacent_cells = _get_orthogonal_neighbors(grid, x, y);

    for (const auto& adj_cell : adjacent_cells) {
        auto adj_cell_type = adj_cell.type;
        auto adj_cell_adj_type_opt = adj_cell.adjacency;
        double weight_from_this_neighbor = 0.0;

        if (adj_cell_adj_type_opt.has_value() && cell_adj_type_opt.has_value() && adj_cell_adj_type_opt.value() != "none" && cell_adj_type_opt.value() != "none") {
            std::string temp_cell_adj_type = cell_adj_type_opt.value();
            if (temp_cell_adj_type.rfind("greater_", 0) == 0) temp_cell_adj_type = "greater";
            if (temp_cell_adj_type.rfind("lesser_", 0) == 0) temp_cell_adj_type = "lesser";

            std::string temp_adj_cell_adj_type = adj_cell_adj_type_opt.value();
            if (temp_adj_cell_adj_type.rfind("greater_", 0) == 0) temp_adj_cell_adj_type = "greater";
            if (temp_adj_cell_adj_type.rfind("lesser_", 0) == 0) temp_adj_cell_adj_type = "lesser";

            if (temp_cell_adj_type == "lesser" && temp_adj_cell_adj_type == "greater") {
                if (tech == "pulse" || tech == "photonix") {
                    weight_from_this_neighbor = -0.01;
                } else {
                    weight_from_this_neighbor = 0.0001;
                }
            } else {
                if (adj_cell_type == "core") {
                    if (temp_adj_cell_adj_type == "greater") weight_from_this_neighbor = WEIGHT_FROM_GREATER_CORE;
                    else if (temp_adj_cell_adj_type == "lesser") weight_from_this_neighbor = WEIGHT_FROM_LESSER_CORE;
                } else if (adj_cell_type == "bonus" || adj_cell_type == "upgrade" || adj_cell_type == "cosmetic" || adj_cell_type == "reactor" || adj_cell_type == "atlantid") {
                    if (temp_adj_cell_adj_type == "greater") weight_from_this_neighbor = WEIGHT_FROM_GREATER_BONUS;
                    else if (temp_adj_cell_adj_type == "lesser") weight_from_this_neighbor = WEIGHT_FROM_LESSER_BONUS;
                }
            }
        }
        total_adjacency_boost_factor += weight_from_this_neighbor;
    }
    return total_adjacency_boost_factor;
}

void populate_all_module_bonuses(Grid& grid, const std::string& tech, bool apply_supercharge_first) {
    std::vector<std::pair<int, int>> tech_module_coords;
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            auto& cell = grid.get_cell(x, y);
            if (cell.module_id.has_value() && cell.tech.has_value() && cell.tech.value() == tech) {
                tech_module_coords.push_back({x, y});
                cell.total = 0.0;
                cell.adjacency_bonus = 0.0;
            }
        }
    }

    if (tech_module_coords.empty()) {
        return;
    }

    std::map<std::pair<int, int>, double> module_adj_factors;
    for (const auto& coords : tech_module_coords) {
        module_adj_factors[coords] = _calculate_adjacency_factor(grid, coords.first, coords.second, tech);
    }

    for (const auto& coords : tech_module_coords) {
        auto& cell = grid.get_cell(coords.first, coords.second);
        double base_bonus = cell.bonus;
        bool is_supercharged = cell.supercharged;
        bool is_sc_eligible = cell.sc_eligible_cell;
        double adj_factor = module_adj_factors[coords];
        std::string module_type = cell.type;
        double total_bonus = 0.0;

        if (module_type == "core") {
            total_bonus = adj_factor;
        } else {
            total_bonus = base_bonus + (base_bonus * adj_factor);
        }

        if (is_supercharged && is_sc_eligible) {
            total_bonus *= SUPERCHARGE_MULTIPLIER;
        }
        cell.total = round(total_bonus * 10000.0) / 10000.0;
        cell.adjacency_bonus = round(adj_factor * 10000.0) / 10000.0;
    }
}

double calculate_grid_score(const Grid& grid, const std::string& tech, bool apply_supercharge_first) {
    Grid temp_grid = grid.copy();
    populate_all_module_bonuses(temp_grid, tech, apply_supercharge_first);
    double total_grid_score = 0.0;
    for (int y = 0; y < temp_grid.height; ++y) {
        for (int x = 0; x < temp_grid.width; ++x) {
            const auto& cell = temp_grid.get_cell(x, y);
            if (cell.module_id.has_value() && cell.tech.has_value() && cell.tech.value() == tech) {
                total_grid_score += cell.total;
            }
        }
    }
    return round(total_grid_score * 10000.0) / 10000.0;
}

void clear_all_modules_of_tech(Grid& grid, const std::string& tech) {
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            Cell& cell = grid.get_cell(x, y);
            if (cell.tech.has_value() && cell.tech.value() == tech) {
                cell.module_id.reset();
                cell.label = "";
                cell.tech.reset();
                cell.type = "";
                cell.bonus = 0.0;
                cell.total = 0.0;
                cell.adjacency_bonus = 0.0;
                cell.adjacency.reset();
                cell.sc_eligible_cell = false;
                cell.image = "";
                cell.module_position.reset();
            }
        }
    }
}

void place_module(Grid& grid, int x, int y, const Module& module) {
    Cell& cell = grid.get_cell(x, y);
    cell.module_id = module.id;
    cell.label = module.label;
    cell.tech = module.tech;
    cell.type = module.type;
    cell.bonus = module.bonus;
    if (!module.adjacency.empty()) {
        cell.adjacency = module.adjacency;
    } else {
        cell.adjacency.reset();
    }
    cell.sc_eligible_cell = module.sc_eligible;
    cell.image = module.image;
    cell.module_position = std::make_pair(x, y);
}

void place_modules_with_supercharged_priority(Grid& grid, const std::vector<Module>& tech_modules, const std::string& tech) {
    std::vector<std::pair<int, int>> supercharged_slots;
    std::vector<std::pair<int, int>> active_slots;
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            const auto& cell = grid.get_cell(x, y);
            if (!cell.module_id.has_value() && cell.active) {
                if (cell.supercharged) {
                    supercharged_slots.push_back({x, y});
                } else {
                    active_slots.push_back({x, y});
                }
            }
        }
    }

    std::optional<Module> core_module;
    std::vector<Module> bonus_modules;
    for(const auto& m : tech_modules) {
        if (m.type == "core") {
            core_module = m;
        } else {
            bonus_modules.push_back(m);
        }
    }
    std::sort(bonus_modules.begin(), bonus_modules.end(), [](const Module& a, const Module& b) {
        return a.bonus > b.bonus;
    });

    size_t num_available_positions = supercharged_slots.size() + active_slots.size();
    std::vector<Module> modules_to_place_candidates;
    if (core_module.has_value()) {
        modules_to_place_candidates.push_back(core_module.value());
    }
    modules_to_place_candidates.insert(modules_to_place_candidates.end(), bonus_modules.begin(), bonus_modules.end());

    std::vector<Module> modules_to_place;
    if (modules_to_place_candidates.size() > num_available_positions) {
        modules_to_place.assign(modules_to_place_candidates.begin(), modules_to_place_candidates.begin() + num_available_positions);
    } else {
        modules_to_place = modules_to_place_candidates;
    }

    std::vector<Module> sc_eligible_modules;
    std::vector<Module> non_sc_eligible_modules;
    for (const auto& m : modules_to_place) {
        if (m.sc_eligible) {
            sc_eligible_modules.push_back(m);
        } else {
            non_sc_eligible_modules.push_back(m);
        }
    }

    auto sort_func = [](const Module& a, const Module& b) {
        bool a_is_core = a.type == "core";
        bool b_is_core = b.type == "core";
        if (a_is_core != b_is_core) {
            return a_is_core;
        }
        return a.bonus > b.bonus;
    };
    std::sort(sc_eligible_modules.begin(), sc_eligible_modules.end(), sort_func);
    std::sort(non_sc_eligible_modules.begin(), non_sc_eligible_modules.end(), sort_func);

    std::vector<std::string> placed_module_ids;

    std::vector<Module> remaining_sc_eligible;
    std::shuffle(supercharged_slots.begin(), supercharged_slots.end(), get_rng());
    for (const auto& module : sc_eligible_modules) {
        if (!supercharged_slots.empty()) {
            auto slot = supercharged_slots.back();
            supercharged_slots.pop_back();
            place_module(grid, slot.first, slot.second, module);
            placed_module_ids.push_back(module.id);
        } else {
            remaining_sc_eligible.push_back(module);
        }
    }

    std::vector<Module> modules_for_active_slots = non_sc_eligible_modules;
    modules_for_active_slots.insert(modules_for_active_slots.end(), remaining_sc_eligible.begin(), remaining_sc_eligible.end());
    std::sort(modules_for_active_slots.begin(), modules_for_active_slots.end(), sort_func);

    std::shuffle(active_slots.begin(), active_slots.end(), get_rng());
    for (const auto& module : modules_for_active_slots) {
        if (std::find(placed_module_ids.begin(), placed_module_ids.end(), module.id) != placed_module_ids.end()) {
            continue;
        }
        if (!active_slots.empty()) {
            auto slot = active_slots.back();
            active_slots.pop_back();
            place_module(grid, slot.first, slot.second, module);
            placed_module_ids.push_back(module.id);
        } else {
            break;
        }
    }
}

std::mt19937& get_rng() {
    static std::random_device rd;
    static std::mt19937 g(rd());
    return g;
}

std::optional<Module> find_module_by_id(const std::string& id, const std::vector<Module>& modules) {
    for (const auto& mod : modules) {
        if (mod.id == id) {
            return mod;
        }
    }
    return std::nullopt;
}

std::tuple<std::optional<std::pair<int, int>>, Cell, std::optional<std::pair<int, int>>, Cell> swap_modules(Grid& grid, const std::string& tech, const std::vector<Module>& tech_modules_on_grid) {
    std::vector<std::pair<int, int>> module_positions;
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            const auto& cell = grid.get_cell(x, y);
            if (cell.tech.has_value() && cell.tech.value() == tech) {
                module_positions.push_back({x, y});
            }
        }
    }

    if (module_positions.size() < 2) {
        return {std::nullopt, Cell(), std::nullopt, Cell()};
    }

    std::shuffle(module_positions.begin(), module_positions.end(), get_rng());
    std::pair<int, int> pos1 = module_positions[0];
    std::pair<int, int> pos2 = module_positions[1];

    Cell original_cell_1_data = grid.get_cell(pos1.first, pos1.second);
    Cell original_cell_2_data = grid.get_cell(pos2.first, pos2.second);

    auto module1_info = find_module_by_id(original_cell_1_data.module_id.value(), tech_modules_on_grid);
    auto module2_info = find_module_by_id(original_cell_2_data.module_id.value(), tech_modules_on_grid);

    if(module1_info.has_value() && module2_info.has_value()){
        place_module(grid, pos1.first, pos1.second, module2_info.value());
        place_module(grid, pos2.first, pos2.second, module1_info.value());
    }

    return {pos1, original_cell_1_data, pos2, original_cell_2_data};
}

std::tuple<std::optional<std::pair<int, int>>, Cell, std::optional<std::pair<int, int>>, Cell> move_module(Grid& grid, const std::string& tech, const std::vector<Module>& tech_modules_on_grid) {
    std::vector<std::pair<int, int>> module_positions;
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            const auto& cell = grid.get_cell(x, y);
            if (cell.tech.has_value() && cell.tech.value() == tech) {
                module_positions.push_back({x, y});
            }
        }
    }

    if (module_positions.empty()) {
        return {std::nullopt, Cell(), std::nullopt, Cell()};
    }

    std::uniform_int_distribution<int> module_dist(0, module_positions.size() - 1);
    std::pair<int, int> pos_from = module_positions[module_dist(get_rng())];

    Cell original_from_cell_data = grid.get_cell(pos_from.first, pos_from.second);
    auto module_to_move_info = find_module_by_id(original_from_cell_data.module_id.value(), tech_modules_on_grid);

    std::vector<std::pair<int, int>> empty_active_positions;
    std::vector<double> weights;
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            const auto& cell = grid.get_cell(x, y);
            if (!cell.module_id.has_value() && cell.active) {
                empty_active_positions.push_back({x, y});
                double weight = 1.0;
                if (cell.supercharged && module_to_move_info.has_value() && module_to_move_info.value().sc_eligible) {
                    weight *= 5.0;
                }
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < grid.width && ny >= 0 && ny < grid.height) {
                            const auto& neighbor = grid.get_cell(nx, ny);
                            if (neighbor.tech.has_value() && neighbor.tech.value() == tech) {
                                weight *= 2.0;
                            }
                        }
                    }
                }
                weights.push_back(weight);
            }
        }
    }

    if (empty_active_positions.empty()) {
        return {std::nullopt, Cell(), std::nullopt, Cell()};
    }

    std::discrete_distribution<> dist(weights.begin(), weights.end());
    std::pair<int, int> pos_to = empty_active_positions[dist(get_rng())];

    Cell original_to_cell_data = grid.get_cell(pos_to.first, pos_to.second);

    if(module_to_move_info.has_value()){
        place_module(grid, pos_to.first, pos_to.second, module_to_move_info.value());

        Cell& from_cell = grid.get_cell(pos_from.first, pos_from.second);
        from_cell.module_id.reset();
        from_cell.label = "";
        from_cell.tech.reset();
        from_cell.type = "";
        from_cell.bonus = 0.0;
        from_cell.total = 0.0;
        from_cell.adjacency_bonus = 0.0;
        from_cell.adjacency.reset();
        from_cell.sc_eligible_cell = false;
        from_cell.image = "";
        from_cell.module_position.reset();
    }

    return {pos_from, original_from_cell_data, pos_to, original_to_cell_data};
}

std::tuple<std::optional<std::pair<int, int>>, Cell, std::optional<std::pair<int, int>>, Cell> swap_adjacent_modules(Grid& grid, const std::string& tech, const std::vector<Module>& tech_modules_on_grid) {
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> adjacent_pairs;
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            const auto& cell = grid.get_cell(x, y);
            if (cell.tech.has_value() && cell.tech.value() == tech) {
                if (x + 1 < grid.width) {
                    const auto& right_neighbor = grid.get_cell(x + 1, y);
                    if (right_neighbor.tech.has_value() && right_neighbor.tech.value() == tech) {
                        adjacent_pairs.push_back({{x, y}, {x + 1, y}});
                    }
                }
                if (y + 1 < grid.height) {
                    const auto& bottom_neighbor = grid.get_cell(x, y + 1);
                    if (bottom_neighbor.tech.has_value() && bottom_neighbor.tech.value() == tech) {
                        adjacent_pairs.push_back({{x, y}, {x, y + 1}});
                    }
                }
            }
        }
    }

    if (adjacent_pairs.empty()) {
        return {std::nullopt, Cell(), std::nullopt, Cell()};
    }

    std::uniform_int_distribution<int> dist(0, adjacent_pairs.size() - 1);
    auto pair_to_swap = adjacent_pairs[dist(get_rng())];
    std::pair<int, int> pos1 = pair_to_swap.first;
    std::pair<int, int> pos2 = pair_to_swap.second;

    Cell original_cell_1_data = grid.get_cell(pos1.first, pos1.second);
    Cell original_cell_2_data = grid.get_cell(pos2.first, pos2.second);

    auto module1_info = find_module_by_id(original_cell_1_data.module_id.value(), tech_modules_on_grid);
    auto module2_info = find_module_by_id(original_cell_2_data.module_id.value(), tech_modules_on_grid);

    if(module1_info.has_value() && module2_info.has_value()){
        place_module(grid, pos1.first, pos1.second, module2_info.value());
        place_module(grid, pos2.first, pos2.second, module1_info.value());
    }

    return {pos1, original_cell_1_data, pos2, original_cell_2_data};
}

double calculate_score_delta(const Grid& grid, const std::vector<std::pair<std::pair<int, int>, Cell>>& modified_cells_info, const std::string& tech) {
    Grid old_grid = grid.copy();
    for (const auto& info : modified_cells_info) {
        const auto& pos = info.first;
        const auto& original_cell_data = info.second;
        old_grid.cells[pos.second][pos.first] = original_cell_data;
    }

    std::set<std::pair<int, int>> affected_coords;
    for (const auto& info : modified_cells_info) {
        const auto& pos = info.first;
        affected_coords.insert(pos);
        auto new_neighbors = _get_orthogonal_neighbors(grid, pos.first, pos.second);
        for(const auto& neighbor : new_neighbors) {
            affected_coords.insert({neighbor.x, neighbor.y});
        }
        auto old_neighbors = _get_orthogonal_neighbors(old_grid, pos.first, pos.second);
        for(const auto& neighbor : old_neighbors) {
            affected_coords.insert({neighbor.x, neighbor.y});
        }
    }

    double old_score_contribution = 0.0;
    populate_all_module_bonuses(old_grid, tech);
    for (const auto& coords : affected_coords) {
        const auto& cell = old_grid.get_cell(coords.first, coords.second);
        if (cell.tech.has_value() && cell.tech.value() == tech) {
            old_score_contribution += cell.total;
        }
    }

    double new_score_contribution = 0.0;
    Grid current_grid_copy = grid.copy();
    populate_all_module_bonuses(current_grid_copy, tech);
    for (const auto& coords : affected_coords) {
        const auto& cell = current_grid_copy.get_cell(coords.first, coords.second);
        if (cell.tech.has_value() && cell.tech.value() == tech) {
            new_score_contribution += cell.total;
        }
    }

    return new_score_contribution - old_score_contribution;
}

double get_swap_probability(double temperature, double initial_temperature, double stopping_temperature, double initial_swap_probability, double final_swap_probability) {
    if (temperature >= initial_temperature) {
        return initial_swap_probability;
    }
    if (temperature <= stopping_temperature) {
        return final_swap_probability;
    }
    double progress = (initial_temperature - temperature) / (initial_temperature - stopping_temperature);
    return initial_swap_probability - (initial_swap_probability - final_swap_probability) * progress;
}


EMSCRIPTEN_BINDINGS(my_module) {
    emscripten::value_object<Module>("Module")
        .field("id", &Module::id)
        .field("label", &Module::label)
        .field("tech", &Module::tech)
        .field("type", &Module::type)
        .field("bonus", &Module::bonus)
        .field("adjacency", &Module::adjacency)
        .field("sc_eligible", &Module::sc_eligible)
        .field("image", &Module::image);

    emscripten::value_object<Cell>("Cell")
        .field("active", &Cell::active)
        .field("supercharged", &Cell::supercharged)
        .field("module_id", &Cell::module_id)
        .field("tech", &Cell::tech)
        .field("x", &Cell::x)
        .field("y", &Cell::y)
        .field("adjacency", &Cell::adjacency);

    emscripten::class_<Grid>("Grid")
        .constructor<int, int>()
        .function("get_cell", emscripten::select_overload<Cell&(int, int)>(&Grid::get_cell), emscripten::allow_raw_pointers())
        .function("copy", &Grid::copy)
        .property("width", &Grid::width)
        .property("height", &Grid::height)
        .property("cells", &Grid::cells);

    emscripten::function("simulated_annealing", &simulated_annealing);

    emscripten::value_object<SimulatedAnnealingResult>("SimulatedAnnealingResult")
        .field("grid", &SimulatedAnnealingResult::grid)
        .field("score", &SimulatedAnnealingResult::score);

    emscripten::register_vector<Module>("VectorModule");
    emscripten::register_vector<Cell>("VectorCell");
    emscripten::register_vector<std::vector<Cell>>("VectorVectorCell");
    emscripten::register_optional<std::string>();
}