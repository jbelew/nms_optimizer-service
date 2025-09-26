const runButton = document.getElementById('run-button');
const statusDiv = document.getElementById('status');
const resultDiv = document.getElementById('result');

// --- Sample Data for the Test ---
// This data mimics the structure the Python code expects.
const sampleGridData = {
    width: 6,
    height: 5,
    cells: Array(5).fill(0).map(() => Array(6).fill(0).map(() => ({
        module: null, label: null, value: 0, type: "", total: 0.0,
        adjacency_bonus: 0.0, bonus: 0.0, active: true, adjacency: false,
        tech: null, supercharged: false, sc_eligible: false, image: null
    })))
};
// Add some supercharged slots for a more realistic test
sampleGridData.cells[1][1].supercharged = true;
sampleGridData.cells[2][3].supercharged = true;

const sampleModulesData = {
    "hauler": { // Ship
        "types": {
            "tech": [
                {
                    "key": "pulse",
                    "label": "Pulse Engine",
                    "modules": [
                        { "id": "P", "label": "P", "type": "core", "bonus": 0, "adjacency": "greater", "sc_eligible": true, "image": "P.png" },
                        { "id": "A", "label": "A", "type": "bonus", "bonus": 10, "adjacency": "lesser", "sc_eligible": true, "image": "A.png" },
                        { "id": "B", "label": "B", "type": "bonus", "bonus": 9, "adjacency": "lesser", "sc_eligible": true, "image": "B.png" },
                        { "id": "C", "label": "C", "type": "bonus", "bonus": 8, "adjacency": "lesser", "sc_eligible": true, "image": "C.png" },
                        { "id": "X", "label": "X", "type": "bonus", "bonus": 15, "adjacency": "greater", "sc_eligible": true, "image": "X.png" },
                        { "id": "Y", "label": "Y", "type": "bonus", "bonus": 14, "adjacency": "greater", "sc_eligible": true, "image": "Y.png" }
                    ]
                }
            ]
        }
    }
};

const ship = "hauler";
const tech = "pulse";
const playerRewards = [];

// --- Main Application Logic ---

// Function to print a text representation of the grid to the console
function printGrid(grid) {
    let output = "Grid Result:\n";
    for (let y = 0; y < grid.height; y++) {
        let row = "";
        for (let x = 0; x < grid.width; x++) {
            const cell = grid.cells[y][x];
            row += `[${cell.module || ' '}] `;
        }
        output += row + "\n";
    }
    console.log(output);
}


async function main() {
    runButton.disabled = true;
    statusDiv.textContent = 'Loading Pyodide runtime...';
    let pyodide = await loadPyodide();

    statusDiv.textContent = 'Pyodide loaded. Loading micropip...';
    await pyodide.loadPackage("micropip");
    const micropip = pyodide.pyimport("micropip");

    statusDiv.textContent = 'Installing custom optimizer package...';
    // We are serving from the `wasm` directory, so the package is one level down.
    await micropip.install('./sa_wasm_optimizer');

    statusDiv.textContent = 'Package installed. Importing function...';
    const optimizerModule = pyodide.pyimport("sa_wasm_optimizer.main");

    statusDiv.textContent = 'Ready to run optimization.';
    runButton.disabled = false;

    runButton.onclick = async () => {
        runButton.disabled = true;
        resultDiv.innerHTML = "";
        statusDiv.textContent = "Running optimization...";

        // Define the JS progress callback
        const progressCallback = (progress) => {
            const data = progress.toJs({ dict_converter: Object.fromEntries });
            statusDiv.innerHTML = `
                Status: ${data.status}<br>
                Best Score: ${data.best_score.toFixed(4)}<br>
                Temperature: ${data.current_temp.toFixed(2)}<br>
                Progress: ${data.progress_percent.toFixed(1)}%
            `;
        };

        try {
            // Call the Python function
            const resultProxy = await optimizerModule.run_optimization(
                sampleGridData,
                sampleModulesData[ship],
                ship,
                tech,
                playerRewards,
                progressCallback
            );

            const result = resultProxy.toJs({ dict_converter: Object.fromEntries });

            if (result.get("success")) {
                statusDiv.textContent = "Optimization complete!";
                const finalGrid = result.get("grid");
                const finalScore = result.get("score");

                let gridHtml = '<table>';
                for (let y = 0; y < finalGrid.get("height"); y++) {
                    gridHtml += '<tr>';
                    for (let x = 0; x < finalGrid.get("width"); x++) {
                        const cell = finalGrid.get("cells")[y][x];
                        const label = cell.get("module") || '';
                        const isSupercharged = cell.get("supercharged") ? 'supercharged' : '';
                        gridHtml += `<td class="${isSupercharged}">${label}</td>`;
                    }
                    gridHtml += '</tr>';
                }
                gridHtml += '</table>';

                resultDiv.innerHTML = `<strong>Final Score: ${finalScore.toFixed(4)}</strong>${gridHtml}`;
                console.log("Final Grid:", finalGrid);
                printGrid(finalGrid.toJs({ dict_converter: Object.fromEntries }));

            } else {
                statusDiv.textContent = `Error: ${result.get("error")}`;
            }

        } catch (error) {
            statusDiv.textContent = `An error occurred: ${error}`;
            console.error(error);
        } finally {
            runButton.disabled = false;
        }
    };
}

main();