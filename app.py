import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for plotting

from flask import Flask, render_template, jsonify, request, Response, send_file
import threading
import time
import io
from collections import deque
from collections import defaultdict
import json
import os
import pickle  # Added for pickle functionality
from pathlib import Path
from typing import Optional, List, Dict, Any
import colorsys # Added for HLS to RGB conversion
import traceback # For detailed error logging

# Import simulation logic (must reside in simulation_logic.py)
try:
    from simulation_logic import AdvancedEvolutionEnvironment, np, plt, NeuralAgent
    from simulation_logic import Genome, AgentMemory
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import from simulation_logic.py. Details: {e}")
    raise


PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"


# --- Restricted Unpickler for safe .pkl uploads ---
SAFE_MODULES = {
    'simulation_logic': {'NeuralAgent', 'AdvancedEvolutionEnvironment', 'Genome', 'AgentMemory'},
    'numpy': None,            # Allow all numpy classes
    'numpy.core': None,
    'numpy.core.multiarray': None,
    'numpy._core': None,
    'numpy._core.multiarray': None,
    'numpy.dtypes': None,
    'collections': {'defaultdict', 'deque', 'OrderedDict'},
    'builtins': {'list', 'dict', 'set', 'tuple', 'int', 'float', 'str', 'bool', 'complex', 'bytes', 'type', 'frozenset', 'slice', 'range'},
    '_codecs': {'encode'},
    'copyreg': {'_reconstructor'},
}

class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows known safe classes to prevent arbitrary code execution."""
    def find_class(self, module: str, name: str):
        allowed = SAFE_MODULES.get(module)
        if allowed is None and module in SAFE_MODULES:
            # Module allowed with all classes (e.g. numpy)
            return super().find_class(module, name)
        if allowed is not None and name in allowed:
            return super().find_class(module, name)
        # Check numpy submodules
        if module.startswith('numpy'):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Blocked unsafe class: {module}.{name}. "
            f"Only simulation_logic, numpy, and basic Python types are allowed."
        )

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)

# --- Global Simulation State ---
ecosystem: Optional[AdvancedEvolutionEnvironment] = None
simulation_thread: Optional[threading.Thread] = None
simulation_lock = threading.Lock()
stop_event = threading.Event()

is_running: bool = False
current_step: int = 0
total_steps_setting: int = 1000
population_size_setting: int = 50 # Default population size for new sims via UI
initial_environmental_pressure_setting: float = 0.3
initial_resource_availability_setting: float = 1.0
initial_environment_type_setting: str = "balanced"
verbose_logging_enabled: bool = True

latest_chart_image_bytes: Optional[bytes] = None
gui_console_messages = deque(maxlen=200)

# For arena visualization:
latest_arena_image_bytes: Optional[bytes] = None

# --- Persistence Settings (mirroring simulation_logic.py) ---
PERSISTENCE_FILENAME = "best_population.pkl" # Should match simulation_logic.py
PERSISTENCE_PATH = PROJECT_ROOT / PERSISTENCE_FILENAME
ELITE_SAVE_COUNT = 50 # Should match simulation_logic.py


def log_to_gui(message: str):
    """Append a timestamped message to the GUI console buffer."""
    timestamp = time.strftime("%H:%M:%S")
    gui_console_messages.append(f"[{timestamp}] {message}")


def get_species_color_for_matplotlib(species_id_str: str) -> tuple[float, float, float]:
    """
    Generates an RGB color tuple for a given species ID string.
    Consistent with HSL-based color generation in frontend JavaScript.
    """
    try:
        species_id = int(species_id_str)
        # Hue calculation consistent with JS: (id * 137) % 360
        h = ((species_id * 137) % 360) / 360.0  # Normalized hue (0-1)
        l = 0.50  # Lightness (fixed, similar to JS)
        s = 0.70  # Saturation (fixed, similar to JS)
        return colorsys.hls_to_rgb(h, l, s)
    except ValueError:
        return (0.5, 0.5, 0.5)  # Default gray for unparseable or missing ID


def generate_arena_frame() -> Optional[bytes]:
    """
    Generate a 2D arena visualization of agent positions, colored by species.
    Returns PNG image bytes or None if ecosystem/agents are unavailable.
    """
    global latest_arena_image_bytes
    global current_step 

    with simulation_lock:
        local_ecosystem = ecosystem
        step_for_title = current_step
        # Snapshot agents under lock for thread safety
        agents_snapshot = list(local_ecosystem.agents) if local_ecosystem and getattr(local_ecosystem, "agents", None) else []
        world_size = local_ecosystem.world_size if local_ecosystem else (100.0, 100.0)

    if not local_ecosystem:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        agent_data_to_plot = []
        current_agents_list = agents_snapshot

        if not current_agents_list: 
             ax.text(world_size[0] / 2, world_size[1] / 2, "No agents",
                    ha='center', va='center', fontsize=12, color='grey')
        else:
            for agent_obj in current_agents_list: 
                if agent_obj.energy > 0:
                    agent_data_to_plot.append({
                        'position': agent_obj.position,
                        'species_id': str(agent_obj.species_id)
                    })

        ax.set_title(f"Agent Arena at Step {step_for_title}")
        ax.set_xlim(0, world_size[0])
        ax.set_ylim(0, world_size[1])
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True, linestyle=':', alpha=0.5)

        if not agent_data_to_plot: 
            # This condition is hit if current_agents_list was non-empty but all agents had energy <= 0
            # or if current_agents_list was initially empty (already handled by the first text).
            # To avoid double text, check if it wasn't already handled.
            if current_agents_list : # Only add this text if agents existed but none are plottable
                 ax.text(world_size[0] / 2, world_size[1] / 2, "No living agents to display",
                    ha='center', va='center', fontsize=12, color='grey')
        else:
            xs = [ad['position'][0] for ad in agent_data_to_plot]
            ys = [ad['position'][1] for ad in agent_data_to_plot]
            agent_colors = [get_species_color_for_matplotlib(ad['species_id']) for ad in agent_data_to_plot]
            ax.scatter(xs, ys, c=agent_colors, s=25, alpha=0.7, edgecolors='black', linewidths=0.5)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_bytes_generated = buf.getvalue()

        with simulation_lock: 
            latest_arena_image_bytes = image_bytes_generated
        
        return image_bytes_generated
    finally:
        plt.close(fig) 


def simulation_runner_thread_target():
    global ecosystem, is_running, current_step, latest_chart_image_bytes, verbose_logging_enabled
    global population_size_setting, total_steps_setting, initial_environmental_pressure_setting
    global initial_resource_availability_setting, initial_environment_type_setting

    thread_local_console_buffer: List[str] = []

    def _flush_thread_logs():
        if not thread_local_console_buffer:
            return
        with simulation_lock:
            for msg_content in thread_local_console_buffer:
                log_to_gui(msg_content)
        thread_local_console_buffer.clear()

    def _local_log(msg_content: str, flush_now: bool = False):
        thread_local_console_buffer.append(msg_content)
        if flush_now or len(thread_local_console_buffer) >= 5:
            _flush_thread_logs()

    _local_log(
        f"SimRunner: Thread started. Pop Target:{population_size_setting}, "
        f"Steps Request:{total_steps_setting}, Pressure:{initial_environmental_pressure_setting:.2f}, "
        f"Resource:{initial_resource_availability_setting:.2f}, EnvType:{initial_environment_type_setting}, "
        f"VerboseLogs:{verbose_logging_enabled}",
        True
    )
    
    local_ecosystem_instance: Optional[AdvancedEvolutionEnvironment] = None
    seed_agents_for_new_sim: Optional[List[NeuralAgent]] = None

    try:
        np.random.seed() # Ensure thread-local randomness for simulation logic if it uses numpy.random globally

        if PERSISTENCE_PATH.exists():
            try:
                with PERSISTENCE_PATH.open("rb") as f:
                    loaded_state = pickle.load(f)
                if isinstance(loaded_state, AdvancedEvolutionEnvironment):
                    local_ecosystem_instance = loaded_state
                    max_existing_id = 0
                    if local_ecosystem_instance.agents:
                        agent_ids = [ag.id for ag in local_ecosystem_instance.agents]
                        lineage_ids = [lid for ag in local_ecosystem_instance.agents if ag.lineage for lid in ag.lineage]
                        all_ids = agent_ids + lineage_ids
                        if all_ids:
                            max_existing_id = max(all_ids)
                            
                    if hasattr(AdvancedEvolutionEnvironment, '_next_agent_id_counter'):
                        AdvancedEvolutionEnvironment._next_agent_id_counter = max(
                            getattr(AdvancedEvolutionEnvironment, '_next_agent_id_counter', 0),
                            max_existing_id + 1
                        )
                    # Migrate history lists from older pkl format
                    local_ecosystem_instance._migrate_histories()

                    # Detect extinct ecosystem and start fresh instead
                    if not local_ecosystem_instance.agents:
                        _local_log(
                            f"SimRunner: ⚠️ Loaded ecosystem from '{PERSISTENCE_FILENAME}' but it has 0 agents (extinct). Starting fresh.", True
                        )
                        local_ecosystem_instance = None
                    else:
                        _local_log(
                            f"SimRunner: ✅ Loaded ecosystem state from '{PERSISTENCE_FILENAME}'. "
                            f"Resuming from step {local_ecosystem_instance.time_step}. Population: {len(local_ecosystem_instance.agents)}", True
                        )
                elif isinstance(loaded_state, list) and all(isinstance(a, NeuralAgent) for a in loaded_state):
                    seed_agents_for_new_sim = loaded_state
                    max_seed_id = 0
                    if seed_agents_for_new_sim:
                        agent_ids = [ag.id for ag in seed_agents_for_new_sim]
                        lineage_ids = [lid for ag in seed_agents_for_new_sim if ag.lineage for lid in ag.lineage]
                        all_ids = agent_ids + lineage_ids
                        if all_ids:
                            max_seed_id = max(all_ids)

                    if hasattr(AdvancedEvolutionEnvironment, '_next_agent_id_counter'):
                         AdvancedEvolutionEnvironment._next_agent_id_counter = max(
                            getattr(AdvancedEvolutionEnvironment, '_next_agent_id_counter', 0),
                            max_seed_id + 1
                        )
                    if not seed_agents_for_new_sim:
                        _local_log(
                            f"SimRunner: ⚠️ Loaded agent list from '{PERSISTENCE_FILENAME}' but it is empty. Starting fresh.", True
                        )
                        seed_agents_for_new_sim = None
                    else:
                        _local_log(
                            f"SimRunner: ✅ Loaded {len(seed_agents_for_new_sim)} agents from '{PERSISTENCE_FILENAME}' as seed.", True
                        )
                else:
                    _local_log(
                        f"SimRunner: ⚠️ PKL file '{PERSISTENCE_FILENAME}' did not contain a valid ecosystem or agent list; starting fresh.", True
                    )
            except Exception as e_load:
                tb_str = traceback.format_exc()
                error_msg = (f"SimRunner: ⚠️ Error loading '{PERSISTENCE_FILENAME}': {e_load}. Starting fresh. "
                             f"This usually means the .pkl file is from an incompatible version of simulation_logic.py "
                             f"or the file is corrupted. Consider using the 'Reset Simulation' button. "
                             f"Detailed Traceback on server console.")
                _local_log(error_msg, True)
                print(f"SimRunner: DETAILED ERROR loading '{PERSISTENCE_FILENAME}':\n{tb_str}")
                # Ensure local_ecosystem_instance is None so a fresh one is created
                local_ecosystem_instance = None 
                seed_agents_for_new_sim = None
        else:
            _local_log(f"SimRunner: No '{PERSISTENCE_FILENAME}' found. Starting with a new population.", True)

        if not local_ecosystem_instance:
            local_ecosystem_instance = AdvancedEvolutionEnvironment(
                population_size=population_size_setting,
                seed_agents=seed_agents_for_new_sim # This will be None if starting completely fresh
            )
            if not seed_agents_for_new_sim : 
                local_ecosystem_instance.environmental_pressure = initial_environmental_pressure_setting
                local_ecosystem_instance.resource_availability = initial_resource_availability_setting
                local_ecosystem_instance.environment_type = initial_environment_type_setting
            _local_log(f"SimRunner: Ecosystem initialized/created. Pop: {len(local_ecosystem_instance.agents)}. Starting {total_steps_setting} simulation steps.", True)
        
        with simulation_lock:
            global ecosystem
            ecosystem = local_ecosystem_instance
            global current_step 
            current_step = ecosystem.time_step


        for step_iter in range(total_steps_setting):
            if stop_event.is_set():
                _local_log("SimRunner: Stop signal received. Terminating simulation loop.", True)
                break

            if not ecosystem.run_advanced_simulation_step(): 
                _local_log(f"💀 SimRunner: ECOSYSTEM EXTINCT at step {ecosystem.time_step}!", True)
                with simulation_lock: 
                    current_step = ecosystem.time_step
                break
            
            with simulation_lock:
                current_step = ecosystem.time_step

            generate_arena_frame()

            if verbose_logging_enabled and (
                current_step % 20 == 0 or current_step == 1 or step_iter == total_steps_setting -1
            ):
                stats_snapshot = ecosystem.get_advanced_ecosystem_stats()
                if stats_snapshot.get("status") == "running":
                    _local_log(
                        f"Step {current_step}: Pop={stats_snapshot['population']}, "
                        f"AvgFit={stats_snapshot['avg_fitness']:.2f}, "
                        f"MaxFit={stats_snapshot['max_fitness']:.2f}, "
                        f"Species={stats_snapshot['num_species']}"
                    )
                elif stats_snapshot.get("status") == "extinct": 
                    _local_log(f"Step {current_step}: Ecosystem extinct reported by stats.", True)
                    break
            
            if current_step % 10 == 0: # More frequent flush for responsiveness
                _flush_thread_logs()
        else: 
            _local_log(f"SimRunner: Simulation loop for {total_steps_setting} steps completed. Total steps in sim: {current_step}", True)


        if ecosystem.agents and len(ecosystem.avg_fitness_history) >= 2:
            _local_log("SimRunner: Generating visualization chart for this run...", True)
            ecosystem.visualize_advanced_evolution() 
            buf_chart = io.BytesIO()
            if plt.get_fignums(): 
                 plt.savefig(buf_chart, format='png')
                 plt.close('all') 
                 with simulation_lock:
                     latest_chart_image_bytes = buf_chart.getvalue()
                 _local_log("SimRunner: Visualization chart generated and stored.", True)
            else:
                 _local_log("SimRunner: No active figure for chart after visualize_advanced_evolution.", True)
        else:
            _local_log("SimRunner: Skipping visualization chart for this run (no agents/insufficient history).", True)

    except Exception as e_thread:
        tb_str_thread = traceback.format_exc()
        _local_log(f"SimRunner: CRITICAL THREAD EXCEPTION: {e_thread}\nTraceback on server console.", True)
        print(f"SimRunner: CRITICAL THREAD EXCEPTION:\n{tb_str_thread}")
    finally:
        current_ecosystem_to_save = None
        with simulation_lock: # Get the most current ecosystem reference safely
            current_ecosystem_to_save = ecosystem

        if current_ecosystem_to_save:
            agent_count = len(current_ecosystem_to_save.agents)
            _local_log(f"SimRunner: Attempting to save final ecosystem state (with {agent_count} agents)...", True)

            if agent_count == 0:
                _local_log("SimRunner: Population is extinct — skipping save to preserve previous elite state.", True)
            else:
                if ELITE_SAVE_COUNT > 0 and agent_count > ELITE_SAVE_COUNT:
                    sorted_final_agents = sorted(current_ecosystem_to_save.agents, key=lambda ag_save: ag_save.fitness, reverse=True)
                    current_ecosystem_to_save.agents = sorted_final_agents[:ELITE_SAVE_COUNT]
                    _local_log(f"SimRunner: Pruned agents to top {len(current_ecosystem_to_save.agents)} for saving.", True)

                    new_species_counts_save = defaultdict(int)
                    for ag_elite_save in current_ecosystem_to_save.agents:
                        new_species_counts_save[ag_elite_save.species_id] += 1
                    current_ecosystem_to_save.species_populations = new_species_counts_save
                    current_ecosystem_to_save.species_fitness = defaultdict(list)
                    current_ecosystem_to_save.species_traits_avg = defaultdict(lambda: defaultdict(float))

                try:
                    with PERSISTENCE_PATH.open("wb") as f_save:
                        pickle.dump(current_ecosystem_to_save, f_save)
                    _local_log(f"SimRunner: 💾 Saved ecosystem state to '{PERSISTENCE_FILENAME}'.", True)
                except Exception as e_save_final:
                    _local_log(f"SimRunner: ⚠️ Failed to save ecosystem state in finally block: {e_save_final}", True)
                    print(f"SimRunner: FAILED TO SAVE in finally: {e_save_final}")
        else:
            _local_log("SimRunner: No ecosystem instance to save in finally block (was None).", True)

        _flush_thread_logs()
        with simulation_lock:
            global is_running
            is_running = False
        log_to_gui("SimRunner: Thread execution concluded.")


@app.route("/")
def index_route():
    return render_template("index.html")


@app.route("/start_evolution", methods=["POST"])
def start_evolution_route():
    global simulation_thread, is_running, current_step, total_steps_setting, ecosystem
    global population_size_setting, initial_environmental_pressure_setting, stop_event
    global latest_chart_image_bytes, gui_console_messages
    global initial_resource_availability_setting, initial_environment_type_setting, verbose_logging_enabled

    with simulation_lock:
        if is_running:
            log_to_gui("FlaskRoute: Simulation is already running. Stopping it first...")
            stop_event.set()
            if simulation_thread and simulation_thread.is_alive():
                simulation_thread.join(timeout=5.0) 
            if simulation_thread and simulation_thread.is_alive():
                log_to_gui("FlaskRoute: Warning! Previous simulation did not exit cleanly after stop signal during restart attempt.")
            is_running = False 

        try:
            request_data = request.get_json()
            if not request_data:
                log_to_gui("FlaskRoute: Error! No JSON data received for start_evolution.")
                return jsonify({"error": "Request requires JSON data"}), 400

            population_size_setting = int(request_data.get("populationSize", 50))
            total_steps_setting = int(request_data.get("evolutionSteps", 1000)) 
            initial_environmental_pressure_setting = float(request_data.get("initialPressure", 0.3))
            initial_resource_availability_setting = float(request_data.get("initialResources", 1.0))
            initial_environment_type_setting = str(request_data.get("initialEnvType", "balanced"))
            verbose_logging_enabled = bool(request_data.get("verbose", True))

        except (ValueError, TypeError) as e_params:
            log_to_gui(f"FlaskRoute: Error parsing parameters: {e_params}")
            return jsonify({"error": f"Invalid parameter format: {e_params}"}), 400
        
        # ecosystem global variable is intentionally not reset to None here.
        # The simulation_runner_thread_target will load from PKL or create new if needed.
        # current_step will also be set by the thread based on loaded/new ecosystem.

        stop_event.clear()
        is_running = True 

        log_to_gui("FlaskRoute: Creating and starting simulation thread...")
        simulation_thread = threading.Thread(target=simulation_runner_thread_target, daemon=True)
        simulation_thread.start()
        log_to_gui(
            f"FlaskRoute: Simulation thread initiated. Target Pop (if new):{population_size_setting}, "
            f"Steps for this run:{total_steps_setting}, InitPressure (if new):{initial_environmental_pressure_setting:.2f}, "
            f"InitResources (if new):{initial_resource_availability_setting:.2f}, InitEnvType (if new):{initial_environment_type_setting}, "
            f"Verbose:{verbose_logging_enabled}"
        )

    return jsonify({"message": "Evolution process initiated."})


@app.route("/pause_evolution", methods=["POST"])
def pause_evolution_route_handler():
    log_to_gui("FlaskRoute: Pause command received (conceptual: use Stop/Reset and Start to resume from saved state).")
    return jsonify({"message": "Pause is conceptual. Simulation continues unless reset or completed."})


@app.route("/reset_evolution", methods=["POST"])
def reset_evolution_route_handler():
    global ecosystem, simulation_thread, is_running, current_step, stop_event
    global latest_chart_image_bytes, gui_console_messages, latest_arena_image_bytes

    with simulation_lock:
        log_to_gui("FlaskRoute: Reset command received; stopping active simulation if any...")
        if is_running and simulation_thread and simulation_thread.is_alive():
            stop_event.set()
            simulation_thread.join(timeout=5.0) 
            if simulation_thread and simulation_thread.is_alive():
                log_to_gui("FlaskRoute: Warning! Active simulation did not stop cleanly during reset.")
        
        if PERSISTENCE_PATH.exists():
            try:
                PERSISTENCE_PATH.unlink()
                log_to_gui(f"FlaskRoute: Persistence file '{PERSISTENCE_FILENAME}' deleted for full reset.")
            except Exception as e_del_reset:
                log_to_gui(f"FlaskRoute: Warning! Could not delete persistence file '{PERSISTENCE_FILENAME}': {e_del_reset}")

        ecosystem = None
        simulation_thread = None 
        is_running = False
        current_step = 0
        latest_chart_image_bytes = None
        latest_arena_image_bytes = None
        gui_console_messages.clear() 
        stop_event.clear() 
        log_to_gui("FlaskRoute: Simulation environment, state, and persistence file have been reset.")

    return jsonify({"message": "Evolution simulation reset and persistence cleared."})


@app.route("/status")
def get_status_route_handler():
    with simulation_lock: 
        sim_stats_payload: Dict[str, Any] = {} # Renamed to avoid conflict with local sim_stats
        display_status_text_payload: str # Renamed
        console_messages_list_payload: List[str] = list(gui_console_messages) 

        local_ecosystem_instance_status_ref = ecosystem 
        
        if local_ecosystem_instance_status_ref:
            sim_stats_payload = local_ecosystem_instance_status_ref.get_advanced_ecosystem_stats()
            effective_sim_status_payload = sim_stats_payload.get("status", "unknown_sim_status")
        else:
            effective_sim_status_payload = "idle_no_ecosystem" 

        if is_running:
            if local_ecosystem_instance_status_ref and getattr(local_ecosystem_instance_status_ref, 'agents', None): # Check .agents attribute exists
                if effective_sim_status_payload == "running":
                     display_status_text_payload = "running_active_simulation"
                else: 
                     display_status_text_payload = "initializing_or_running_core"
            else: 
                display_status_text_payload = "initializing_simulation_thread" 
        else: 
            if current_step > 0: 
                if effective_sim_status_payload == "extinct":
                    display_status_text_payload = "extinct_previous_run"
                elif local_ecosystem_instance_status_ref: 
                    display_status_text_payload = "finished_completed_run"
                else: 
                    display_status_text_payload = "completed_or_reset"
            else: 
                display_status_text_payload = "idle_ready_to_start"
        
        if effective_sim_status_payload == "extinct": # Override if logic reports extinct
            display_status_text_payload = "extinct_reported_by_logic"


        avg_accuracy_value_payload = 0.0
        if local_ecosystem_instance_status_ref and getattr(local_ecosystem_instance_status_ref, 'agents', None) and local_ecosystem_instance_status_ref.agents:
            all_accuracies_payload: List[float] = []
            for agent_obj_status_payload in local_ecosystem_instance_status_ref.agents:
                if agent_obj_status_payload.classification_accuracy: 
                    all_accuracies_payload.extend(list(agent_obj_status_payload.classification_accuracy))
            if all_accuracies_payload:
                avg_accuracy_value_payload = float(np.mean(all_accuracies_payload))

        payload_response = { # Renamed
            "isRunning": is_running,
            "currentStep": current_step, 
            "totalSteps": total_steps_setting, 
            "population": int(sim_stats_payload.get("population", 0)),
            "avg_fitness": float(sim_stats_payload.get("avg_fitness", 0.0)),
            "max_fitness": float(sim_stats_payload.get("max_fitness", 0.0)),
            "avg_accuracy": avg_accuracy_value_payload,
            "num_species": int(sim_stats_payload.get("num_species", 0)),
            "avg_age": float(sim_stats_payload.get("avg_age", 0.0)),
            "avg_cooperation": float(sim_stats_payload.get("avg_cooperation", 0.0)),
            "avg_aggression": float(sim_stats_payload.get("avg_aggression", 0.0)),
            "avg_stress": float(sim_stats_payload.get("avg_stress", 0.0)),
            "speciations_total": int(sim_stats_payload.get("speciations_total", 0)),
            "extinctions_total": int(sim_stats_payload.get("extinctions_total", 0)),
            "ecosystemStatusText": display_status_text_payload,
            "isExtinct": "extinct" in display_status_text_payload,
            "consoleMessages": console_messages_list_payload,
            "environmentType": str(sim_stats_payload.get("environment_type", "N/A")),
            "environmentalPressure": float(sim_stats_payload.get("environmental_pressure", 0.0)),
            "resourceAvailability": float(sim_stats_payload.get("resource_availability", 0.0)),
            "species_details": sim_stats_payload.get("species_details", {})
        }
    return jsonify(payload_response)


@app.route("/arena_data")
def arena_data_route_handler():
    with simulation_lock:
        local_eco_ref_arena = ecosystem
        if not local_eco_ref_arena or not getattr(local_eco_ref_arena, "agents", None) or not local_eco_ref_arena.agents:
            return jsonify({"world_size": [100, 100], "max_fitness": 1.0, "agents": []})
        # Snapshot under lock
        agents_snapshot_arena = list(local_eco_ref_arena.agents)
        ws_arena = local_eco_ref_arena.world_size

    # Build response outside lock
    max_fit_arena = 1.0
    agents_list_arena = []
    for agent_obj_arena_data in agents_snapshot_arena:
        if agent_obj_arena_data.energy <= 0:
            continue
        fit = float(agent_obj_arena_data.fitness)
        if fit > max_fit_arena:
            max_fit_arena = fit
        agents_list_arena.append({
            "x": float(agent_obj_arena_data.position[0]),
            "y": float(agent_obj_arena_data.position[1]),
            "species": str(agent_obj_arena_data.species_id),
            "fitness": fit,
            "generation": int(agent_obj_arena_data.generation),
        })

    return jsonify({
        "world_size": [float(ws_arena[0]), float(ws_arena[1])],
        "max_fitness": max_fit_arena,
        "agents": agents_list_arena
    })


@app.route("/chart")
def get_chart_route_handler():
    global latest_chart_image_bytes
    with simulation_lock:
        snapshot_bytes_chart = latest_chart_image_bytes 

    if snapshot_bytes_chart:
        return Response(snapshot_bytes_chart, mimetype="image/png")
    else:
        placeholder_path_chart = STATIC_DIR / "placeholder_chart.png"
        if placeholder_path_chart.exists():
            return send_file(placeholder_path_chart, mimetype="image/png")
        return "Chart image is not available and no placeholder found.", 404


@app.route("/arena")
def get_arena_route_handler():
    return render_template("arena.html")


@app.route("/arena_frame")
def get_arena_frame_route_handler():
    image_bytes_arena = generate_arena_frame() 
    if image_bytes_arena:
        return Response(image_bytes_arena, mimetype="image/png")
    else:
        placeholder_path_arena_frame = STATIC_DIR / "placeholder_arena.png"
        if placeholder_path_arena_frame.exists():
            return send_file(placeholder_path_arena_frame, mimetype="image/png")
        return ("", 204) 


@app.route("/export_data")
def export_data_route_handler():
    global ecosystem
    with simulation_lock:
        local_eco_ref_export = ecosystem 
        if local_eco_ref_export:
            data_to_export_payload = local_eco_ref_export.get_advanced_ecosystem_stats() 
            
            history_data_payload_export = {
                "population_history": list(local_eco_ref_export.population_history),
                "avg_fitness_history": list(local_eco_ref_export.avg_fitness_history),
                "max_fitness_history": list(local_eco_ref_export.max_fitness_history),
                "num_species_history": list(local_eco_ref_export.diversity_history),
                "avg_age_history": list(local_eco_ref_export.avg_age_history),
                "avg_cooperation_history": list(local_eco_ref_export.avg_cooperation_tendency_history),
                "avg_aggression_history": list(local_eco_ref_export.avg_aggression_history),
                "avg_stress_history": list(local_eco_ref_export.avg_stress_history),
                "avg_specialization_history": list(local_eco_ref_export.avg_specialization_history),
                "resource_history": list(local_eco_ref_export.resource_history),
                "pressure_history": list(local_eco_ref_export.pressure_history),
                "extinction_events_log": list(local_eco_ref_export.extinction_events),
                "speciation_events_log": list(local_eco_ref_export.speciation_events)
            }
            data_to_export_payload["history_data"] = history_data_payload_export


            def convert_numpy_types_for_json_export(obj_export: Any) -> Any: # Renamed function parameter
                if isinstance(obj_export, np.integer):
                    return int(obj_export)
                if isinstance(obj_export, np.floating):
                    return float(obj_export)
                if isinstance(obj_export, (np.bool_, bool)): 
                    return bool(obj_export)
                if isinstance(obj_export, np.ndarray):
                    return obj_export.tolist()
                if isinstance(obj_export, dict):
                    return {k_export: convert_numpy_types_for_json_export(v_export) for k_export, v_export in obj_export.items()}
                if isinstance(obj_export, list):
                    return [convert_numpy_types_for_json_export(i_export) for i_export in obj_export]
                if isinstance(obj_export, deque): 
                    return list(obj_export)
                return obj_export

            try:
                json_ready_data_export = convert_numpy_types_for_json_export(data_to_export_payload)
                return jsonify(json_ready_data_export)
            except TypeError as e_json_export:
                log_to_gui(f"JSON Conversion Error for export: {e_json_export}. Data might be incomplete.")
                return jsonify({"error": f"Could not serialize data for JSON: {e_json_export}"}), 500

    return jsonify({"error": "No ecosystem data available for export."}), 404


@app.route("/top_performers")
def top_performers_route_handler():
    global ecosystem
    with simulation_lock:
        local_eco_ref_top = ecosystem 
        if local_eco_ref_top and local_eco_ref_top.agents:
            top_agents_info_list_payload: List[Dict[str, Any]] = [] 
            top_list_agents_payload = local_eco_ref_top.get_top_performers(n=5, sort_key="fitness")
            for agent_instance_top_payload in top_list_agents_payload: 
                accuracy_list_current_top_payload = list(agent_instance_top_payload.classification_accuracy)
                avg_acc_current_top_payload = float(np.mean(accuracy_list_current_top_payload)) if accuracy_list_current_top_payload else 0.0
                genome_data_dict_top_payload = {
                    "layers": agent_instance_top_payload.genome.layers,
                    "learning_rate": round(float(agent_instance_top_payload.genome.learning_rate), 5),
                    "aggression": round(float(agent_instance_top_payload.genome.aggression), 3),
                    "cooperation_tendency": round(float(agent_instance_top_payload.genome.cooperation_tendency), 3),
                    "stress_resistance": round(float(agent_instance_top_payload.genome.stress_resistance), 3),
                    "mutation_rate": round(float(agent_instance_top_payload.genome.mutation_rate), 3),
                    "exploration_rate": round(float(agent_instance_top_payload.genome.exploration_rate), 3),
                    "memory_capacity": int(agent_instance_top_payload.genome.memory_capacity),
                }
                agent_info_dict_top_payload = {
                    "id": agent_instance_top_payload.id,
                    "species_id": agent_instance_top_payload.species_id,
                    "generation": agent_instance_top_payload.generation,
                    "fitness": round(float(agent_instance_top_payload.fitness), 2),
                    "energy": round(float(agent_instance_top_payload.energy), 1),
                    "age": agent_instance_top_payload.age,
                    "offspring_count": agent_instance_top_payload.offspring_count,
                    "reputation": round(float(agent_instance_top_payload.reputation), 2),
                    "accuracy_avg": round(avg_acc_current_top_payload, 3),
                    "stress_level": round(float(agent_instance_top_payload.stress_level), 3),
                    "genome": genome_data_dict_top_payload
                }
                top_agents_info_list_payload.append(agent_info_dict_top_payload)
            return jsonify(top_agents_info_list_payload)

    return jsonify({"error": "No agents or ecosystem not initialized."}), 404


@app.route("/step_evolution", methods=["POST"])
def step_evolution_route_handler():
    global ecosystem, current_step, latest_chart_image_bytes, latest_arena_image_bytes

    with simulation_lock:
        if is_running:
            return jsonify({"error": "Cannot step manually while the simulation is running."}), 400
        if not ecosystem:
            return jsonify({"error": "No ecosystem available. Start or reset the simulation first."}), 400
        
        step_success_manual = ecosystem.run_advanced_simulation_step() # Renamed
        current_step = ecosystem.time_step 
        
        if not step_success_manual:
            log_to_gui(f"FlaskRoute: Manual step to {current_step} resulted in extinction or failure.")
            generate_arena_frame() 
            final_stats_after_manual_step = ecosystem.get_advanced_ecosystem_stats() # Renamed
            return jsonify({
                "error": f"Ecosystem extinct or cannot run step. At step {current_step}.",
                "population": final_stats_after_manual_step.get("population",0),
                "step": current_step
                }), 400 
            
        log_to_gui(f"FlaskRoute: Manual step executed. Now at step {current_step}.")

        if ecosystem.agents and len(ecosystem.avg_fitness_history) >= 2:
            ecosystem.visualize_advanced_evolution() 
            buf_manual_step = io.BytesIO() # Renamed
            if plt.get_fignums(): 
                 plt.savefig(buf_manual_step, format='png')
                 plt.close('all')
                 latest_chart_image_bytes = buf_manual_step.getvalue()
        
        generate_arena_frame()
        current_stats_after_manual_step = ecosystem.get_advanced_ecosystem_stats() # Renamed
        return jsonify({
            "message": f"Advanced one step to {current_step}.",
            "population": current_stats_after_manual_step.get("population", 0),
            "step": current_step
        })


@app.route("/set_environment", methods=["POST"])
def set_environment_route_handler():
    global ecosystem 
    with simulation_lock:
        if not ecosystem:
            return jsonify({"error": "No ecosystem available. Start the simulation first."}), 400
        
        request_data_env = request.get_json() # Renamed
        if not request_data_env:
            return jsonify({"error": "Request requires JSON data"}), 400

        env_type_req = request_data_env.get("environmentType") # Renamed
        pressure_req = request_data_env.get("pressure") # Renamed
        resources_req = request_data_env.get("resources") # Renamed

        try:
            env_type_to_set = str(env_type_req) if env_type_req is not None else ecosystem.environment_type # Renamed
            pressure_to_set = float(pressure_req) if pressure_req is not None else ecosystem.environmental_pressure # Renamed
            resources_to_set = float(resources_req) if resources_req is not None else ecosystem.resource_availability # Renamed

            ecosystem.set_environment_conditions(env_type_to_set, pressure_to_set, resources_to_set)
            log_to_gui(
                f"FlaskRoute: Environment updated to Type:{ecosystem.environment_type}, "
                f"Pressure:{ecosystem.environmental_pressure:.2f}, Resources:{ecosystem.resource_availability:.2f}."
            )
            return jsonify({
                "message": "Environment conditions updated.",
                "environmentType": ecosystem.environment_type,
                "environmentalPressure": ecosystem.environmental_pressure,
                "resourceAvailability": ecosystem.resource_availability
            })
        except (ValueError, TypeError) as e_env_param_set: # Renamed
            log_to_gui(f"FlaskRoute: Error parsing environment parameters: {e_env_param_set}")
            return jsonify({"error": f"Invalid environment parameter format: {e_env_param_set}"}), 400


@app.route("/toggle_verbose", methods=["POST"])
def toggle_verbose_route_handler():
    global verbose_logging_enabled
    with simulation_lock: 
        verbose_logging_enabled = not verbose_logging_enabled
        log_to_gui(f"FlaskRoute: Verbose logging set to {verbose_logging_enabled}.")
        return jsonify({"message": f"Verbose logging {'enabled' if verbose_logging_enabled else 'disabled'}."})


@app.route("/download_elite", methods=["GET"])
def download_elite_route_handler():
    if PERSISTENCE_PATH.exists():
        return send_file(
            PERSISTENCE_PATH,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name=PERSISTENCE_FILENAME 
        )
    return jsonify({"error": "No ecosystem state file available to download."}), 404


@app.route("/upload_seed", methods=["POST"])
def upload_seed_route_handler():
    global ecosystem, is_running, simulation_lock 

    with simulation_lock: 
        if is_running:
            return jsonify({"error": "Simulation is running. Please stop or reset before uploading a new seed/ecosystem."}), 400
        
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400
        uploaded_file = request.files['file'] # Renamed
        if uploaded_file.filename == '':
            return jsonify({"error": "No selected file."}), 400
        
        try:
            uploaded_data = uploaded_file.read()
            # Use RestrictedUnpickler to prevent arbitrary code execution
            loaded_state_upload = RestrictedUnpickler(io.BytesIO(uploaded_data)).load()

            if isinstance(loaded_state_upload, AdvancedEvolutionEnvironment) or \
               (isinstance(loaded_state_upload, list) and all(isinstance(a_upload, NeuralAgent) for a_upload in loaded_state_upload)):

                with PERSISTENCE_PATH.open("wb") as f_upload:
                    f_upload.write(uploaded_data) 
                
                if isinstance(loaded_state_upload, AdvancedEvolutionEnvironment):
                    ecosystem = loaded_state_upload 
                    log_to_gui(f"FlaskRoute: Full ecosystem state uploaded and replaced '{PERSISTENCE_FILENAME}'. Will be used on next start.")
                else: 
                    ecosystem = None 
                    log_to_gui(f"FlaskRoute: Agent list seed uploaded and replaced '{PERSISTENCE_FILENAME}'. Will be used to seed new ecosystem on next start.")
                
                return jsonify({"message": "File uploaded and saved as persistence seed successfully."})
            else:
                return jsonify({"error": "Uploaded file is not a valid ecosystem or agent list pickle."}), 400
        except pickle.UnpicklingError as e_security:
            log_to_gui(f"FlaskRoute: Upload blocked for security: {e_security}")
            return jsonify({"error": f"Upload rejected: {e_security}. Only .pkl files from this application are accepted."}), 400
        except Exception as e_upload_seed:
            log_to_gui(f"FlaskRoute: Could not process uploaded file: {e_upload_seed}")
            return jsonify({"error": f"Could not process uploaded file: {e_upload_seed}"}), 400


if __name__ == "__main__":
    STATIC_DIR.mkdir(exist_ok=True)
    TEMPLATE_DIR.mkdir(exist_ok=True)

    host = os.environ.get("NEURAL_EVOLUTION_HOST", "127.0.0.1")
    port = int(os.environ.get("NEURAL_EVOLUTION_PORT", "5000"))
    debug = os.environ.get("NEURAL_EVOLUTION_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    print("Starting Neural Evolution Flask Application...")
    print(f"Project directory: {PROJECT_ROOT}")
    print(f"Templates directory: {TEMPLATE_DIR}")
    print(f"Static directory: {STATIC_DIR}")
    print(f"Persistence file will be: {PERSISTENCE_PATH}")
    print(f"Access the UI at http://127.0.0.1:{port}")
    app.run(debug=debug, threaded=True, host=host, port=port)
