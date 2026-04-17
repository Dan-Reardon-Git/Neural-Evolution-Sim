import pickle
import numpy as np
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Note: removed seaborn import to keep visualization strictly matplotlib (avoids extra dependencies).


# --- Data Classes for Genome and Memory ---

@dataclass
class Genome:
    """Enhanced DNA of our neural agents—defines structure, learning, and behavioral traits."""
    layers: List[int]               # e.g. [input_dim, hidden1, ..., output_dim]
    activation_mutations: List[str] # activation function for each layer (except input)
    learning_rate: float            # base step size for gradient-based updates
    aggression: float               # how strongly they compete
    reproduction_threshold: float   # energy needed to reproduce
    mutation_rate: float            # probability of mutating at reproduction
    memory_capacity: int            # how many past experiences to store
    crossover_rate: float           # probability of crossover when mating

    # Enhanced traits
    cooperation_tendency: float = 0.5   # base likelihood to help other agents
    risk_tolerance: float = 0.5         # willingness to take risky actions
    exploration_rate: float = 0.       # tendency to add noise/explore
    social_learning_rate: float = 0.5   # ability to learn from peers
    metabolic_efficiency: float = 1.0   # energy consumption multiplier (lower = more efficient)
    stress_resistance: float = 0.1      # ability to recover from stress

    # Behavioral traits
    territorial_radius: float = 5.0     # radius for social/competitive interactions
    mating_selectivity: float = 0.5     # higher = more choosy
    parental_investment: float = 0.5    # energy fraction invested in offspring


@dataclass
class AgentMemory:
    """Enhanced memory system for each agent."""
    experiences: deque = field(default_factory=lambda: deque(maxlen=100))  
        # stores (input_vector, true_label, reward)
    social_interactions: Dict[int, List[Tuple[str, float]]] = field(default_factory=lambda: defaultdict(list))
        # mapping: peer_id -> [(event_type, value)]
    successful_strategies: deque = field(default_factory=lambda: deque(maxlen=20))  
        # stores (strategy_repr, success_metric)
    environmental_patterns: Dict[str, float] = field(default_factory=dict)  
        # hashed patterns -> frequency or importance
    threat_assessments: Dict[int, float] = field(default_factory=lambda: defaultdict(float))  
        # mapping: other_agent_id -> threat_level (0..1)


# --- NeuralAgent Class ---

class NeuralAgent:
    """An enhanced self-modifying neural agent with complex behaviors."""

    def __init__(
        self,
        agent_id: int,
        genome: Optional[Genome] = None,
        position: Optional[Tuple[float, float]] = None
    ):
        self.id = agent_id
        self.genome = genome or self._random_genome()

        # Enforce correct input/output dimensions: input_dim=20, output_dim=3
        if self.genome.layers[0] != 20 or self.genome.layers[-1] != 3:
            raise ValueError(
                f"Agent {self.id} initialized with invalid genome.layers: {self.genome.layers} (must start at 20, end at 3)."
            )
        if len(self.genome.activation_mutations) != len(self.genome.layers) - 1:
            raise ValueError(
                f"Agent {self.id} has {len(self.genome.activation_mutations)} activation_mutations, "
                f"but expected {len(self.genome.layers)-1}."
            )

        # Core state
        self.energy = 250.0
        self.age = 0
        self.fitness = 0.0            # cumulative sum of rewards
        self.offspring_count = 0
        self.position = position or (
            random.uniform(0, 100),
            random.uniform(0, 100)
        )

        # Extended state
        self.stress_level = 0.0       # [0..1]
        self.social_bonds: Dict[int, float] = defaultdict(float)  
            # mapping: other_agent_id -> bond_strength [0..1]
        self.reputation = 0.0         # aggregate social standing
        self.specialization_score = 0.0 # measure: generalist (-1) ↔ specialist (+1)
        self.adaptation_history: deque = deque(maxlen=100)

        # Memory system
        self.memory = AgentMemory(
            experiences=deque(maxlen=self.genome.memory_capacity)
        )

        # Neural network parameters
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._stored_preactivations: List[np.ndarray] = []  # store z-values for backprop
        self._build_network()

        # Store original genome traits (before stress adaptation modifies them)
        # so offspring inherit the genetic values, not stress-modified phenotype
        self._original_genome_traits = {
            'risk_tolerance': self.genome.risk_tolerance,
            'exploration_rate': self.genome.exploration_rate,
            'aggression': self.genome.aggression,
            'cooperation_tendency': self.genome.cooperation_tendency,
        }

        # Evolution tracking
        self.generation = 0
        self.lineage = [agent_id]
        self.species_id = self._calculate_species_id()

        # Performance metrics
        self.classification_accuracy: deque = deque(maxlen=50)
        self.survival_challenges_faced = 0
        self.survival_challenges_overcome = 0
        self.cooperative_acts = 0
        self.competitive_wins = 0

    def _random_genome(self) -> Genome:
        """Generate a scientifically-plausible random genome."""
        hidden_layer_count = random.randint(1, 3)
        hidden_sizes = [random.randint(8, 20) for _ in range(hidden_layer_count)]
        layers = [20] + hidden_sizes + [3]

        activation_mutations = []
        for i in range(len(layers) - 1):
            if i == len(layers) - 2:
                activation_mutations.append('softmax')
            else:
                activation_mutations.append(
                    random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu', 'swish'])
                )

        return Genome(
            layers=layers,
            activation_mutations=activation_mutations,
            learning_rate=random.uniform(0.002, 0.09),
            aggression=random.uniform(0.0, 1.9),
            reproduction_threshold=random.uniform(90, 220),
            mutation_rate=random.uniform(0.04, 0.35),
            memory_capacity=random.randint(30, 70),
            crossover_rate=random.uniform(0.4, 0.95),

            cooperation_tendency=random.uniform(0.1, 0.95),
            risk_tolerance=random.uniform(0.1, 0.9),
            exploration_rate=random.uniform(0.1, 0.7),
            social_learning_rate=random.uniform(0.05, 0.3),
            metabolic_efficiency=random.uniform(0.7, 1.3),
            stress_resistance=random.uniform(0.3, 0.9),

            territorial_radius=random.uniform(2.0, 15.0),
            mating_selectivity=random.uniform(0.1, 0.9),
            parental_investment=random.uniform(0.05, 0.6)
        )

    def _calculate_species_id(self) -> int:
        """Generate a (pseudo-stable) species ID by hashing key genome traits."""
        arch_tuple = tuple(self.genome.layers)
        if len(self.genome.activation_mutations) > 1:
            hidden_activations = self.genome.activation_mutations[:-1]
            dominant_hidden = max(set(hidden_activations), key=hidden_activations.count)
        else:
            dominant_hidden = 'none'

        trait_tuple = (
            round(self.genome.aggression, 2),
            round(self.genome.cooperation_tendency, 2),
            round(self.genome.exploration_rate, 2),
            dominant_hidden
        )
        # XOR of two hashes, then mod to limit species IDs
        return abs(hash(arch_tuple) ^ hash(trait_tuple)) % 1000

    def _build_network(self):
        """Initialize weights/biases using appropriate variance scaling."""
        # Double-check validity
        if self.genome.layers[0] != 20 or self.genome.layers[-1] != 3:
            raise ValueError(
                f"CRITICAL: Agent {self.id} has invalid genome.layers {self.genome.layers}."
            )
        if len(self.genome.activation_mutations) != len(self.genome.layers) - 1:
            raise ValueError(
                f"CRITICAL: Agent {self.id} has mismatched activations {len(self.genome.activation_mutations)} vs layers {len(self.genome.layers)}."
            )

        self.weights.clear()
        self.biases.clear()

        for i in range(len(self.genome.layers) - 1):
            in_dim = self.genome.layers[i]
            out_dim = self.genome.layers[i + 1]
            func = self.genome.activation_mutations[i]

            # He/Xavier initialization based on activation
            if func in ('relu', 'leaky_relu', 'swish'):
                std = np.sqrt(2.0 / in_dim)
            elif func in ('tanh', 'sigmoid'):
                std = np.sqrt(1.0 / in_dim)
            else:
                # softmax or linear
                std = np.sqrt(2.0 / (in_dim + out_dim))

            w = np.random.normal(0.0, std, size=(in_dim, out_dim))
            b = np.zeros((out_dim,))
            self.weights.append(w)
            self.biases.append(b)

    def _activate(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """Apply activation function for layer_idx, given preactivation x."""
        func = self.genome.activation_mutations[layer_idx]
        if func == 'relu':
            return np.maximum(0, x)
        elif func == 'leaky_relu':
            return np.where(x > 0, x, x * 0.01)
        elif func == 'swish':
            # swish(x) = x * sigmoid(x)
            z = np.clip(x, -700, 700)
            sig = 1.0 / (1.0 + np.exp(-z))
            return x * sig
        elif func == 'tanh':
            return np.tanh(x)
        elif func == 'sigmoid':
            z = np.clip(x, -700, 700)
            return 1.0 / (1.0 + np.exp(-z))
        elif func == 'softmax':
            e = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e / np.sum(e, axis=-1, keepdims=True)
        else:
            # fallback to linear
            return x

    def think(self, input_data: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """Forward-pass with optional Gaussian noise (exploration). Returns final output (shape: (1,3))."""
        x_act = input_data.reshape(1, -1).astype(np.float64)  # ensure 2D
        self._stored_preactivations = []

        # Inject noise for exploration
        if add_noise and random.random() < self.genome.exploration_rate:
            noise_scale = 0.01 + 0.05 * self.genome.risk_tolerance
            x_act += np.random.normal(0.0, noise_scale, size=x_act.shape)

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(x_act, w) + b  # preactivation
            self._stored_preactivations.append(z)
            x_act = self._activate(z, i)
        return x_act

    def learn_from_environment(self, data: np.ndarray, true_label: int, reward: float):
        """Update memory, perform a delta-rule backprop (if |reward|>threshold), and always update fitness/energy."""
        if data is None:
            return

        # 1) Store experience
        self.memory.experiences.append((data.copy(), true_label, reward))

        # 2) If reward magnitude is substantial, attempt a backprop update
        if abs(reward) > 0.1:
            # (a) Recompute forward pass without noise to populate preactivations
            output_probs = self.think(data, add_noise=False)  # shape: (1,3)
            pred_label = int(np.argmax(output_probs[0]))
            learning_signal = np.clip(reward / 10.0, -1.0, 1.0)

            # (b) Build activation list: a⁰ = data, a¹ through aᴸ from stored preactivations
            activations_for_learning = [data.reshape(1, -1)]
            for layer_idx in range(len(self.weights)):
                z_val = self._stored_preactivations[layer_idx]
                a_val = self._activate(z_val, layer_idx)
                activations_for_learning.append(a_val)

            # (c) Initialize delta at output layer
            delta_L = np.zeros_like(output_probs)  # shape: (1,3)

            # Positive learning: correct classification & positive reward
            if (pred_label == true_label) and (learning_signal > 0):
                target = np.zeros_like(output_probs)
                target[0, true_label] = 1.0
                delta_L = (target - output_probs) * (learning_signal * self.genome.learning_rate)

            # Negative learning: incorrect classification & negative reward
            elif (pred_label != true_label) and (learning_signal < 0):
                target = np.zeros_like(output_probs)
                target[0, true_label] = 1.0
                delta_L = (target - output_probs) * (abs(learning_signal) * self.genome.learning_rate * 0.5)

            # Otherwise: reward does not align with correct/incorrect → skip weight update
            else:
                delta_L = None

            # (d) If we computed a valid delta, backpropagate through all layers
            if delta_L is not None:
                # Output‐layer grads
                grad_w_L = np.dot(activations_for_learning[-2].T, delta_L)  # shape: (hidden_size, 3)
                grad_b_L = np.sum(delta_L, axis=0)  # shape: (3,)

                self.weights[-1] += grad_w_L
                self.biases[-1] += grad_b_L

                current_delta = delta_L.copy()  # shape: (1, output_dim)

                # Backprop to hidden layers
                for l_idx in reversed(range(len(self.weights) - 1)):
                    z_l_plus_1 = self._stored_preactivations[l_idx]
                    a_l_plus_1 = activations_for_learning[l_idx + 1]
                    act_name = self.genome.activation_mutations[l_idx]

                    # Derivative of activation w.r.t. z
                    if act_name == 'relu':
                        deriv = (z_l_plus_1 > 0).astype(float)
                    elif act_name == 'leaky_relu':
                        deriv = np.where(z_l_plus_1 > 0, 1.0, 0.01)
                    elif act_name == 'tanh':
                        deriv = 1.0 - np.tanh(z_l_plus_1) ** 2
                    elif act_name == 'sigmoid':
                        sig_z = 1.0 / (1.0 + np.exp(-np.clip(z_l_plus_1, -500, 500)))
                        deriv = sig_z * (1.0 - sig_z)
                    elif act_name == 'swish':
                        sig_z = 1.0 / (1.0 + np.exp(-np.clip(z_l_plus_1, -500, 500)))
                        deriv = sig_z + z_l_plus_1 * sig_z * (1 - sig_z)
                    else:
                        # linear or softmax derivative unused here; approximate as 1.0
                        deriv = np.ones_like(a_l_plus_1)

                    # Propagate error: δ⁽ˡ⁾ = (δ⁽ˡ⁺¹⁾ ⋅ W⁽ˡ⁺¹⁾ᵀ) * σ'(z⁽ˡ⁺¹⁾)
                    error_to_prev = np.dot(current_delta, self.weights[l_idx + 1].T)
                    delta_z = error_to_prev * deriv  # shape: (1, layer_size)

                    # Gradients for W⁽ˡ⁾ and b⁽ˡ⁾
                    grad_w_hidden = np.dot(activations_for_learning[l_idx].T, delta_z)
                    grad_b_hidden = np.sum(delta_z, axis=0)

                    # Update
                    lr_factor = self.genome.learning_rate * 0.6
                    self.weights[l_idx] += grad_w_hidden * lr_factor
                    self.biases[l_idx] += grad_b_hidden * lr_factor

                    current_delta = delta_z.copy()

        # 3) Always update fitness and energy, regardless of whether we backprop'd above
        self.fitness += reward
        energy_delta = reward * self.genome.metabolic_efficiency * 0.6
        self.energy += np.clip(energy_delta, -15.0, 15.0)

    def compete_with(self, other: 'NeuralAgent', environment_data: Tuple[np.ndarray, int]) -> bool:
        """Compete on a shared input; winner steals energy. Returns True if self wins."""
        data, true_label = environment_data

        my_out = self.think(data)
        their_out = other.think(data)

        my_pred = int(np.argmax(my_out[0]))
        their_pred = int(np.argmax(their_out[0]))

        my_correct = 1.0 if my_pred == true_label else 0.0
        their_correct = 1.0 if their_pred == true_label else 0.0

        # Base score = correctness × (1 + confidence)
        my_score = my_correct * (1.0 + float(np.max(my_out[0])))
        their_score = their_correct * (1.0 + float(np.max(their_out[0])))

        # Effective score weighted by aggression
        my_effective = my_score * (1.0 + 0.6 * self.genome.aggression)
        their_effective = their_score * (1.0 + 0.6 * other.genome.aggression)

        # Determine winner
        if my_effective > their_effective + 1e-9:
            # Self wins
            stolen_base = 10.0 + 6.0 * self.genome.aggression
            stolen_amount = min(stolen_base, other.energy * 0.15, 30.0)
            stolen_amount = max(1.0, stolen_amount)

            self.energy += stolen_amount
            other.energy -= stolen_amount
            self.competitive_wins += 1
            other.reputation -= 0.05 * self.genome.aggression
            self.reputation += 0.05

            # Log social interactions
            self.memory.social_interactions[other.id].append(('competition_win', stolen_amount))
            other.memory.social_interactions[self.id].append(('competition_loss', -stolen_amount))
            return True

        elif their_effective > my_effective + 1e-9:
            # Other wins
            stolen_base = 10.0 + 6.0 * other.genome.aggression
            stolen_amount = min(stolen_base, self.energy * 0.15, 30.0)
            stolen_amount = max(1.0, stolen_amount)

            other.energy += stolen_amount
            self.energy -= stolen_amount
            other.reputation += 0.05
            self.reputation -= 0.05 * other.genome.aggression

            other.memory.social_interactions[self.id].append(('competition_win', stolen_amount))
            self.memory.social_interactions[other.id].append(('competition_loss', -stolen_amount))
            return False

        # Draw or indeterminate
        return False

    def social_learn_from_peer(self, peer: 'NeuralAgent', shared_experience: Tuple[np.ndarray, int]):
        """Attempt a one-layer imitation if peer has a stronger correct prediction."""
        bond_strength = self.social_bonds.get(peer.id, 0.0)
        if bond_strength <= 0.2:
            return
        if random.random() >= self.genome.social_learning_rate:
            return

        data, true_label = shared_experience
        peer_out = peer.think(data)[0]
        peer_pred = int(np.argmax(peer_out))

        my_out_before = self.think(data)[0]
        my_pred_before = int(np.argmax(my_out_before))

        # Only update if peer was correct and self was incorrect OR peer is more confident when both correct
        if (peer_pred == true_label) and (
            (my_pred_before != true_label) or
            ((my_pred_before == true_label) and (float(np.max(peer_out)) > float(np.max(my_out_before))))
        ):
            # Build last hidden activation (activations except last layer)
            if len(self.weights) > 1:
                temp_acts = [data.reshape(1, -1)]
                for i in range(len(self.weights) - 1):
                    z_val = np.dot(temp_acts[-1], self.weights[i]) + self.biases[i]
                    temp_acts.append(self._activate(z_val, i))
                x_prev_layer = temp_acts[-1]
            else:
                x_prev_layer = data.reshape(1, -1)

            error_signal = peer_out - my_out_before  # shape: (3,)
            eff_lr = self.genome.learning_rate * self.genome.social_learning_rate * bond_strength * 0.1

            # Update only the last layer
            self.weights[-1] += np.dot(x_prev_layer.T, error_signal.reshape(1, -1)) * eff_lr
            self.biases[-1]  += error_signal * eff_lr

            # Log the social learning attempt
            self.memory.social_interactions[peer.id].append(('social_learning_attempt', 1.0))

    def cooperate_with(self, other: 'NeuralAgent', environment_data: Tuple[np.ndarray, int]) -> float:
        """Attempt cooperation: combine predictions. Returns cooperation bonus given to each."""
        if random.random() > self.genome.cooperation_tendency:
            self.memory.social_interactions[other.id].append(('cooperation_skipped', 0.0))
            return 0.0

        data, true_label = environment_data
        my_out = self.think(data)[0]
        their_out = other.think(data)[0]

        bond_to_other = self.social_bonds.get(other.id, 0.1)
        bond_from_other = other.social_bonds.get(self.id, 0.1)
        trust_weight = (bond_to_other + bond_from_other) / 2.0

        # Weights for combining: emphasize trust and cooperation_tendency
        self_w = (1 - trust_weight) * (0.5 + 0.5 * self.genome.cooperation_tendency)
        other_w = trust_weight * (0.5 + 0.5 * other.genome.cooperation_tendency)
        norm = (self_w + other_w) if (self_w + other_w) > 0 else 1.0

        combined = (my_out * self_w + their_out * other_w) / norm
        combined_pred = int(np.argmax(combined))

        coop_bonus = 0.0
        if combined_pred == true_label:
            base_bonus = 3.0
            confidence = float(np.max(combined))
            risk_factor = 1.0 + 0.2 * self.genome.risk_tolerance
            coop_bonus = base_bonus * confidence * risk_factor

            # Split energy bonus
            self_share = coop_bonus * self.genome.metabolic_efficiency * 0.5
            other_share = coop_bonus * other.genome.metabolic_efficiency * 0.5

            self.energy += self_share
            other.energy += other_share

            # Strengthen bonds and reputations
            new_bond = min(1.0, bond_to_other + 0.05)
            self.social_bonds[other.id] = new_bond
            other.social_bonds[self.id] = min(1.0, bond_from_other + 0.05)

            self.cooperative_acts += 1
            self.reputation += 0.1 * trust_weight
            other.reputation += 0.1 * trust_weight

            self.memory.social_interactions[other.id].append(('cooperation_success', self_share))
            other.memory.social_interactions[self.id].append(('cooperation_success', other_share))
        else:
            # Cooperation failed: weaken bonds & reputation
            self.social_bonds[other.id] = max(0.0, bond_to_other - 0.01)
            other.social_bonds[self.id] = max(0.0, bond_from_other - 0.01)
            self.reputation -= 0.02
            other.reputation -= 0.02

            self.memory.social_interactions[other.id].append(('cooperation_fail', 0.0))
            other.memory.social_interactions[self.id].append(('cooperation_fail', 0.0))

        return coop_bonus

    def adapt_to_stress(self):
        """If stress_level is high, either recover (energy gain) or incur penalty."""
        self.survival_challenges_faced += 1
        if self.stress_level <= 0.6:
            return

        if random.random() < self.genome.stress_resistance:
            # Successful adaptation → reduce stress, gain small energy
            reduction = 0.1 + 0.1 * self.genome.stress_resistance
            self.stress_level = max(0.0, self.stress_level - reduction)
            self.energy += 1.0 * self.genome.metabolic_efficiency
            self.survival_challenges_overcome += 1
        else:
            # Failed adaptation → lose energy & fitness
            penalty = (self.stress_level * 2.0) / max(0.1, self.genome.metabolic_efficiency)
            self.energy -= penalty
            self.fitness -= self.stress_level * 0.5

        # If stress is very high, gradually shift behavioral traits
        if self.stress_level > 0.8:
            self.genome.risk_tolerance = max(0.05, 0.95 * self.genome.risk_tolerance)
            self.genome.exploration_rate = max(0.05, 0.9 * self.genome.exploration_rate)
            if self.genome.aggression > 0.5:
                self.genome.aggression *= 0.98
            else:
                self.genome.cooperation_tendency = min(1.0, 1.02 * self.genome.cooperation_tendency)

        # Clip stress to [0,1]
        self.stress_level = np.clip(self.stress_level, 0.0, 1.0)

    def territorial_behavior(self, nearby_agents: List['NeuralAgent']) -> Tuple[List['NeuralAgent'], List['NeuralAgent']]:
        """Partition nearby agents into competitors vs. potential allies."""
        competitors = []
        allies = []
        for agent in nearby_agents:
            if agent.id == self.id or agent.energy <= 0:
                continue
            dx = self.position[0] - agent.position[0]
            dy = self.position[1] - agent.position[1]
            distance = np.sqrt(dx * dx + dy * dy)
            if distance >= self.genome.territorial_radius:
                continue

            # If same species or strong bond, consider as ally with some probability
            is_ally = False
            if agent.species_id == self.species_id:
                prob_ally = (self.genome.cooperation_tendency + self.social_bonds.get(agent.id, 0.0)) / 1.5
                if random.random() < prob_ally:
                    is_ally = True
            elif self.social_bonds.get(agent.id, 0.0) > 0.7:
                if random.random() < 0.5 * self.genome.cooperation_tendency:
                    is_ally = True

            if is_ally:
                allies.append(agent)
            else:
                competitors.append(agent)

        return competitors, allies

    def environmental_adaptation(self, environment_type: str, difficulty: float, resource_availability: float) -> float:
        """
        Compute an adaptation_score based on genome vs. environment_type,
        adjust energy accordingly, and update specialization_score.
        """
        adaptation_score = 0.0
        base_energy_cost = 0.5 / max(0.1, self.genome.metabolic_efficiency)

        if environment_type == "resource_scarce":
            if self.genome.metabolic_efficiency > 1.0: # Higher is less efficient here, so this should be < 1.0
                adaptation_score += 0.3 * (2.0 - self.genome.metabolic_efficiency) # Penalize inefficiency
            if self.genome.exploration_rate > 0.4:
                adaptation_score += 0.2
            if self.genome.cooperation_tendency > 0.5:
                adaptation_score += 0.2
            base_energy_cost *= (1.5 / max(0.1, resource_availability))

        elif environment_type == "highly_competitive":
            if self.genome.aggression > 0.6:
                adaptation_score += 0.3 * self.genome.aggression
            if self.genome.risk_tolerance > 0.5:
                adaptation_score += 0.2
            adaptation_score += 0.2 * self.genome.stress_resistance
            base_energy_cost *= (1.0 + difficulty * 0.5)

        elif environment_type == "rapidly_changing":
            if self.genome.exploration_rate > 0.5:
                adaptation_score += 0.3
            if self.genome.learning_rate > 0.015:
                adaptation_score += 0.2
            if self.genome.social_learning_rate > 0.1:
                adaptation_score += 0.2
            base_energy_cost *= (1.0 + difficulty * 0.3)

        else:  # "balanced"
            adaptation_score += 0.1 + 0.1 * self.genome.stress_resistance

        adaptation_energy = adaptation_score * difficulty * 2.0 * self.genome.metabolic_efficiency
        self.energy += adaptation_energy
        self.energy -= base_energy_cost

        # Update specialization: positive if well-adapted, negative if poorly adapted
        self.specialization_score += 0.1 * (adaptation_score - 0.2)
        self.specialization_score = np.clip(self.specialization_score, -1.0, 1.0)

        self.adaptation_history.append(adaptation_score)
        return adaptation_score

    def can_reproduce(self) -> bool:
        """Check if agent meets minimum criteria to attempt reproduction."""
        return (
            self.energy > self.genome.reproduction_threshold and
            self.age > 12 and
            self.stress_level < 0.85
        )

    def _calculate_compatibility(self, partner: 'NeuralAgent') -> float:
        """Compute a compatibility score (0..1) combining genetics, species, and fitness similarity."""
        # (a) Layer/architecture similarity
        len_diff = abs(len(self.genome.layers) - len(partner.genome.layers))
        layer_similarity = max(0.0, 1.0 - len_diff / 5.0)

        act_self = self.genome.activation_mutations[0] if self.genome.activation_mutations else ""
        act_partner = partner.genome.activation_mutations[0] if partner.genome.activation_mutations else ""
        activation_similarity = 1.0 if act_self == act_partner else 0.3

        genetic_sim = 0.5 * layer_similarity + 0.5 * activation_similarity

        # (b) Behavioral trait similarity
        trait_diff = abs(self.genome.aggression - partner.genome.aggression)
        trait_diff += abs(self.genome.cooperation_tendency - partner.genome.cooperation_tendency)
        behavioral_comp = max(0.0, 1.0 - (trait_diff / 2.0))

        # (c) Fitness ratio
        f1, f2 = max(1.0, self.fitness), max(1.0, partner.fitness)
        fitness_ratio = min(f1, f2) / max(f1, f2)

        # (d) Species match bonus
        species_bonus = 1.0 if self.species_id == partner.species_id else 0.2

        compatibility = (
            0.4 * species_bonus +
            0.3 * genetic_sim +
            0.2 * behavioral_comp +
            0.1 * fitness_ratio
        )
        return np.clip(compatibility, 0.0, 1.0)

    def _create_child_genome(
        self,
        partner: Optional['NeuralAgent'],
        environmental_pressure: float
    ) -> Genome:
        """Perform crossover + mutation to produce an offspring genome."""
        parent1 = self.genome
        child_attrs: Dict[str, Any] = {}

        # Use original (non-stress-modified) genome traits for inheritance
        p1_originals = getattr(self, '_original_genome_traits', {})
        p2_originals = getattr(partner, '_original_genome_traits', {}) if partner else {}

        # 1) Crossover
        if partner and random.random() < parent1.crossover_rate:
            parent2 = partner.genome

            # (a) Decide child depth (#layers)
            len1, len2 = len(parent1.layers), len(parent2.layers)
            if random.random() < 0.5:
                new_len = (len1 + len2) // 2
            else:
                new_len = random.choice([len1, len2])
            new_len = max(3, min(7, new_len))  # enforce at least [20, hidden, 3], at most 5 hidden layers

            # (b) Build new layers list
            num_hidden = new_len - 2
            layers_new = [20]
            for h in range(num_hidden):
                s1 = parent1.layers[h + 1] if h + 1 < (len1 - 1) else random.randint(8, 20)
                s2 = parent2.layers[h + 1] if h + 1 < (len2 - 1) else random.randint(8, 20)
                if random.random() < 0.7:
                    chosen = (s1 + s2) // 2
                else:
                    chosen = random.choice([s1, s2])
                layers_new.append(chosen)
            layers_new.append(3)
            child_attrs['layers'] = layers_new

            # (c) Build new activation_mutations
            acts_new = []
            for idx in range(new_len - 1):
                if idx == (new_len - 2):
                    acts_new.append('softmax')
                else:
                    a1 = parent1.activation_mutations[idx] if idx < (len1 - 1) else random.choice(['relu', 'tanh'])
                    a2 = parent2.activation_mutations[idx] if idx < (len2 - 1) else random.choice(['relu', 'tanh'])
                    acts_new.append(random.choice([a1, a2]))
            child_attrs['activation_mutations'] = acts_new

            # (d) Average numerical traits (use original genetic values for stress-affected traits)
            for field_name, val1 in parent1.__dict__.items():
                if field_name in ('layers', 'activation_mutations'):
                    continue
                if isinstance(val1, (int, float)):
                    # Use original genetic values instead of stress-modified phenotype
                    effective_val1 = p1_originals.get(field_name, val1)
                    val2 = getattr(parent2, field_name)
                    if isinstance(val2, (int, float)):
                        effective_val2 = p2_originals.get(field_name, val2)
                        avg = (effective_val1 + effective_val2) / 2.0
                        if field_name == 'memory_capacity':
                            avg = int(round(avg))
                        child_attrs[field_name] = avg
                    else:
                        child_attrs[field_name] = effective_val1
                else:
                    child_attrs[field_name] = val1

        else:
            # No crossover: copy parent1 (use original genetic values for stress-affected traits)
            for field_name, val in parent1.__dict__.items():
                if isinstance(val, list):
                    child_attrs[field_name] = val.copy()
                else:
                    child_attrs[field_name] = p1_originals.get(field_name, val)

        child_genome = Genome(**child_attrs)

        # 2) Mutation: adjust mutation_rate based on environmental_pressure
        eff_mutation = np.clip(
            child_genome.mutation_rate * (1.0 + 0.2 * environmental_pressure),
            0.01, 0.5
        )

        if random.random() < eff_mutation:
            self._mutate_genome(child_genome, environmental_pressure, eff_mutation)

        # 3) Enforce consistency: input_dim=20, output_dim=3
        child_genome.layers[0] = 20
        child_genome.layers[-1] = 3

        # Fix activation length if needed
        required_act_len = len(child_genome.layers) - 1
        curr_act_len = len(child_genome.activation_mutations)
        if curr_act_len < required_act_len:
            for _ in range(required_act_len - curr_act_len):
                child_genome.activation_mutations.insert(-1, random.choice(['relu', 'tanh']))
        elif curr_act_len > required_act_len:
            child_genome.activation_mutations = child_genome.activation_mutations[:required_act_len]

        # Always ensure last activation is softmax
        if required_act_len > 0:
            child_genome.activation_mutations[-1] = 'softmax'
        else: # Should not happen with layers min 3 (e.g. [20,10,3] -> 2 activation layers)
              # If layers is e.g. [20,3] (len 2), required_act_len is 1.
              # If somehow layers is [20] (len 1), required_act_len is 0.
            child_genome.activation_mutations = []


        return child_genome

    def _mutate_genome(self, genome: Genome, environmental_pressure: float, current_mutation_rate: float):
        """Perform structural and parametric mutations on the genome."""
        # A) Structural mutations on layers
        if random.random() < current_mutation_rate * 0.2:
            action = random.choice(['add', 'remove', 'resize'])
            if action == 'add' and len(genome.layers) < 6: # Max 4 hidden layers if total < 6 (input+output+hidden)
                idx_insert = random.randint(1, len(genome.layers) - 1) # Insert before output layer
                new_size = random.randint(5, 25)
                genome.layers.insert(idx_insert, new_size)
                # activation for layer PRECEDING new_size needs to be inserted
                genome.activation_mutations.insert(idx_insert -1 , random.choice(['relu', 'tanh', 'sigmoid']))
            elif action == 'remove' and len(genome.layers) > 3: # Min 1 hidden layer
                idx_remove = random.randint(1, len(genome.layers) - 2) # Remove a hidden layer
                genome.layers.pop(idx_remove)
                genome.activation_mutations.pop(idx_remove -1) # Remove corresponding activation
            elif action == 'resize' and len(genome.layers) > 2: # Has hidden layers
                idx_resize = random.randint(1, len(genome.layers) - 2) # Resize a hidden layer
                change = random.randint(-5, 5)
                genome.layers[idx_resize] = max(3, min(30, genome.layers[idx_resize] + change))

        # B) Activation mutations (except final softmax)
        for i in range(max(0, len(genome.activation_mutations) - 1)):
            if random.random() < current_mutation_rate * 0.15:
                genome.activation_mutations[i] = random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu', 'swish'])

        # C) Numeric trait perturbations
        params = [
            ('learning_rate',        0.0005, 0.08,  0.01),
            ('aggression',           0.0,    1.0,   0.1),
            ('cooperation_tendency', 0.0,    1.0,   0.1),
            ('risk_tolerance',       0.0,    1.0,   0.1),
            ('exploration_rate',     0.05,   0.8,   0.1),
            ('social_learning_rate', 0.01,   0.4,   0.05),
            ('metabolic_efficiency', 0.6,    1.4,   0.1),
            ('stress_resistance',    0.1,    0.95,  0.1),
            ('reproduction_threshold',80.0, 250.0,10.0),
            ('mutation_rate',        0.02,   0.3,   0.02),
            ('memory_capacity',      15,     100,   5),
            ('crossover_rate',       0.3,    0.95,  0.1),
            ('territorial_radius',   2.0,    20.0,  1.0),
            ('mating_selectivity',   0.1,    0.9,   0.1),
            ('parental_investment',  0.05,   0.6,   0.05)
        ]
        for name, min_v, max_v, scale in params:
            if random.random() < current_mutation_rate * 0.4:
                curr = getattr(genome, name)
                change = random.gauss(0, scale * (0.1 + 0.05 * environmental_pressure))
                new = curr + change
                new = np.clip(new, min_v, max_v)
                if name == 'memory_capacity':
                    new = int(round(new))
                setattr(genome, name, new)

    def _get_offspring_position(self) -> Tuple[float, float]:
        """Place offspring near parent, but still within [0,100]×[0,100]."""
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(0.5, max(1.0, self.genome.territorial_radius / 3.0))
        x_new = self.position[0] + distance * np.cos(angle)
        y_new = self.position[1] + distance * np.sin(angle)
        return (np.clip(x_new, 0.0, 100.0), np.clip(y_new, 0.0, 100.0))

    def advanced_reproduce(
        self,
        partner: Optional['NeuralAgent'],
        environmental_pressure: float,
        next_id: Optional[int] = None
    ) -> Optional['NeuralAgent']:
        """
        Attempt sexual or asexual reproduction. Return a new child agent if successful, else None.
        next_id: unique ID for the child, provided by the environment's ID counter.
        """

        if not self.can_reproduce():
            return None

        # 1) If partner is provided, check compatibility + pickiness
        if partner:
            compatibility = self._calculate_compatibility(partner)
            # Now: higher mating_selectivity should make rejection easier
            # so we compare random() > compatibility × (1 - selectivity)
            threshold = compatibility * (1.0 - self.genome.mating_selectivity)
            if random.random() > threshold : # Higher selectivity or lower compatibility increases chance of this being true (rejection)
                self.memory.social_interactions[partner.id].append(('reproduction_rejected', compatibility))
                return None
        else:
            # Asexual reproduction only 30% of the time
            if random.random() > 0.3:
                return None

        # 2) Pay reproduction cost
        cost = 50.0 + 50.0 * self.genome.parental_investment
        cost /= max(0.5, self.genome.metabolic_efficiency)
        if self.energy < cost:
            return None

        self.energy -= cost
        self.offspring_count += 1

        # 3) Build child's genome via crossover + mutation
        child_genome = self._create_child_genome(partner, environmental_pressure)

        # 4) Instantiate child agent (use provided ID or fallback to random)
        child_id = next_id if next_id is not None else random.randint(1_000_000, 9_999_999)
        child_pos = self._get_offspring_position()
        child = NeuralAgent(agent_id=child_id, genome=child_genome, position=child_pos)

        # (a) Generation & lineage
        if partner:
            child.generation = max(self.generation, partner.generation) + 1
            # Randomly pick one parent’s lineage to continue, then append child_id
            chosen_lineage = random.choice([self.lineage, partner.lineage]).copy()
            child.lineage = chosen_lineage + [child_id]
        else:
            child.generation = self.generation + 1
            child.lineage = self.lineage.copy() + [child_id]


        # (b) Initial energy transfer
        gift = 60.0 * self.genome.parental_investment * self.genome.metabolic_efficiency
        gift = min(gift, 0.5 * self.energy) # Parent cannot give more than half its remaining energy
        child.energy = min(250.0 + gift, child.genome.reproduction_threshold * 1.5 + 100.0)


        # (c) Initial social bonds
        child.social_bonds[self.id] = 0.5 + 0.5 * self.genome.parental_investment
        if partner:
            child.social_bonds[partner.id] = 0.5 + 0.5 * partner.genome.parental_investment

        return child

    def decay(self) -> bool:
        """
        Age the agent by one step, deduct energy based on age and stress.
        Return True if still alive (energy>0), else False.
        """
        self.age += 1
        age_factor = 1.0 + (self.age / 200.0)
        base_decay = 0.8 * age_factor
        energy_lost = base_decay / max(0.1, self.genome.metabolic_efficiency)
        energy_lost += 0.1 * self.stress_level

        self.energy -= energy_lost

        return self.energy > 0.0


# --- AdvancedEvolutionEnvironment Class ---

class AdvancedEvolutionEnvironment:
    """Simulate a population of NeuralAgents in an evolving environment."""

    _next_agent_id_counter = 10000 # Class variable for generating unique IDs

    def __init__(
        self,
        population_size: int = 50,
        world_size: Tuple[float, float] = (100.0, 100.0),
        seed_agents: Optional[List[NeuralAgent]] = None
    ):
        self.agents: List[NeuralAgent] = []
        self.time_step = 0
        self.world_size = world_size

        # Current environmental state
        self.environment_type = "balanced"
        self.environmental_pressure = 0.3
        self.resource_availability = 1.0

        # Species tracking
        self.species_populations: Dict[int, int] = defaultdict(int)
        self.species_fitness: Dict[int, List[float]] = defaultdict(list)
        self.species_traits_avg: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Event logs
        self.cooperation_events_total = 0
        self.competition_events_total = 0
        self.extinction_events: List[Tuple[int, int]] = []   # (time_step, species_id)
        self.speciation_events: List[Tuple[int, int, int]] = []  # (time_step, old_sid, new_sid)

        # Time-series histories (capped at 10,000 entries to prevent memory bloat)
        _MAX_HISTORY = 10000
        self.population_history: deque = deque(maxlen=_MAX_HISTORY)
        self.avg_fitness_history: deque = deque(maxlen=_MAX_HISTORY)
        self.max_fitness_history: deque = deque(maxlen=_MAX_HISTORY)
        self.diversity_history: deque = deque(maxlen=_MAX_HISTORY)
        self.avg_cooperation_tendency_history: deque = deque(maxlen=_MAX_HISTORY)
        self.avg_aggression_history: deque = deque(maxlen=_MAX_HISTORY)
        self.avg_age_history: deque = deque(maxlen=_MAX_HISTORY)
        self.avg_specialization_history: deque = deque(maxlen=_MAX_HISTORY)
        self.avg_stress_history: deque = deque(maxlen=_MAX_HISTORY)
        self.resource_history: deque = deque(maxlen=_MAX_HISTORY)
        self.pressure_history: deque = deque(maxlen=_MAX_HISTORY)
        
        # Initialize unique ID counter from max existing ID if seeded, or default
        if seed_agents:
             max_id_so_far = 0
             for sa in seed_agents:
                 if sa.id > max_id_so_far:
                     max_id_so_far = sa.id
                 # also check lineage for max historical ID
                 if sa.lineage:
                     max_lineage_id = max(sa.lineage)
                     if max_lineage_id > max_id_so_far:
                         max_id_so_far = max_lineage_id
             AdvancedEvolutionEnvironment._next_agent_id_counter = max_id_so_far + 1
        # else it uses the class default


        # Initialize population
        if seed_agents:
            for idx, agent in enumerate(seed_agents[:population_size]):
                self.agents.append(agent)
                self.species_populations[agent.species_id] += 1

            while len(self.agents) < population_size:
                new_id = AdvancedEvolutionEnvironment._next_agent_id_counter
                AdvancedEvolutionEnvironment._next_agent_id_counter +=1
                pos = (random.uniform(0, world_size[0]), random.uniform(0, world_size[1]))
                agent = NeuralAgent(agent_id=new_id, position=pos)
                agent.lineage = [new_id] # Initialize lineage for new random agents
                self.agents.append(agent)
                self.species_populations[agent.species_id] += 1
        else:
            for _ in range(population_size): # Corrected loop variable from i to _
                new_id = AdvancedEvolutionEnvironment._next_agent_id_counter
                AdvancedEvolutionEnvironment._next_agent_id_counter +=1
                pos = (random.uniform(0, world_size[0]), random.uniform(0, world_size[1]))
                agent = NeuralAgent(agent_id=new_id, position=pos)
                agent.lineage = [new_id] # Initialize lineage
                self.agents.append(agent)
                self.species_populations[agent.species_id] += 1
                
    @classmethod
    def _get_next_agent_id(cls) -> int:
        next_id = cls._next_agent_id_counter
        cls._next_agent_id_counter += 1
        return next_id


    def _migrate_histories(self):
        """Convert plain lists to capped deques for backward compat with older pkl files."""
        _MAX_HISTORY = 10000
        history_attrs = [
            'population_history', 'avg_fitness_history', 'max_fitness_history',
            'diversity_history', 'avg_cooperation_tendency_history',
            'avg_aggression_history', 'avg_age_history', 'avg_specialization_history',
            'avg_stress_history', 'resource_history', 'pressure_history'
        ]
        for attr in history_attrs:
            val = getattr(self, attr, None)
            if val is None:
                setattr(self, attr, deque(maxlen=_MAX_HISTORY))
            elif isinstance(val, list):
                d = deque(val[-_MAX_HISTORY:], maxlen=_MAX_HISTORY)
                setattr(self, attr, d)
            elif isinstance(val, deque) and val.maxlen != _MAX_HISTORY:
                d = deque(list(val)[-_MAX_HISTORY:], maxlen=_MAX_HISTORY)
                setattr(self, attr, d)

    def set_environment_conditions(self, env_type: str, pressure: float, resources: float):
        """Manually override environment type, pressure, and resource availability."""
        self.environment_type = env_type
        self.environmental_pressure = np.clip(pressure, 0.0, 2.0)
        self.resource_availability = np.clip(resources, 0.1, 3.0)

    def _change_environment(self):
        """Stochastically shift to a new environment every 200 steps."""
        choices = ["resource_scarce", "highly_competitive", "rapidly_changing", "balanced"]
        if self.environment_type == "balanced" and random.random() < 0.7:
            # Stay balanced most of the time
            pass
        else:
            self.environment_type = random.choice(choices)

        if self.environment_type == "resource_scarce":
            self.environmental_pressure = random.uniform(0.2, 0.7)
            self.resource_availability = random.uniform(0.15, 0.5)
        elif self.environment_type == "highly_competitive":
            self.environmental_pressure = random.uniform(0.6, 1.3)
            self.resource_availability = random.uniform(0.6, 1.1)
        elif self.environment_type == "rapidly_changing":
            self.environmental_pressure = random.uniform(0.4, 0.9)
            self.resource_availability = random.uniform(0.5, 1.0)
        else:  # balanced
            self.environmental_pressure = random.uniform(0.1, 0.4)
            self.resource_availability = random.uniform(0.7, 1.3)

        print(
            f"--- ENV CHANGE @ step {self.time_step}: "
            f"type={self.environment_type}, P={self.environmental_pressure:.2f}, R={self.resource_availability:.2f} ---"
        )

    def generate_complex_environment_data(self) -> Tuple[np.ndarray, int]:
        """
        Produce a 20-dim pattern and a label {0,1,2} depending on time_step.
        - pattern_type 0: sinusoidal
        - pattern_type 1: random walk + trend
        - pattern_type 2: chaotic sequence
        """
        t = np.linspace(0, 5 * np.pi, 20)
        selector = (self.time_step // 150) % 3

        if selector == 0:
            freq = 1.0 + 0.4 * np.sin(self.time_step * 0.003 + (self.time_step // 500))
            phase = self.time_step * 0.015
            amp = 0.6 + 0.4 * np.sin(self.time_step * 0.007)
            pattern = amp * np.sin(freq * t + phase)
            label = 0

        elif selector == 1:
            volatility = 0.08 + 0.07 * abs(np.sin(self.time_step * 0.005))
            steps = np.random.randn(20) * volatility
            trend = np.linspace(-0.25, 0.25, 20) + 0.015 * np.sin(self.time_step * 0.008) * np.arange(20)
            pattern = np.cumsum(steps) + trend
            label = 1

        else:
            seq = [random.uniform(0.1, 0.2), random.uniform(0.1, 0.2)]
            modulus = max(2, 4 + int(3 * abs(np.sin(self.time_step * 0.004))))
            chaos_factor = 0.05 + 0.1 * abs(np.sin(self.time_step * 0.01)) * self.environmental_pressure
            for _ in range(18):
                nxt = seq[-1] * 1.05 + seq[-2] * 0.95
                nxt = (nxt % modulus) if modulus > 0 else nxt # Modulus can be 0 if sin result is specific
                nxt += random.uniform(-chaos_factor, chaos_factor)
                seq.append(nxt)
            pattern = np.array(seq, dtype=float) / max(1, modulus) # Avoid div by zero for modulus
            label = 2

        noise_level = 0.015 + 0.06 * self.environmental_pressure
        noise = np.random.randn(20) * noise_level
        clarity = np.clip(self.resource_availability * 0.7, 0.4, 1.1)
        pattern = pattern * clarity + noise
        return np.clip(pattern, -1.5, 1.5), label

    def _get_nearby_agents(self, agent: NeuralAgent, radius: float) -> List[NeuralAgent]:
        """Return living agents (excluding self) within Euclidean distance < radius."""
        nearby = []
        for other in self.agents:
            if other.id == agent.id or other.energy <= 0:
                continue
            dx = agent.position[0] - other.position[0]
            dy = agent.position[1] - other.position[1]
            dist2 = dx * dx + dy * dy
            if dist2 < radius * radius:
                nearby.append(other)
        return nearby

    def run_advanced_simulation_step(self) -> bool:
        """
        Execute one full time step:
         1) Possibly change environment
         2) Generate a new environment pattern
         3) Phase 1: Individual actions (move, think, learn, stress, adapt)
         4) Phase 2: Social interactions (cooperate, compete, social learning)
         5) Phase 3: Reproduction (pairing, producing offspring)
         6) Phase 4: Offspring integration, aging & death, species events, culling
         7) Phase 5: Record statistics
        Returns False if the population is extinct before starting, else True.
        """
        if not self.agents:
            return False

        self.time_step += 1

        # 1) Possibly change environment every 200 steps
        if self.time_step > 1 and self.time_step % 200 == 0:
            self._change_environment()

        # 2) Generate environment stimulus
        env_data, true_label = self.generate_complex_environment_data()

        # PHASE 1: Individual actions
        agent_indices = list(range(len(self.agents)))
        random.shuffle(agent_indices)

        for idx in agent_indices:
            if idx >= len(self.agents): # Agent might have been removed by earlier interaction in a more complex step
                continue
            agent = self.agents[idx]
            if agent.energy <= 0.0:
                continue

            # A) Random movement scaled by exploration and stress
            move_dist = max(0.0, (0.5 + 1.5 * agent.genome.exploration_rate) * (1.0 - 0.5 * agent.stress_level))
            angle = random.uniform(0, 2 * np.pi)
            new_x = np.clip(agent.position[0] + move_dist * np.cos(angle), 0.0, self.world_size[0])
            new_y = np.clip(agent.position[1] + move_dist * np.sin(angle), 0.0, self.world_size[1])
            agent.position = (new_x, new_y)

            # B) Think on environment pattern
            output = agent.think(env_data, add_noise=True)
            pred = int(np.argmax(output[0]))

            # C) Compute reward: +6 if correct, -2.5 if wrong, plus confidence & entropy bonuses
            base_reward = 6.0 if pred == true_label else -2.5
            confidence = float(np.max(output[0]))
            conf_bonus = 1.5 * confidence

            # Entropy penalty → reward if model is certain
            eps = 1e-9
            ent = -np.sum((output[0] + eps) * np.log(output[0] + eps))
            cert_bonus = (np.log(3.0) - ent) * 1.0 # Max entropy for 3 classes is log(3)

            total_reward = (base_reward + conf_bonus + cert_bonus)
            total_reward *= np.clip(self.resource_availability, 0.5, 1.5)
            total_reward = np.clip(total_reward, -10.0, 12.0)

            agent.learn_from_environment(env_data, true_label, total_reward)
            agent.classification_accuracy.append(1.0 if pred == true_label else 0.0)

            # D) Update stress and attempt adaptation
            agent.stress_level += (0.05 * self.environmental_pressure) - (0.03 * agent.genome.stress_resistance)
            agent.stress_level = np.clip(agent.stress_level, 0.0, 1.0)
            agent.adapt_to_stress()
            agent.environmental_adaptation(self.environment_type, self.environmental_pressure, self.resource_availability)

        # PHASE 2: Social Interactions
        random.shuffle(agent_indices) # Re-shuffle for interaction order
        coop_acts_step = 0

        for idx in agent_indices:
            if idx >= len(self.agents):
                continue
            agent = self.agents[idx]
            if agent.energy <= 0.0:
                continue

            interaction_radius = agent.genome.territorial_radius * (1.0 + 0.2 * agent.genome.exploration_rate)
            nearby = self._get_nearby_agents(agent, interaction_radius)
            if not nearby:
                continue

            competitors, allies = agent.territorial_behavior(nearby)

            # A) Cooperation attempt
            if allies and random.random() < (agent.genome.cooperation_tendency + 0.1 * agent.reputation):
                ally = random.choice(allies)
                if ally.energy > 0.0: # Check if ally is still alive
                    bonus = agent.cooperate_with(ally, (env_data, true_label))
                    if bonus > 0.0:
                        coop_acts_step += 1

            # B) Competition attempt
            if competitors and random.random() < (
                agent.genome.aggression + 0.2 * agent.genome.risk_tolerance - 0.1 * agent.reputation
            ):
                opponent = random.choice(competitors)
                if opponent.energy > 0.0: # Check if opponent is still alive
                    agent.compete_with(opponent, (env_data, true_label))
                    self.competition_events_total += 1

            # C) Social learning attempt
            if nearby and random.random() < (0.5 * agent.genome.social_learning_rate):
                peer = random.choice(nearby) # Could be ally or competitor for learning
                if peer.id != agent.id and peer.energy > 0.0: # Check if peer is alive
                    agent.social_learn_from_peer(peer, (env_data, true_label))

        self.cooperation_events_total += coop_acts_step

        # PHASE 3: Reproduction
        newly_born: List[NeuralAgent] = []
        parents = [ag for ag in self.agents if ag.can_reproduce() and ag.energy > 0.0]
        random.shuffle(parents)

        for parent in parents:
            if parent.energy <= 0: continue # Parent might have died during interactions

            if random.random() > 0.35: # Chance to attempt reproduction
                continue

            partner = None
            if random.random() < parent.genome.crossover_rate:
                extended_radius = parent.genome.territorial_radius * 1.8
                candidates = [
                    ag for ag in self._get_nearby_agents(parent, extended_radius)
                    if ag.can_reproduce() and ag.id != parent.id and ag.energy > 0.0
                ]
                if candidates:
                    comps = [parent._calculate_compatibility(c) for c in candidates]
                    threshold = 0.2 + 0.3 * parent.genome.mating_selectivity
                    valid_pairs = [(c, comp) for c, comp in zip(candidates, comps) if comp > threshold]
                    if valid_pairs:
                        total_w = sum(comp**2 for _, comp in valid_pairs)
                        if total_w > 1e-9: # Avoid division by zero
                            probs = np.array([comp**2 for _, comp in valid_pairs]) / total_w
                            try:
                                partner = np.random.choice([c for c, _ in valid_pairs], p=probs)
                            except ValueError: # If sum(probs) != 1 due to precision
                                partner = random.choice([c for c, _ in valid_pairs])
                        else: # if all compatibilities were zero (should be filtered by threshold)
                            partner = random.choice([c for c, _ in valid_pairs])


            child_id = self._get_next_agent_id()
            child = parent.advanced_reproduce(partner, self.environmental_pressure, next_id=child_id)
            if child:
                is_speciation = False
                old_sid = parent.species_id
                if child.species_id != parent.species_id:
                    is_speciation = True
                    old_sid = parent.species_id
                if partner and child.species_id != partner.species_id:
                    # If child is different from BOTH parents, it's a more significant speciation.
                    # The current logic logs if different from *either*.
                    is_speciation = True 
                    # old_sid for logging might be ambiguous if parents are different species.
                    # Let's use parent1's (current parent in loop) as primary reference.

                if is_speciation:
                    recent = any(evt[2] == child.species_id and evt[0] > self.time_step - 10
                                 for evt in self.speciation_events)
                    if not recent:
                        self.speciation_events.append((self.time_step, old_sid, child.species_id))
                newly_born.append(child)

        # PHASE 4: Offspring integration, aging, death, species events, culling
        # Build the new agent list atomically to avoid partial mutation visible to other threads.

        # (a) Age existing agents & collect survivors + newborns
        survivors: List[NeuralAgent] = []
        for ag in self.agents:
            if ag.decay():  # ag.decay() returns True if alive
                survivors.append(ag)
        # Add newborns to the survivors list (they skip decay on their birth step)
        survivors.extend(newly_born)
        # Single atomic assignment replaces the agent list
        self.agents = survivors


        # (c) Recompute species counts AFTER death AND adding newborns
        current_species_counts = defaultdict(int)
        for ag in self.agents:
            current_species_counts[ag.species_id] += 1

        # (d) Check for extinction events (compare with counts *before* this step's deaths but *after* births)
        # This requires species_populations to be up-to-date before this check.
        # Let's use a temporary snapshot of species counts before deaths for comparison.
        species_before_deaths_this_step = defaultdict(int)
        # Populate this from self.agents *before* filtering out dead ones but *after* adding newborns
        # This means self.agents list before survivors assignment.
        # The current self.species_populations reflects counts from end of *previous* step.
        # For accurate extinction detection:
        # 1. Add newborns.
        # 2. Get counts (snapshot_A).
        # 3. Perform decay.
        # 4. Get counts (snapshot_B).
        # 5. Compare snapshot_A and snapshot_B for extinctions.
        
        # Let's refine extinction logic: use self.species_populations (from end of PREVIOUS step)
        # and compare with current_species_counts (after births and deaths THIS step)
        
        # Store previous species counts before updating
        previous_step_species_counts = self.species_populations.copy()

        for sid, count_before_current_step_processing in previous_step_species_counts.items():
            count_after_current_step_processing = current_species_counts.get(sid, 0)
            if (count_before_current_step_processing > 0) and (count_after_current_step_processing == 0):
                # Make sure not just a very recent speciation event that might "die" immediately
                # if its first member didn't survive its first step.
                is_recent_new_species = any(evt[2] == sid and evt[0] >= self.time_step -1 # Looser: allow 1 step survival
                                         for evt in self.speciation_events)
                if not is_recent_new_species:
                    self.extinction_events.append((self.time_step, sid))
                    if sid in self.species_fitness: del self.species_fitness[sid]
                    if sid in self.species_traits_avg: del self.species_traits_avg[sid]
        
        self.species_populations = current_species_counts # Update to current counts

         # PHASE 5: Record all statistics for this time step
        self._record_step_statistics()
        if not self.agents: # Population might go extinct after culling/decay
             print(f"💀 ECOSYSTEM EXTINCT at step {self.time_step} (post-processing)!")
             return False
        return True

    def _record_step_statistics(self):
        """Collect population and species‐level stats and append to history lists."""
        if not self.agents:
            self.population_history.append(0)
            self.avg_fitness_history.append(0.0)
            self.max_fitness_history.append(0.0)
            self.diversity_history.append(0)
            self.avg_cooperation_tendency_history.append(0.0)
            self.avg_aggression_history.append(0.0)
            self.avg_age_history.append(0.0)
            self.avg_specialization_history.append(0.0)
            self.avg_stress_history.append(0.0)
        else:
            pop = len(self.agents)
            self.population_history.append(pop)

            fitnesses = [ag.fitness for ag in self.agents]
            self.avg_fitness_history.append(float(np.mean(fitnesses)) if fitnesses else 0.0)
            self.max_fitness_history.append(float(np.max(fitnesses)) if fitnesses else 0.0)

            diversity = len([sid for sid, cnt in self.species_populations.items() if cnt > 0])
            self.diversity_history.append(diversity)

            coop_vals = [ag.genome.cooperation_tendency for ag in self.agents]
            agg_vals = [ag.genome.aggression for ag in self.agents]
            age_vals = [ag.age for ag in self.agents]
            spec_vals = [ag.specialization_score for ag in self.agents]
            stress_vals = [ag.stress_level for ag in self.agents]

            self.avg_cooperation_tendency_history.append(float(np.mean(coop_vals)) if coop_vals else 0.0)
            self.avg_aggression_history.append(float(np.mean(agg_vals)) if agg_vals else 0.0)
            self.avg_age_history.append(float(np.mean(age_vals)) if age_vals else 0.0)
            self.avg_specialization_history.append(float(np.mean(spec_vals)) if spec_vals else 0.0)
            self.avg_stress_history.append(float(np.mean(stress_vals)) if stress_vals else 0.0)

        self.resource_history.append(self.resource_availability)
        self.pressure_history.append(self.environmental_pressure)

        self.species_fitness.clear()
        self.species_traits_avg.clear()

        temp_counts_for_avg = defaultdict(int)
        for ag in self.agents:
            sid = ag.species_id
            self.species_fitness[sid].append(ag.fitness)
            temp_counts_for_avg[sid] += 1
            for trait in (
                'aggression', 'cooperation_tendency',
                'exploration_rate', 'metabolic_efficiency',
                'stress_resistance', 'learning_rate'
            ):
                self.species_traits_avg[sid][trait] += getattr(ag.genome, trait)

        for sid, trait_dict in list(self.species_traits_avg.items()): # Iterate over a copy
            count = temp_counts_for_avg.get(sid, 0)
            if count > 0:
                for tname in trait_dict:
                    trait_dict[tname] /= count
            else: # Should not happen if agents exist for this sid
                if sid in self.species_traits_avg: # Check again before del
                     del self.species_traits_avg[sid]


    def get_advanced_ecosystem_stats(self) -> Dict[str, Any]:
        """
        Return a summary of the current ecosystem state:
        - status: "initializing", "running", or "extinct"
        - time_step, population, avg_fitness, max_fitness, num_species, avg_age, ...
        - environment_type, pressure, resource_availability
        - total_coop_events, extinctions_total, speciations_total
        - species_details: for each active species: {count, avg_fitness, traits}
        """
        if not self.agents and self.time_step == 0: # Just initialized, no steps run
            # Provide default initial values for a cleaner UI start
            return {
                "status": "initializing", 
                "time_step": self.time_step,
                "population": len(self.agents), # could be >0 if seeded
                "avg_fitness": 0.0, "max_fitness": 0.0, "num_species": 0,
                "avg_age": 0.0, "avg_cooperation": 0.0, "avg_aggression": 0.0,
                "avg_stress": 0.0, "avg_specialization": 0.0,
                "environment_type": self.environment_type,
                "environmental_pressure": self.environmental_pressure,
                "resource_availability": self.resource_availability,
                "total_coop_events": 0, "extinctions_total": 0, "speciations_total": 0,
                "species_details": {}
            }


        if not self.agents and self.time_step > 0: # Extinct after running
            return {"status": "extinct", "time_step": self.time_step}

        # If agents exist or it's step 0 with seed agents
        current_pop = self.population_history[-1] if self.population_history else len(self.agents)
        avg_fit = self.avg_fitness_history[-1] if self.avg_fitness_history else (np.mean([a.fitness for a in self.agents]) if self.agents else 0.0)
        max_fit = self.max_fitness_history[-1] if self.max_fitness_history else (np.max([a.fitness for a in self.agents]) if self.agents else 0.0)
        num_spec = self.diversity_history[-1] if self.diversity_history else len(self.species_populations)


        stats: Dict[str, Any] = {
            "status": "running",
            "time_step": self.time_step,
            "population": current_pop,
            "avg_fitness": avg_fit,
            "max_fitness": max_fit,
            "num_species": num_spec,
            "avg_age": self.avg_age_history[-1] if self.avg_age_history else (np.mean([a.age for a in self.agents]) if self.agents else 0.0),
            "avg_cooperation": self.avg_cooperation_tendency_history[-1] if self.avg_cooperation_tendency_history else (np.mean([a.genome.cooperation_tendency for a in self.agents]) if self.agents else 0.0),
            "avg_aggression": self.avg_aggression_history[-1] if self.avg_aggression_history else (np.mean([a.genome.aggression for a in self.agents]) if self.agents else 0.0),
            "avg_stress": self.avg_stress_history[-1] if self.avg_stress_history else (np.mean([a.stress_level for a in self.agents]) if self.agents else 0.0),
            "avg_specialization": self.avg_specialization_history[-1] if self.avg_specialization_history else (np.mean([a.specialization_score for a in self.agents]) if self.agents else 0.0),
            "environment_type": self.environment_type,
            "environmental_pressure": self.environmental_pressure,
            "resource_availability": self.resource_availability,
            "total_coop_events": self.cooperation_events_total,
            "extinctions_total": len(self.extinction_events),
            "speciations_total": len(self.speciation_events),
            "species_details": {}
        }

        for sid, cnt in self.species_populations.items():
            if cnt > 0:
                # Use current agent data for species_fitness if available, otherwise use history (less accurate for current step)
                current_agents_of_species = [ag.fitness for ag in self.agents if ag.species_id == sid]
                avg_fit_species = float(np.mean(current_agents_of_species)) if current_agents_of_species else 0.0
                
                # For traits, self.species_traits_avg should be up-to-date from _record_step_statistics
                traits = {
                    k: round(v, 3)
                    for k, v in self.species_traits_avg.get(sid, {}).items()
                }
                stats["species_details"][str(sid)] = { # Ensure species ID is string for JSON
                    "count": cnt,
                    "avg_fitness": avg_fit_species,
                    "traits": traits
                }
        return stats

    def get_top_performers(self, n: int = 3, sort_key: str = "fitness") -> List[NeuralAgent]:
        """Return the top-n agents sorted by ‘sort_key’ (“fitness” or “accuracy”)."""
        if not self.agents:
            return []

        if sort_key == "accuracy":
            return sorted(
                self.agents,
                key=lambda ag: np.mean(list(ag.classification_accuracy)) if ag.classification_accuracy else 0.0,
                reverse=True
            )[:n]
        
        # Default: sort by arbitrary attribute (e.g. 'fitness', 'energy', etc.)
        # Ensure attribute exists and provide a default if not, for robustness
        return sorted(self.agents, key=lambda ag: getattr(ag, sort_key, 0.0), reverse=True)[:n]


    def visualize_advanced_evolution(self):
        """
        Display a multi-panel figure showing time-series of:
         - avg_fitness vs max_fitness
         - population size & species diversity
         - avg_age vs avg_stress
         - avg_cooperation vs avg_aggression
         - current fitness distribution
         - current age distribution
         - environmental conditions (resources & pressure)
         - avg_specialization
        """
        if len(self.avg_fitness_history) < 2:
            print("Not enough data to visualize.")
            return

        fig, axes = plt.subplots(4, 2, figsize=(18, 24))
        fig.suptitle(f"Advanced Neural Ecosystem Evolution – Step {self.time_step}", fontsize=18, y=0.99)

        xs = np.arange(len(self.avg_fitness_history))

        # 1) Avg vs Max Fitness
        ax1 = axes[0, 0]
        ax1.plot(xs, self.avg_fitness_history, label="Avg Fitness", linewidth=2)
        ax1.plot(xs, self.max_fitness_history, label="Max Fitness", linestyle='--', linewidth=2)
        ax1.set_title("Fitness Over Time", fontsize=14)
        ax1.set_xlabel("Time Steps", fontsize=12)
        ax1.set_ylabel("Fitness", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.7)

        # 2) Population vs Species Diversity (dual y)
        ax2 = axes[0, 1]
        ax2.plot(xs, self.population_history, color='green', label="Population Size", linewidth=2)
        ax2.set_xlabel("Time Steps", fontsize=12)
        ax2.set_ylabel("Population Size", color='green', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='green')

        ax2b = ax2.twinx()
        ax2b.plot(xs, self.diversity_history, color='purple', linestyle=':', label="Num Species", linewidth=2)
        ax2b.set_ylabel("Number of Species", color='purple', fontsize=12)
        ax2b.tick_params(axis='y', labelcolor='purple')

        ax2.set_title("Population & Species Diversity", fontsize=14)
        lines1, labs1 = ax2.get_legend_handles_labels()
        lines2, labs2 = ax2b.get_legend_handles_labels()
        ax2b.legend(lines1 + lines2, labs1 + labs2, loc='upper center', fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.7)

        # 3) Avg Age vs Avg Stress (dual y)
        ax3 = axes[1, 0]
        ax3.plot(xs, self.avg_age_history, color='brown', label="Avg Age", linewidth=2)
        ax3.set_xlabel("Time Steps", fontsize=12)
        ax3.set_ylabel("Avg Age", color='brown', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='brown')

        ax3b = ax3.twinx()
        ax3b.plot(xs, self.avg_stress_history, color='red', linestyle=':', label="Avg Stress", linewidth=2)
        ax3b.set_ylabel("Avg Stress (0–1)", color='red', fontsize=12)
        ax3b.tick_params(axis='y', labelcolor='red')

        ax3.set_title("Average Age & Stress", fontsize=14)
        lines3, labs3 = ax3.get_legend_handles_labels()
        lines3b, labs3b = ax3b.get_legend_handles_labels()
        ax3b.legend(lines3 + lines3b, labs3 + labs3b, loc='center right', fontsize=10)
        ax3.grid(True, linestyle=':', alpha=0.7)

        # 4) Avg Cooperation vs Avg Aggression
        ax4 = axes[1, 1]
        ax4.plot(xs, self.avg_cooperation_tendency_history, color='skyblue', label="Avg Cooperation", linewidth=2)
        ax4.plot(xs, self.avg_aggression_history, color='orangered', linestyle='--', label="Avg Aggression", linewidth=2)
        ax4.set_title("Average Behavioral Traits", fontsize=14)
        ax4.set_xlabel("Time Steps", fontsize=12)
        ax4.set_ylabel("Value (0–1)", fontsize=12)
        ax4.set_ylim(0.0, max(1.0, np.max(self.avg_aggression_history) if self.avg_aggression_history else 1.0) * 1.1 ) # Adjust y-limit if aggression can exceed 1
        ax4.legend(fontsize=10)
        ax4.grid(True, linestyle=':', alpha=0.7)

        # 5) Current Fitness Distribution (histogram)
        ax5 = axes[2, 0]
        if self.agents:
            fitness_vals = [ag.fitness for ag in self.agents if np.isfinite(ag.fitness)]
            if fitness_vals:
                ax5.hist(fitness_vals, bins=20, color='teal', alpha=0.7, density=False, edgecolor='black')
            ax5.set_title(f"Current Fitness Distribution (N={len(self.agents)})", fontsize=14)
            ax5.set_xlabel("Fitness", fontsize=12)
            ax5.set_ylabel("Count", fontsize=12)
        else:
            ax5.text(0.5, 0.5, "No agents to display", ha='center', va='center')
            ax5.set_title("Current Fitness Distribution")
        ax5.grid(True, linestyle=':', alpha=0.7)


        # 6) Current Age Distribution (histogram)
        ax6 = axes[2, 1]
        if self.agents:
            age_vals = [ag.age for ag in self.agents]
            bins = max(1, min(20, len(set(age_vals)))) if age_vals else 1
            ax6.hist(age_vals, bins=bins, color='gold', alpha=0.7, edgecolor='black')
            ax6.set_title("Current Age Distribution", fontsize=14)
            ax6.set_xlabel("Age", fontsize=12)
            ax6.set_ylabel("Count", fontsize=12)
        else:
            ax6.text(0.5, 0.5, "No agents to display", ha='center', va='center')
            ax6.set_title("Current Age Distribution")
        ax6.grid(True, linestyle=':', alpha=0.7)


        # 7) Environmental Conditions: resources & pressure
        ax7 = axes[3, 0]
        ax7.plot(xs, self.resource_history, color='limegreen', label="Resource Availability", linewidth=2)
        ax7.plot(xs, self.pressure_history, color='tomato', linestyle='--', label="Environmental Pressure", linewidth=2)
        ax7.set_title("Environmental Conditions", fontsize=14)
        ax7.set_xlabel("Time Steps", fontsize=12)
        ax7.set_ylabel("Factor Value", fontsize=12)
        ax7.legend(fontsize=10)
        ax7.grid(True, linestyle=':', alpha=0.7)

        # 8) Average Specialization Score
        ax8 = axes[3, 1]
        ax8.plot(xs, self.avg_specialization_history, color='purple', label="Avg Specialization", linewidth=2)
        ax8.set_title("Average Specialization Score", fontsize=14)
        ax8.set_xlabel("Time Steps", fontsize=12)
        ax8.set_ylabel("Specialization (-1..+1)", fontsize=12)
        if self.avg_specialization_history:
            ymin = min(-1.0, min(self.avg_specialization_history) - 0.1) if self.avg_specialization_history else -1.1
            ymax = max(1.0, max(self.avg_specialization_history) + 0.1) if self.avg_specialization_history else 1.1
            ax8.set_ylim(ymin, ymax)
        ax8.legend(fontsize=10)
        ax8.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        # plt.show() # In a script, plt.show() is blocking. For server, save to BytesIO.
                      # Assuming this is run as a script for now. If used in Flask app, this needs adjustment.


# --- Main Evolution Runner with Persistence ---

PERSISTENCE_FILENAME = "best_population.pkl"
PERSISTENCE_PATH = Path(__file__).resolve().parent / PERSISTENCE_FILENAME
ELITE_SAVE_COUNT = 50  # number of top agents to retain for next run


def run_advanced_evolution_experiment(
    steps: int = 1000,
    population_size: int = 50,
    verbose_interval: int = 100
) -> AdvancedEvolutionEnvironment:
    """
    Launch or resume an evolutionary run of ‘steps’ time steps.
    If a persistence file exists, load it (either as an environment or as a seed list).
    After completion, save the pruned environment (top ELITE_SAVE_COUNT agents).
    """
    print("🌱 ADVANCED NEURAL EVOLUTION v3.0 🌱")
    print("Initializing evolutionary ecosystem...")

    ecosystem: Optional[AdvancedEvolutionEnvironment] = None
    seed_agents: Optional[List[NeuralAgent]] = None

    # Attempt to load previous state
    if PERSISTENCE_PATH.exists():
        try:
            with PERSISTENCE_PATH.open("rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, AdvancedEvolutionEnvironment):
                ecosystem = loaded
                # When loading a full ecosystem, update its internal _next_agent_id_counter
                # to be higher than any ID currently in the loaded ecosystem to prevent reuse.
                max_existing_id = 0
                if ecosystem.agents:
                    max_existing_id = max(ag.id for ag in ecosystem.agents)
                    for ag in ecosystem.agents: # Check lineage as well
                        if ag.lineage:
                             max_lineage_id = max(ag.lineage)
                             if max_lineage_id > max_existing_id:
                                 max_existing_id = max_lineage_id
                AdvancedEvolutionEnvironment._next_agent_id_counter = max(
                    AdvancedEvolutionEnvironment._next_agent_id_counter, 
                    max_existing_id + 1
                )
                ecosystem._migrate_histories()
                if not ecosystem.agents:
                    print(f"⚠️  Loaded ecosystem from '{PERSISTENCE_FILENAME}' but it has 0 agents (extinct). Starting fresh.")
                    ecosystem = None
                else:
                    print(f"✅ Loaded ecosystem from '{PERSISTENCE_FILENAME}' at step {ecosystem.time_step}.")
            elif isinstance(loaded, list) and all(isinstance(a, NeuralAgent) for a in loaded):
                seed_agents = loaded
                if not seed_agents:
                    print(f"⚠️  Loaded agent list from '{PERSISTENCE_FILENAME}' but it is empty. Starting fresh.")
                    seed_agents = None
                else:
                    max_seed_id = 0
                    max_seed_id = max(ag.id for ag in seed_agents)
                    for ag in seed_agents:
                        if ag.lineage:
                             max_lineage_id = max(ag.lineage)
                             if max_lineage_id > max_seed_id:
                                 max_seed_id = max_lineage_id
                    AdvancedEvolutionEnvironment._next_agent_id_counter = max(
                        AdvancedEvolutionEnvironment._next_agent_id_counter,
                        max_seed_id + 1
                    )
                    print(f"✅ Loaded {len(seed_agents)} seed agents from '{PERSISTENCE_FILENAME}'.")
            else:
                print(f"⚠️  File '{PERSISTENCE_FILENAME}' did not contain valid data; starting fresh.")
        except Exception as e:
            print(f"⚠️  Error loading '{PERSISTENCE_FILENAME}': {e}. Starting new run.")

    # Create environment if not loaded
    if ecosystem is None:
        ecosystem = AdvancedEvolutionEnvironment(
            population_size=population_size,
            seed_agents=seed_agents # seed_agents will be None if new run, or list from PKL
        )
        print(f"🚀 STARTING EVOLUTION – {steps} steps, population={ecosystem.population_history[-1] if ecosystem.population_history else len(ecosystem.agents)}")
    else:
        # If ecosystem was loaded, it might have a different population size (ELITE_SAVE_COUNT)
        # than the 'population_size' parameter of this function.
        # The simulation continues with the loaded population size.
        print(f"🚀 RESUMING EVOLUTION – {steps} more steps (continuing from step {ecosystem.time_step})")
        print(f"   Current population from PKL: {len(ecosystem.agents)}")


    print("=" * 70)
    start_time = time.time()

    # Main simulation loop
    for i in range(steps): # loop 'steps' times
        if not ecosystem.run_advanced_simulation_step():
            print(f"💀 ECOSYSTEM EXTINCT at step {ecosystem.time_step}!")
            break

        if verbose_interval > 0 and (ecosystem.time_step % verbose_interval == 0):
            stats = ecosystem.get_advanced_ecosystem_stats()
            if stats.get("status") != "extinct":
                print(
                    f"\n--- Step {ecosystem.time_step} --- "
                    f"Env={stats['environment_type']} "
                    f"(P={stats['environmental_pressure']:.2f}, "
                    f"R={stats['resource_availability']:.2f})\n"
                    f"Pop={stats['population']}, Species={stats['num_species']}, "
                    f"AvgFit={stats['avg_fitness']:.2f}, MaxFit={stats['max_fitness']:.2f}\n"
                    f"AvgAge={stats['avg_age']:.1f}, AvgCoop={stats['avg_cooperation']:.2f}, "
                    f"AvgAggr={stats['avg_aggression']:.2f}, AvgStress={stats['avg_stress']:.2f}"
                )
                top_performers_list = ecosystem.get_top_performers(1) # Renamed variable
                if top_performers_list: # Check if list is not empty
                    top_agent = top_performers_list[0]
                    print(
                        f"👑 Champion: ID {top_agent.id}, Spec {top_agent.species_id}, Gen {top_agent.generation}\n"
                        f"    Fitness={top_agent.fitness:.2f}, Age={top_agent.age}, Energy={top_agent.energy:.1f}, "
                        f"Offspring={top_agent.offspring_count}\n"
                        f"    Arch={top_agent.genome.layers}, "
                        f"Coop={top_agent.genome.cooperation_tendency:.2f}, "
                        f"Aggr={top_agent.genome.aggression:.2f}"
                    )
                else:
                    print("👑 No agents available to determine a champion.")
            else:
                print(f"💀 ECOSYSTEM EXTINCT at step {ecosystem.time_step} (by stats)!")
                break
        if i == steps -1 and ecosystem.time_step % verbose_interval !=0 : # Log last step if not caught by verbose
             stats = ecosystem.get_advanced_ecosystem_stats()
             print(f"--- Final Step of this run: {ecosystem.time_step} --- Pop={stats['population']}, AvgFit={stats['avg_fitness']:.2f}, MaxFit={stats['max_fitness']:.2f}")


    elapsed = time.time() - start_time
    print(f"\n🕒 Evolution finished in {elapsed:.2f}s ({ecosystem.time_step} total steps in ecosystem).")
    print("=" * 70)
    print("🧬 FINAL ECOSYSTEM STATE (at end of this run):")

    final_stats = ecosystem.get_advanced_ecosystem_stats()
    for key, val in final_stats.items():
        if key == "species_details":
            print(f"  Species Details ({len(val)} active):")
            # Sort species by count for display
            sorted_species = sorted(val.items(), key=lambda item: item[1]['count'], reverse=True) # item[0] is sid, item[1] is details dict
            for sid_str, sd in sorted_species[:min(5, len(sorted_species))]: # sid is string from JSON conversion
                print(f"    ID={sid_str}: Count={sd['count']}, AvgFit={sd['avg_fitness']:.2f}, Traits={sd['traits']}")
            if len(val) > 5:
                print("    ... (more species exist)")
        elif isinstance(val, float):
            print(f"  {key.replace('_',' ').capitalize()}: {val:.3f}")
        else:
            print(f"  {key.replace('_',' ').capitalize()}: {val}")

    print("\n🏆 TOP 3 PERFORMERS (by Fitness, at end of this run):")
    top3 = ecosystem.get_top_performers(3, sort_key="fitness")
    if top3:
        for rank, ag in enumerate(top3, start=1):
            acc = np.mean(list(ag.classification_accuracy)) if ag.classification_accuracy else 0.0
            print(
                f"  #{rank}: Agent {ag.id} (Species {ag.species_id}, Gen {ag.generation})\n"
                f"      Fit={ag.fitness:.2f}, Energy={ag.energy:.1f}, Age={ag.age}\n"
                f"      Genome: Layers={ag.genome.layers}, LR={ag.genome.learning_rate:.4f}\n"
                f"              Coop={ag.genome.cooperation_tendency:.2f}, "
                f"Aggr={ag.genome.aggression:.2f}, StressRes={ag.genome.stress_resistance:.2f}\n"
                f"      Reputation={ag.reputation:.2f}, Accuracy(last {len(ag.classification_accuracy)})={acc:.2f}"
            )
    else:
        print("  No agents available to list top performers.")


    # Prune to elites before saving
    if ecosystem.agents: # Only prune if there are agents
        sorted_final = sorted(ecosystem.agents, key=lambda a: a.fitness, reverse=True)
        elites_to_save = sorted_final[:ELITE_SAVE_COUNT] # Renamed variable
        ecosystem.agents = elites_to_save

        # Recompute species_populations to reflect pruned set
        new_counts = defaultdict(int)
        for ag_elite in elites_to_save: # Renamed variable
            new_counts[ag_elite.species_id] += 1
        ecosystem.species_populations = new_counts

        # Clear species_fitness & traits to avoid stale data based on non-elite population
        ecosystem.species_fitness = defaultdict(list)
        ecosystem.species_traits_avg = defaultdict(lambda: defaultdict(float))
        # Optionally, re-calculate these for the elite set if needed for immediate stats post-save,
        # but they will be rebuilt on next run's _record_step_statistics anyway.
        # For now, this clearing is consistent with previous logic.
        print(f"\nPruned ecosystem to {len(ecosystem.agents)} elite agents for saving.")
    else:
        print("\nNo agents to prune or save.")


    # Persist environment for next run (skip if extinct to preserve previous save)
    if ecosystem.agents:
        try:
            with PERSISTENCE_PATH.open("wb") as f:
                pickle.dump(ecosystem, f)
            print(f"💾 Saved ecosystem state to '{PERSISTENCE_FILENAME}'.")
        except Exception as e:
            print(f"⚠️  Failed to save ecosystem state: {e}")
    else:
        print("⚠️  Population is extinct — skipping save to preserve previous elite state.")

    return ecosystem


if __name__ == "__main__":
    steps_to_run = 100  # Set this to your desired number of steps for this run

    if PERSISTENCE_PATH.exists():
        try:
            with PERSISTENCE_PATH.open("rb") as f:
                ecosystem = pickle.load(f)
            print(f"✅ Loaded ecosystem from '{PERSISTENCE_FILENAME}' at step {ecosystem.time_step}.")
        except Exception as e:
            print(f"⚠️  Failed to load existing PKL file: {e}. Starting new ecosystem.")
            ecosystem = AdvancedEvolutionEnvironment(population_size=500)
    else:
        ecosystem = AdvancedEvolutionEnvironment(population_size=500)
        print("🚀 Created new ecosystem.")

    print(f"Starting simulation from step {ecosystem.time_step} for {steps_to_run} more steps.")

    for step in range(steps_to_run):
        ecosystem.run_advanced_simulation_step()

    print(f"Simulation completed up to step {ecosystem.time_step}.")

    # Prune ecosystem before saving (retain top ELITE_SAVE_COUNT agents)
    ecosystem.agents.sort(key=lambda a: a.fitness, reverse=True)
    ecosystem.agents = ecosystem.agents[:ELITE_SAVE_COUNT]

    # Update species populations after pruning
    ecosystem.species_populations = defaultdict(int)
    for agent in ecosystem.agents:
        ecosystem.species_populations[agent.species_id] += 1

    try:
        with PERSISTENCE_PATH.open("wb") as f:
            pickle.dump(ecosystem, f)
        print(f"💾 Ecosystem state saved to '{PERSISTENCE_FILENAME}'.")
    except Exception as e:
        print(f"⚠️  Failed to save ecosystem state: {e}")

    # Optionally visualize
    ecosystem.visualize_advanced_evolution()
