import math

import torch
from scoda.tokenisation.base_tokenisation import BaseTokeniser
from torch import nn

from paul3.management.manager_base import BaseManager
from paul3.utils.inference_constrainer import InferenceConstrainer
from paul3.utils.paul_logging import get_logger

LOGGER = get_logger(__name__)


class InferenceManager(BaseManager):
    STOP_TOKEN = 2

    def __init__(self, model: nn.Module,
                 model_general_settings: dict,
                 model_hyperparameters: dict,
                 inference_settings: dict,
                 tokeniser: BaseTokeniser,
                 device: torch.device,
                 log_frequency: int = 1,
                 constrainer: InferenceConstrainer = None,
                 approach_settings: dict = None):
        super().__init__(model, model_general_settings, model_hyperparameters, {}, inference_settings, device)

        self.tokeniser = tokeniser
        self.max_len = self.inference_settings.get("max_len")
        self.temperature = self.inference_settings.get("temperature")

        self.log_frequency = log_frequency
        self.constrainer = constrainer
        self.approach_settings = approach_settings

    def inference(self, input_dict: dict):
        self.model = self.model.to(self.device)
        self.model.eval()

        # Move input to device
        input_dict = self.move_to_device(input_dict, self.device)

        with torch.no_grad():
            approach = "beam_search"
            if approach == "beam_search":
                completed_states, completed_scores = self.approach_nd_beam_search(input_dict,
                                                                                  self.max_len,
                                                                                  temperature=self.temperature,
                                                                                  constrainer=self.constrainer,
                                                                                  **self.approach_settings[
                                                                                      "beam_search"])
            elif approach == "base":
                completed_states, completed_scores = self.approach_base(input_dict,
                                                                        self.max_len,
                                                                        self.temperature,
                                                                        constrainer=self.constrainer), [None]

        return [completed_states[0]], [completed_scores[0]]

    def _shared_model_pass(self, i_step, state, temperature: float, constrainer: InferenceConstrainer):
        # Prepare input
        step_input = {"state": state}

        # Run inference
        step_logits = self.model(**step_input)

        # Constrain output
        if constrainer is not None:
            step_logits = constrainer.constrain(state, step_logits)

        # Apply temperature and calculate probabilities
        step_logits /= temperature
        step_probabilities = torch.softmax(step_logits, dim=-1)

        # Check for NaN or infinite values
        if torch.isnan(step_probabilities).any() or torch.isinf(step_probabilities).any():
            LOGGER.error("Numerical instability detected")

        return step_probabilities

    def approach_base(self,
                      input_dict: dict,
                      max_len: int,
                      temperature: float,
                      constrainer: InferenceConstrainer):
        # Initialise current states and scores
        state = input_dict["state"]

        # Initialise step counter
        i_step = len(input_dict["state"][-1]) - 1

        while True and i_step < max_len - 1:
            # Get probabilities
            step_probabilities = self._shared_model_pass(i_step, state, temperature, constrainer)[:, i_step, :]

            # Flatten probabilities
            step_probabilities_flat = torch.reshape(step_probabilities, (-1,))

            # Draw predictions
            step_prediction = torch.multinomial(step_probabilities_flat, 1)

            # Add the next token to the state
            state = torch.cat([state, step_prediction.unsqueeze(0)], dim=-1)

            if step_prediction == self.STOP_TOKEN:
                break

            if i_step % self.log_frequency == 0 or i_step == 0:
                LOGGER.info(
                    f"Completed step {i_step + 1}.")

            i_step += 1

        LOGGER.info(
            f"Completed base search.")

        return state.tolist()

    def approach_nd_beam_search(self,
                                input_dict: dict,
                                max_len: int,
                                beam_size: int,
                                batch_size: int,
                                temperature: float,
                                normalisation_coefficient: float,
                                constrainer: InferenceConstrainer):
        # Initialise current states and scores
        states = torch.cat([input_dict["state"]])
        scores = torch.cat([torch.tensor([0], device=self.device)])

        # Initialise final states and scores
        completed_states = []
        completed_scores = []

        # Initialise step counter
        i_step = len(input_dict["state"][-1]) - 1

        # Start beam search
        while True and i_step < max_len:
            # Normalisation factor according to https://doi.org/10.48550/arXiv.1609.08144
            normalisation_lp = (math.pow(5 + i_step + 2, normalisation_coefficient) /
                                math.pow(5 + 1, normalisation_coefficient))

            # Split states and scores into batches
            states_batches = torch.split(states, batch_size)
            scores_batches = torch.split(scores, batch_size)
            step_batch_probabilities_list = []

            # Run inference for each batch
            for states_batch, scores_batch in zip(states_batches, scores_batches):
                # Get probabilities and add to list
                step_batch_probabilities = self._shared_model_pass(i_step, states_batch, temperature, constrainer)
                step_batch_probabilities_list.append(step_batch_probabilities)

            # Combine probabilities from batches
            step_probabilities = torch.cat(step_batch_probabilities_list)[:, i_step, :]
            step_probabilities_flat = torch.reshape(step_probabilities, (-1,))

            # Calculate scores
            step_scores = torch.log(step_probabilities) / normalisation_lp + scores.unsqueeze(-1)
            step_scores_flat = torch.reshape(step_scores, (-1,))

            # Retrieve k candidates from probability distribution across all possible candidates
            top_k_states = torch.multinomial(step_probabilities_flat, beam_size, replacement=False)
            top_k_scores = step_scores_flat[top_k_states]

            # Retrieve which candidates the tokens stem from and which tokens were chosen
            i_selected_states = top_k_states // step_probabilities.size(-1)
            i_selected_tokens = top_k_states % step_probabilities.size(-1)

            # Add token predictions to current states
            states = torch.cat([states[i_selected_states], i_selected_tokens.unsqueeze(-1)], dim=-1)
            scores = top_k_scores

            # Check which candidates are completed
            completed_candidate_indices = i_selected_tokens == self.STOP_TOKEN
            completed_states.extend(states[completed_candidate_indices].tolist())
            completed_scores.extend(scores[completed_candidate_indices].tolist())

            # Stop beam search if enough sequences reach end
            if len(completed_states) >= beam_size:
                break

            states = states[~completed_candidate_indices]
            scores = scores[~completed_candidate_indices]

            # Add candidates to completed if max len exceeded
            if i_step == max_len - 1:
                completed_states.extend(states.tolist())
                completed_scores.extend(scores.tolist())

            if i_step % self.log_frequency == 0 or i_step == 0:
                LOGGER.info(
                    f"Completed step {i_step + 1}. Solution size: {len(completed_states)}. Solution scores: {completed_scores}")

            i_step += 1

        LOGGER.info(
            f"Completed beam search. Solution size: {len(completed_states)}. Solution scores: {completed_scores}")

        return completed_states, completed_scores
