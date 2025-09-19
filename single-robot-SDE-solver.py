import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
from copy import copy

import dill as pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import epde
from epde.integrate.pinn_integration import SolverAdapter
from scipy import stats
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Configure logging to provide detailed execution information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RobotDataProcessor:
    """Handles loading, processing, and normalizing robot movement data."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.raw_data = None
        self.coord_data: Dict[int, List] = {}
        self.normalized_coord_data: Dict[int, np.ndarray] = {}
        self.keys_list: List[int] = []

    def load_data(self, pickle_filename: str) -> None:
        """Loads data from a specified pickle file."""
        pickle_file = self.data_path / pickle_filename
        try:
            with open(pickle_file, 'rb') as f:
                self.raw_data = pickle.load(f)
            logger.info(f"Successfully loaded data from {pickle_file}")
        except FileNotFoundError:
            logger.error(f"File not found: {pickle_file}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while loading data: {e}")
            raise

    def extract_coordinates(self) -> None:
        """Extracts coordinate data from the loaded raw data structure."""
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.keys_list = [inner_list[0] for inner_list in self.raw_data[0]]
        for inner_list in self.raw_data:
            for item in inner_list:
                key, _, coordinates = item
                if key in self.coord_data:
                    self.coord_data[key].append(coordinates)
                else:
                    self.coord_data[key] = [coordinates]

    @staticmethod
    def _normalize_to_normal(data: np.ndarray) -> np.ndarray:
        """
        Normalizes data by applying a Box-Cox transformation followed by Min-Max scaling.
        This helps to stabilize variance and make the data more normal-distribution-like.
        """
        # Ensure data is positive for Box-Cox transformation
        if np.min(data) <= 0:
            data = data - np.min(data) + 1e-6

        transformed_data, _ = stats.boxcox(data)
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(transformed_data.reshape(-1, 1)).flatten()

    def normalize_all_coordinates(self) -> None:
        """Normalizes all extracted x and y coordinates."""
        all_x = [point[0] for points_list in self.coord_data.values() for point in points_list]
        all_y = [point[1] for points_list in self.coord_data.values() for point in points_list]

        x_normalized = self._normalize_to_normal(np.array(all_x))
        y_normalized = self._normalize_to_normal(np.array(all_y))

        current_idx = 0
        for obj_id, points in self.coord_data.items():
            num_points = len(points)
            obj_x_norm = x_normalized[current_idx:current_idx + num_points]
            obj_y_norm = y_normalized[current_idx:current_idx + num_points]

            self.normalized_coord_data[obj_id] = np.column_stack((obj_x_norm, obj_y_norm))
            current_idx += num_points

    def get_trajectory(self, object_id: int, frequency: int = 1, max_points: int = 2750) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Retrieves the trajectory for a specific object."""
        if object_id not in self.normalized_coord_data:
            raise ValueError(f"Object with ID {object_id} not found.")

        coords = self.normalized_coord_data[object_id][:max_points:frequency]
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        time_points = np.arange(len(coords))
        return x_coords, y_coords, time_points

    @staticmethod
    def split_trajectory(t: np.ndarray, x: np.ndarray, y: np.ndarray, num_segments: int = 3) -> List[
        Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Splits a trajectory into a specified number of segments."""
        total_points = len(t)
        segment_size = total_points // num_segments
        segments = []
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < num_segments - 1 else total_points
            segments.append((t[start_idx:end_idx], x[start_idx:end_idx], y[start_idx:end_idx]))
        return segments


class EquationAnalyzer:
    """Loads and analyzes equation coefficients from CSV files."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.monte_carlo_samples: pd.DataFrame = pd.DataFrame()

    def load_equation_data(self, csv_filenames: List[str]) -> None:
        """Loads and concatenates equation data from multiple CSV files."""
        dfs = []
        for filename in csv_filenames:
            csv_path = self.data_path / filename
            try:
                df = pd.read_csv(csv_path, index_col='Unnamed: 0').drop(columns=['count'], errors='ignore').fillna(0)
                dfs.append(df)
                logger.info(f"Successfully loaded equation data from {csv_path}")
            except FileNotFoundError:
                logger.error(f"Equation file not found: {csv_path}")
                raise

        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined equation data shape: {self.df.shape}")
        else:
            raise ValueError("No equation data found to concatenate.")

    def analyze_equation_patterns(self) -> None:
        """Analyzes patterns in the equation coefficients."""
        if self.df.empty:
            raise ValueError("Equation data is not loaded.")

        most_common_structure = self.df.groupby(self.df.columns.tolist()).size().reset_index(
            name='count').sort_values(by='count', ascending=False)
        logger.info(f"Most common equation structure:\n{most_common_structure.head(1)}")

        zero_patterns = (self.df == 0).astype(int).astype(str).agg(''.join, axis=1)
        pattern_counts = zero_patterns.value_counts()
        logger.info(f"Most common zero-coefficient pattern: {pattern_counts.idxmax()} "
                    f"(appears {pattern_counts.max()} times)")

    def normalize_and_filter_equations(self) -> None:
        """Normalizes and filters the equation data, handling columns with zero variance."""
        scaler = StandardScaler()
        # Find columns with non-zero variance
        valid_cols = self.df.columns[(self.df != 0).any(axis=0)]
        if valid_cols.empty:
            logger.warning("All columns have zero variance after initial filtering. Skipping scaling.")
            return

        # Filter the DataFrame to only include these columns
        self.df = self.df[valid_cols]

        scaled_values = scaler.fit_transform(self.df.values)
        normalized_df = pd.DataFrame(scaled_values, columns=self.df.columns, index=self.df.index).fillna(0)

        # Re-filter based on the specified condition
        self.df = normalized_df.loc[(normalized_df.sum(axis=1) != -1)]
        logger.info(f"Filtered equation data shape: {self.df.shape}")

    def generate_monte_carlo_samples(self, num_samples: int = 100) -> pd.DataFrame:
        """Generates new coefficient samples using a Monte Carlo method."""
        samples_df = pd.DataFrame(np.zeros((num_samples, len(self.df.columns))),
                                  columns=self.df.columns)

        for column in self.df.columns:
            mean_val, std_dev = self.df[column].mean(), self.df[column].std()
            min_val, max_val = self.df[column].min(), self.df[column].max()

            for i in range(num_samples):
                current_mean = self.df[column].iloc[i % len(self.df)]
                noise = np.random.normal(loc=0, scale=std_dev * 0.01)
                sample = np.clip(current_mean + noise, min_val, max_val)
                samples_df.loc[i, column] = sample

        logger.info(f"Generated Monte Carlo samples with shape {samples_df.shape}")
        return samples_df

    def find_similar_coefficients(self, samples_df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
        """Finds generated coefficients that are similar to the original ones."""
        if self.df.empty or samples_df.empty:
            raise ValueError("DataFrames for comparison are empty.")

        distances = pairwise_distances(self.df, samples_df)
        similar_rows_indices = np.where(distances < threshold)[1]
        similar_coefficients_df = samples_df.iloc[np.unique(similar_rows_indices)]

        logger.info(f"Found {len(similar_coefficients_df)} similar coefficient sets.")
        return similar_coefficients_df


class EquationConstructor:
    """
    Constructs equation dictionaries from a DataFrame of coefficients.
    """

    @staticmethod
    def _create_equation_term(row: pd.Series) -> Dict:
        """Helper function to create a single equation dictionary from a row of coefficients."""
        return {
            # --- Terms for X equation ---
            'C_x': {'coeff': row.get('C_x', 0), 'term': [[None]], 'pow': [1], 'var': [0]},
            'x_x': {'coeff': row.get('x{power: 1.0}_x', 0), 'term': [[None]], 'pow': [1], 'var': [0]},
            'y_x': {'coeff': row.get('y{power: 1.0}_x', 0), 'term': [[None]], 'pow': [1], 'var': [1]},
            'dx/dt_x': {'coeff': row.get('dx/dx0{power: 1.0}_x', 0), 'term': [[0]], 'pow': [1], 'var': [0]},
            'd^2x/dt^2_x': {'coeff': row.get('d^2x/dx0^2{power: 1.0}_x', 0), 'term': [[0, 0]], 'pow': [1], 'var': [0]},
            'dy/dt_x': {'coeff': row.get('dy/dx0{power: 1.0}_x', 0), 'term': [[0]], 'pow': [1], 'var': [1]},
            'd^2y/dt^2_x': {'coeff': row.get('d^2y/dx0^2{power: 1.0}_x', 0), 'term': [[0, 0]], 'pow': [1], 'var': [1]},
            't_x': {'coeff': row.get('t{power: 1.0, dim: 0.0}_x', 0), 'term': [[None]], 'pow': [1], 'var': [0],
                    'grid_term': True},
            't^2_x': {'coeff': row.get('t{power: 2.0, dim: 0.0}_x', 0), 'term': [[None]], 'pow': [2], 'var': [0],
                      'grid_term': True},

            # --- Terms for Y equation ---
            'C_y': {'coeff': row.get('C_y', 0), 'term': [[None]], 'pow': [1], 'var': [0]},
            'x_y': {'coeff': row.get('x{power: 1.0}_y', 0), 'term': [[None]], 'pow': [1], 'var': [0]},
            'y_y': {'coeff': row.get('y{power: 1.0}_y', 0), 'term': [[None]], 'pow': [1], 'var': [1]},
            'dx/dt_y': {'coeff': row.get('dx/dx0{power: 1.0}_y', 0), 'term': [[0]], 'pow': [1], 'var': [0]},
            'd^2x/dt^2_y': {'coeff': row.get('d^2x/dx0^2{power: 1.0}_y', 0), 'term': [[0, 0]], 'pow': [1], 'var': [0]},
            'dy/dt_y': {'coeff': row.get('dy/dx0{power: 1.0}_y', 0), 'term': [[0]], 'pow': [1], 'var': [1]},
            'd^2y/dt^2_y': {'coeff': row.get('d^2y/dx0^2{power: 1.0}_y', 0), 'term': [[0, 0]], 'pow': [1], 'var': [1]},
            't_y': {'coeff': row.get('t{power: 1.0, dim: 0.0}_y', 0), 'term': [[None]], 'pow': [1], 'var': [0],
                    'grid_term': True},
            't^2_y': {'coeff': row.get('t{power: 2.0, dim: 0.0}_y', 0), 'term': [[None]], 'pow': [2], 'var': [0],
                      'grid_term': True},

            # --- Composite terms for X equation ---
            'x*dx/dt_x': {'coeff': row.get('x{power: 1.0} * dx/dx0{power: 1.0}_x', 0), 'term': [[None], [0]],
                          'pow': [1, 1], 'var': [0, 0]},
            'y*dy/dt_x': {'coeff': row.get('y{power: 1.0} * dy/dx0{power: 1.0}_x', 0), 'term': [[None], [0]],
                          'pow': [1, 1], 'var': [1, 1]},
            'dx/dt*dy/dt_x': {'coeff': row.get('dx/dx0{power: 1.0} * dy/dx0{power: 1.0}_x', 0), 'term': [[0], [0]],
                              'pow': [1, 1], 'var': [0, 1]},
            'dx/dt*d^2x/dt^2_x': {'coeff': row.get('dx/dx0{power: 1.0} * d^2x/dx0^2{power: 1.0}_x', 0),
                                  'term': [[0], [0, 0]], 'pow': [1, 1], 'var': [0, 0]},
            'dx/dt*d^2y/dt^2_x': {'coeff': row.get('dx/dx0{power: 1.0} * d^2y/dx0^2{power: 1.0}_x', 0),
                                  'term': [[0], [0, 0]], 'pow': [1, 1], 'var': [0, 1]},
            'dx/dt*t_x': {'coeff': row.get('dx/dx0{power: 1.0} * t{power: 1.0, dim: 0.0}_x', 0), 'term': [[0], [None]],
                          'pow': [1, 1], 'var': [0, 0], 'grid_term': [False, True]},
            'd^2y/dt^2*t_x': {'coeff': row.get('d^2y/dx0^2{power: 1.0} * t{power: 1.0, dim: 0.0}_x', 0),
                              'term': [[0, 0], [None]], 'pow': [1, 1], 'var': [1, 0], 'grid_term': [False, True]},
            'd^2y/dt^2*sin_x': {'coeff': row.get('d^2y/dx0^2{power: 1.0} * sin{power: 1.0, dim: 0.0}_x', 0),
                                'term': [[0, 0], [None]], 'pow': [1, 1], 'var': [1, 0], 'like_grad': [False, True]},

            # --- Composite terms for Y equation ---
            'y*d^2y/dt^2_y': {'coeff': row.get('y{power: 1.0} * d^2y/dx0^2{power: 1.0}_y', 0), 'term': [[None], [0, 0]],
                              'pow': [1, 1], 'var': [1, 1]},
            'dy/dt*x_y': {'coeff': row.get('dy/dx0{power: 1.0} * x{power: 1.0}_y', 0), 'term': [[0], [None]],
                          'pow': [1, 1], 'var': [1, 0]},
            'dy/dt*dx/dt_y': {'coeff': row.get('dy/dx0{power: 1.0} * dx/dx0{power: 1.0}_y', 0), 'term': [[0], [0]],
                              'pow': [1, 1], 'var': [1, 0]},
            'dy/dt*d^2x/dt^2_y': {'coeff': row.get('dy/dx0{power: 1.0} * d^2x/dx0^2{power: 1.0}_y', 0),
                                  'term': [[0], [0, 0]], 'pow': [1, 1], 'var': [1, 0]},
            'd^2x/dt^2*d^2y/dt^2_y': {'coeff': row.get('d^2x/dx0^2{power: 1.0} * d^2y/dx0^2{power: 1.0}_y', 0),
                                      'term': [[0, 0], [0, 0]], 'pow': [1, 1], 'var': [0, 1]},
            't*d^2y/dt^2_y': {'coeff': row.get('t{power: 1.0, dim: 0.0} * d^2y/dx0^2{power: 1.0}_y', 0),
                              'term': [[None], [0, 0]], 'pow': [1, 1], 'var': [0, 1], 'grid_term': [True, False]},
            't^2*y_y': {'coeff': row.get('t{power: 2.0, dim: 0.0} * y{power: 1.0}_y', 0), 'term': [[None], [None]],
                        'pow': [2, 1], 'var': [0, 1], 'grid_term': [True, False]},
            'dx/dt*sin_y': {'coeff': row.get('dx/dx0{power: 1.0} * sin{power: 1.0, dim: 0.0}_y', 0),
                            'term': [[0], [None]], 'pow': [1, 1], 'var': [0, 0], 'like_grad': [False, True]},
            'dx/dt*cos_y': {'coeff': row.get('dx/dx0{power: 1.0} * cos{power: 1.0, dim: 0.0}_y', 0),
                            'term': [[0], [None]], 'pow': [1, 1], 'var': [0, 0], 'like_grad': [False, True]},
        }

    def construct_equation_systems(self, coefficients_df: pd.DataFrame) -> List[Tuple[Dict, Dict]]:
        """Constructs a list of equation system dictionaries from a DataFrame of coefficients."""
        equation_systems = []
        for _, row in coefficients_df.iterrows():
            full_equation = self._create_equation_term(row)

            eq_x = {key: val for key, val in full_equation.items() if '_x' in key and val['coeff'] != 0}
            eq_y = {key: val for key, val in full_equation.items() if '_y' in key and val['coeff'] != 0}

            equation_systems.append((eq_x, eq_y))
        return equation_systems


class ODESolver:
    """Solves systems of Ordinary Differential Equations using a PINN-based adapter."""

    def __init__(self):
        self.adapter = SolverAdapter()
        # Configure solver parameters for better performance
        self.adapter.change_parameter('epochs', 5000)
        self.adapter.change_parameter('optimizer', 'LBFGS')

    @staticmethod
    def create_boundary_operator(key: str, var_idx: int, term: Union[int, List], grid_loc: float, value: float) -> Dict[
        str, Any]:
        """Creates a boundary operator dictionary required by the solver."""
        if not isinstance(term, list):
            term = [term]
        bop = epde.integrate.BOPElement(axis=0, key=key, term=term, power=1, var=var_idx)
        bop.set_grid(torch.from_numpy(np.array([[grid_loc]])).float())
        bop.values = torch.from_numpy(np.array([[value]])).float()
        return {
            'bnd_loc': torch.Tensor([bop.location, ]),
            'bnd_op': {bop.operator_form[0]: bop.operator_form[1]},
            'bnd_val': bop.values
        }

    def solve_system(self, equations: Dict, grids: List, boundary_conditions: List) -> Any:
        """Solves a system of ODEs."""
        try:
            _, solution = self.adapter.solve_epde_system(
                system=equations,
                grids=grids,
                boundary_conditions=boundary_conditions,
                mode='NN',
                to_numpy=True,
                grid_var_keys=['t']
            )
            logger.info("ODE system solved successfully.")
            return solution
        except Exception as e:
            logger.error(f"Error while solving ODE system: {e}")
            return None


class SolutionManager:
    """Manages saving solutions and coefficients to files."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def save_solution(self, solution: np.ndarray, time_points: np.ndarray, object_id: int, part_idx: int) -> None:
        """Saves a trajectory segment solution to a CSV file."""
        df = pd.DataFrame({'t': time_points, 'x': solution[:, 0], 'y': solution[:, 1]})
        filename = self.output_path / f'solution_{object_id}_part_{part_idx + 1}.csv'
        df.to_csv(filename, index=False)
        logger.info(f"Solution segment saved to {filename}")

    def save_combined_solution(self, solutions: List[np.ndarray], time_points: np.ndarray, object_id: int) -> None:
        """Saves the combined trajectory solution to a CSV file."""
        combined_solution = np.vstack(solutions)
        df = pd.DataFrame({'t': time_points, 'x': combined_solution[:, 0], 'y': combined_solution[:, 1]})
        filename = self.output_path / f'solution_{object_id}_combined.csv'
        df.to_csv(filename, index=False)
        logger.info(f"Combined solution saved to {filename}")

    def save_equation_coefficients(self, equations: List[Tuple[Dict, Dict]], object_id: int) -> None:
        """
        Saves the coefficients of the discovered equations.
        Expects a list of tuples, where each tuple contains two equation dictionaries.
        """
        coeff_data = []
        for i, eq_tuple in enumerate(equations):
            eq_x, eq_y = eq_tuple[0], eq_tuple[1]

            # Process the X-equation
            for term_name, term_params in eq_x.items():
                coeff_data.append({
                    'equation_set': i + 1,
                    'equation_type': 'x',
                    'term_name': term_name,
                    'coefficient': term_params['coeff']
                })

            # Process the Y-equation
            for term_name, term_params in eq_y.items():
                coeff_data.append({
                    'equation_set': i + 1,
                    'equation_type': 'y',
                    'term_name': term_name,
                    'coefficient': term_params['coeff']
                })

        df = pd.DataFrame(coeff_data)
        filename = self.output_path / f'coefficients_{object_id}.csv'
        df.to_csv(filename, index=False)
        logger.info(f"Equation coefficients saved to {filename}")

    def save_numpy_and_tensor(self, data: np.ndarray, filename: str) -> None:
        """Saves a numpy array as a .npy file and a PyTorch tensor."""
        numpy_path = self.output_path / (filename + '.npy')
        torch_path = self.output_path / (filename + '.pt')

        np.save(numpy_path, data)
        torch.save(torch.from_numpy(data).float(), torch_path)
        logger.info(f"Solution saved in .npy and .pt formats: {numpy_path}, {torch_path}")


def average_solutions(solutions_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the element-wise average and standard deviation of a list of solutions."""
    if not solutions_list:
        return np.array([]), np.array([])

    # Pad solutions with NaNs to make them all the same length
    max_len = max(sol.shape[0] for sol in solutions_list)
    padded_solutions = np.array(
        [np.pad(sol, ((0, max_len - sol.shape[0]), (0, 0)), 'constant', constant_values=np.nan) for sol in
         solutions_list])

    # Calculate the mean and std, ignoring NaNs
    mean_solution = np.nanmean(padded_solutions, axis=0)
    std_solution = np.nanstd(padded_solutions, axis=0)

    return mean_solution, std_solution


def main():
    """Main function to execute the entire robot trajectory analysis workflow."""
    # --- Configuration ---
    DATA_PATH = Path(r'C:\Users\Ksenia\NSS\ODE_projects\robots\new_data')
    OUTPUT_PATH = DATA_PATH / 'output'
    PICKLE_FILENAME = 'data_00_330_[30_bots_PWM_10_15cw_15ccw_D_41cm].MP4.pickle'
    OBJECT_ID = 78
    FREQUENCY = 1
    MAX_POINTS = 2750
    NUM_SEGMENTS = 3
    NUM_MONTE_CARLO_SAMPLES = 10  # Number of different equation systems to solve

    CSV_FILENAMES = [f'output_{OBJECT_ID}_part_{i}.csv' for i in range(1, NUM_SEGMENTS + 1)]

    try:
        # --- 1. Data Loading and Preprocessing ---
        logger.info("--- Stage 1: Processing Robot Movement Data ---")
        data_processor = RobotDataProcessor(DATA_PATH)
        data_processor.load_data(PICKLE_FILENAME)
        data_processor.extract_coordinates()
        data_processor.normalize_all_coordinates()
        x_full, y_full, t_full = data_processor.get_trajectory(OBJECT_ID, FREQUENCY, MAX_POINTS)
        segments = data_processor.split_trajectory(t_full, x_full, y_full, num_segments=NUM_SEGMENTS)

        # --- 2. Equation Analysis and Coefficient Generation ---
        logger.info("--- Stage 2: Analyzing Equations ---")
        equation_analyzer = EquationAnalyzer(DATA_PATH)
        equation_analyzer.load_equation_data(CSV_FILENAMES)
        equation_analyzer.analyze_equation_patterns()
        equation_analyzer.normalize_and_filter_equations()

        # --- 3. Equation Construction and Solver setup ---
        logger.info("--- Stage 3: Preparing for Monte Carlo Simulations ---")
        equation_constructor = EquationConstructor()
        ode_solver = ODESolver()
        solution_manager = SolutionManager(OUTPUT_PATH)

        # Lists to store solutions from all Monte Carlo runs
        all_mc_solutions = []

        # --- 4. Monte Carlo Simulation Loop ---
        logger.info(f"--- Stage 4: Solving ODE System for {NUM_MONTE_CARLO_SAMPLES} Monte Carlo Samples ---")
        for i in range(NUM_MONTE_CARLO_SAMPLES):
            logger.info(f"--- Starting Monte Carlo Run {i + 1}/{NUM_MONTE_CARLO_SAMPLES} ---")

            # Generate new coefficients for this run
            monte_carlo_samples_df = equation_analyzer.generate_monte_carlo_samples(num_samples=1)

            # Find similar coefficients (or use the generated one if no similarity is found)
            similar_coefficients_df = equation_analyzer.find_similar_coefficients(monte_carlo_samples_df)
            if similar_coefficients_df.empty:
                similar_coefficients_df = monte_carlo_samples_df

            equation_systems = equation_constructor.construct_equation_systems(similar_coefficients_df)

            # Store coefficients for this run
            solution_manager.save_equation_coefficients(equation_systems, f'{OBJECT_ID}_mc_run_{i+1}')

            current_run_solutions = []
            for seg_idx, (t_seg, x_seg, y_seg) in enumerate(segments):
                x0, y0 = (x_seg[0], y_seg[0]) if seg_idx == 0 else (
                current_run_solutions[-1][-1, 0], current_run_solutions[-1][-1, 1])

                b_op_x = ode_solver.create_boundary_operator('x', 0, [None], t_seg[0], x0)
                b_op_y = ode_solver.create_boundary_operator('y', 1, [None], t_seg[0], y0)

                # Use the first equation set from the generated list
                equation_set = equation_systems[0]

                solution = ode_solver.solve_system(
                    equations={'x': equation_set[0], 'y': equation_set[1]},
                    grids=[torch.from_numpy(t_seg).float()],
                    boundary_conditions=[b_op_y, b_op_x]
                )

                if solution is not None:
                    current_run_solutions.append(solution)
                else:
                    logger.warning(f"Solution for segment {seg_idx + 1} failed. Skipping this run.")
                    current_run_solutions = []
                    break  # Break out of the segment loop for this run

            if current_run_solutions:
                combined_run_solution = np.vstack(current_run_solutions)
                all_mc_solutions.append(combined_run_solution)

        if not all_mc_solutions:
            logger.error("No successful Monte Carlo simulations were completed. Exiting.")
            return

        # --- 5. Averaging Solutions and Final Output ---
        logger.info("--- Stage 5: Averaging Solutions and Saving Results ---")
        avg_solution, std_solution = average_solutions(all_mc_solutions)

        # Save the average solution as a numpy array and tensor
        solution_manager.save_numpy_and_tensor(avg_solution, f'average_solution_{OBJECT_ID}')
        solution_manager.save_numpy_and_tensor(std_solution, f'std_deviation_solution_{OBJECT_ID}')

        # Save the combined average solution to a CSV for plotting
        df_avg = pd.DataFrame({'t': t_full[:len(avg_solution)], 'x': avg_solution[:, 0], 'y': avg_solution[:, 1]})
        df_avg.to_csv(OUTPUT_PATH / f'average_solution_{OBJECT_ID}_combined.csv', index=False)
        logger.info(f"Average solution saved to {OUTPUT_PATH / f'average_solution_{OBJECT_ID}_combined.csv'}")

        # Visualize the average solution with original data and uncertainty
        plt.figure(figsize=(12, 8))
        plt.plot(t_full, avg_solution[:, 0], label='x (average solution)')
        plt.plot(t_full, avg_solution[:, 1], label='y (average solution)')
        plt.fill_between(t_full[:len(avg_solution)], avg_solution[:, 0] - std_solution[:, 0],
                         avg_solution[:, 0] + std_solution[:, 0], alpha=0.2, label='x uncertainty')
        plt.fill_between(t_full[:len(avg_solution)], avg_solution[:, 1] - std_solution[:, 1],
                         avg_solution[:, 1] + std_solution[:, 1], alpha=0.2, label='y uncertainty')
        plt.plot(t_full, x_full, '--', label='x (original data)', alpha=0.7)
        plt.plot(t_full, y_full, '--', label='y (original data)', alpha=0.7)
        plt.title(f'Combined Average Solution for Object {OBJECT_ID}')
        plt.xlabel('Time')
        plt.ylabel('Coordinates')
        plt.legend()
        plt.savefig(OUTPUT_PATH / f'average_combined_solution_{OBJECT_ID}.png')
        plt.close()

        # Plot the standard deviation for x and y separately
        plt.figure(figsize=(12, 6))
        plt.plot(t_full[:len(std_solution)], std_solution[:, 0], label='x std dev')
        plt.plot(t_full[:len(std_solution)], std_solution[:, 1], label='y std dev')
        plt.title('Standard Deviation of Solutions over Time')
        plt.xlabel('Time')
        plt.ylabel('Standard Deviation')
        plt.legend()
        plt.savefig(OUTPUT_PATH / f'std_deviation_plot_{OBJECT_ID}.png')
        plt.close()

        # Calculate final MSE with the average solution
        avg_solution_len = min(len(avg_solution), len(x_full))
        mse_x = np.mean((avg_solution[:avg_solution_len, 0] - x_full[:avg_solution_len]) ** 2)
        mse_y = np.mean((avg_solution[:avg_solution_len, 1] - y_full[:avg_solution_len]) ** 2)
        logger.info(f"Final Mean Squared Error (vs. Average Solution): x={mse_x:.6f}, y={mse_y:.6f}")
        logger.info("Workflow completed successfully.")

    except Exception as e:
        logger.error(f"A critical error occurred in the main workflow: {e}", exc_info=True)


if __name__ == "__main__":
    main()
    

