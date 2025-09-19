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
        scaled_values = scaler.fit_transform(self.df.values)
        normalized_df = pd.DataFrame(scaled_values, columns=self.df.columns, index=self.df.index).fillna(0)

        self.df = normalized_df.loc[(self.df.sum(axis=1) != -1)]
        self.df = self.df.loc[:, (self.df != 0).any(axis=0)]
        logger.info(f"Filtered equation data shape: {self.df.shape}")

    def generate_monte_carlo_samples(self, num_samples: int = 100) -> None:
        """Generates new coefficient samples using a Monte Carlo method."""
        self.monte_carlo_samples = pd.DataFrame(np.zeros((num_samples, len(self.df.columns))),
                                                columns=self.df.columns)

        for column in self.df.columns:
            mean_val, std_dev = self.df[column].mean(), self.df[column].std()
            min_val, max_val = self.df[column].min(), self.df[column].max()

            for i in range(num_samples):
                current_mean = self.df[column].iloc[i % len(self.df)]
                noise = np.random.normal(loc=0, scale=std_dev * 0.01)
                sample = np.clip(current_mean + noise, min_val, max_val)
                self.monte_carlo_samples.loc[i, column] = sample

        logger.info(f"Generated Monte Carlo samples with shape {self.monte_carlo_samples.shape}")

    def find_similar_coefficients(self, threshold: float = 1.0) -> pd.DataFrame:
        """Finds generated coefficients that are similar to the original ones."""
        if self.df.empty or self.monte_carlo_samples.empty:
            raise ValueError("DataFrames for comparison are empty.")

        distances = pairwise_distances(self.df, self.monte_carlo_samples)
        similar_rows_indices = np.where(distances < threshold)[1]
        similar_coefficients_df = self.monte_carlo_samples.iloc[np.unique(similar_rows_indices)]

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
            raise


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
        # Assuming logger is defined elsewhere
        # logger.info(f"Solution segment saved to {filename}")

    def save_combined_solution(self, solutions: List[np.ndarray], time_points: np.ndarray, object_id: int) -> None:
        """Saves the combined trajectory solution to a CSV file."""
        combined_solution = np.vstack(solutions)
        df = pd.DataFrame({'t': time_points, 'x': combined_solution[:, 0], 'y': combined_solution[:, 1]})
        filename = self.output_path / f'solution_{object_id}_combined.csv'
        df.to_csv(filename, index=False)
        # Assuming logger is defined elsewhere
        # logger.info(f"Combined solution saved to {filename}")

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
        # Assuming logger is defined elsewhere
        # logger.info(f"Equation coefficients saved to {filename}")

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
        equation_analyzer.generate_monte_carlo_samples()
        similar_coefficients_df = equation_analyzer.find_similar_coefficients()

        if similar_coefficients_df.empty:
            logger.warning("No similar coefficients were found. Using the mean of original coefficients as a fallback.")
            similar_coefficients_df = pd.DataFrame([equation_analyzer.df.mean()], columns=equation_analyzer.df.columns)

        # --- 3. Equation Construction ---
        logger.info("--- Stage 3: Constructing Equations ---")
        equation_constructor = EquationConstructor()
        equation_systems = equation_constructor.construct_equation_systems(similar_coefficients_df)

        # --- 4. Solving ODE Systems for Each Segment ---
        logger.info("--- Stage 4: Solving ODE System for All Segments ---")
        ode_solver = ODESolver()
        solution_manager = SolutionManager(OUTPUT_PATH)
        solution_manager.save_equation_coefficients(equation_systems, OBJECT_ID)

        all_solutions = []
        for seg_idx, (t_seg, x_seg, y_seg) in enumerate(segments):
            logger.info(f"Solving segment {seg_idx + 1}/{NUM_SEGMENTS}...")

            x0, y0 = (x_seg[0], y_seg[0]) if seg_idx == 0 else (all_solutions[-1][-1, 0], all_solutions[-1][-1, 1])

            b_op_x = ode_solver.create_boundary_operator('x', 0, [None], t_seg[0], x0)
            b_op_y = ode_solver.create_boundary_operator('y', 1, [None], t_seg[0], y0)

            equation_set = equation_systems[seg_idx % len(equation_systems)]

            # This line is correct. equation_set is a tuple, so use integer indices.
            solution = ode_solver.solve_system(
                equations={'x': equation_set[0], 'y': equation_set[1]},
                grids=[torch.from_numpy(t_seg).float()],
                boundary_conditions=[b_op_y, b_op_x]
            )
            all_solutions.append(solution)
            solution_manager.save_solution(solution, t_seg, OBJECT_ID, seg_idx)

            plt.figure(figsize=(10, 6))
            plt.plot(t_seg, solution[:, 0], label='x (solved)')
            plt.plot(t_seg, solution[:, 1], label='y (solved)')
            plt.plot(t_seg, x_seg, '--', label='x (original)')
            plt.plot(t_seg, y_seg, '--', label='y (original)')
            plt.title(f'Solution for Segment {seg_idx + 1} (Object {OBJECT_ID})')
            plt.xlabel('Time')
            plt.ylabel('Coordinates')
            plt.legend()
            plt.savefig(OUTPUT_PATH / f'solution_segment_{OBJECT_ID}_{seg_idx + 1}.png')
            plt.close()

        # --- 5. Saving and Visualizing Combined Solution ---
        logger.info("--- Stage 5: Finalizing and Saving Combined Solution ---")
        solution_manager.save_combined_solution(all_solutions, t_full, OBJECT_ID)

        combined_solution = np.vstack(all_solutions)
        plt.figure(figsize=(12, 8))
        plt.plot(t_full, combined_solution[:, 0], label='x (full solution)')
        plt.plot(t_full, combined_solution[:, 1], label='y (full solution)')
        plt.plot(t_full, x_full, '--', label='x (original data)', alpha=0.7)
        plt.plot(t_full, y_full, '--', label='y (original data)', alpha=0.7)
        plt.title(f'Combined Solution for Object {OBJECT_ID}')
        plt.xlabel('Time')
        plt.ylabel('Coordinates')
        plt.legend()
        plt.savefig(OUTPUT_PATH / f'combined_solution_{OBJECT_ID}.png')
        plt.close()

        mse_x = np.mean((combined_solution[:, 0] - x_full) ** 2)
        mse_y = np.mean((combined_solution[:, 1] - y_full) ** 2)
        logger.info(f"Final Mean Squared Error: x={mse_x:.6f}, y={mse_y:.6f}")
        logger.info("Workflow completed successfully.")

    except Exception as e:
        logger.error(f"A critical error occurred in the main workflow: {e}", exc_info=True)


if __name__ == "__main__":
    main()

    