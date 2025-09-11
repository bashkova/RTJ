import os
import math
import re
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import dill as pickle
import matplotlib.pyplot as plt
from matplotlib import image
import plotly
import plotly.graph_objects as go
import plotly.express as px
import torch
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import pairwise_distances

import epde
import epde.loader as Loader
from epde import EpdeSearch
from epde.integrate import OdeintAdapter, SolverAdapter
from epde.integrate.pinn_integration import SolverAdapter


class DataProcessor:
    """CLASS FOR DATA PREPROCESSING"""

    def __init__(self, data: List):
        self.data = data
        self.coord_data = {}
        self.normalized_coord_data = {}

    def extract_data(self) -> Tuple[Dict, Dict]:
        """EXTRACTING RAW DATA"""
        keys_list = [inner_list[0] for inner_list in self.data[0]]
        dict_variable = {i: [] for i in keys_list}
        coord_data = {}

        for inner_list in self.data:
            for item in inner_list:
                key = item[0]
                value = item[1]
                coordinates = item[2]

                if key in dict_variable:
                    dict_variable[key].append(value)
                else:
                    dict_variable[key] = [value]

                if key in coord_data:
                    coord_data[key].append(coordinates)
                else:
                    coord_data[key] = [coordinates]

        self.coord_data = coord_data
        return dict_variable, coord_data

    @staticmethod
    def normalize_to_normal_distribution(data: np.ndarray) -> np.ndarray:
        if np.min(data) <= 0:
            data = data - np.min(data) + 1e-6

        transformed_data, _ = stats.boxcox(data)
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(transformed_data.reshape(-1, 1)).flatten()

        return normalized_data

    def normalize_all_coordinates(self) -> Dict:
        """NORMALYZING DATA"""
        all_x, all_y = [], []

        for points in self.coord_data.values():
            points_array = np.array(points)
            all_x.extend(points_array[:, 0])
            all_y.extend(points_array[:, 1])

        x_normalized = self.normalize_to_normal_distribution(np.array(all_x))
        y_normalized = self.normalize_to_normal_distribution(np.array(all_y))

        idx = 0
        for obj_id, points in self.coord_data.items():
            points_array = np.array(points)
            num_points = len(points_array)

            obj_x_norm = x_normalized[idx:idx + num_points]
            obj_y_norm = y_normalized[idx:idx + num_points]

            self.normalized_coord_data[obj_id] = np.column_stack((obj_x_norm, obj_y_norm))
            idx += num_points

        return self.normalized_coord_data


class MonteExp:
    """MONTE-CARLO SAMPLING STATISTICS CLASS"""

    def __init__(self, samples: list):
        self._samples = samples
        self._stats = [{'mean': np.mean(sample), 'std': np.std(sample)} for sample in samples]

    def discover_system_equation(self, epde_search_obj: EpdeSearch,
                                 additional_tokens: List, dimensionality: int = 0) -> Any:
        """DISCOVERYNG SYSTEMS WITH EPDE"""
        factors_max_number = {'factors_num': [1, 2], 'probas': [0.85, 0.15]}

        trig_tokens = epde.TrigonometricTokens(freq=(0.999, 1.001), dimensionality=dimensionality)
        grid_tokens = epde.GridTokens(['t'], dimensionality=dimensionality, max_power=2)

        epde_search_obj.fit(
            data=self._samples,
            variable_names=['x', 'y'],
            max_deriv_order=2,
            equation_terms_max_number=3,
            data_fun_pow=1,
            additional_tokens=additional_tokens + [trig_tokens, grid_tokens],
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=(1e-10, 1)
        )

        return epde_search_obj.equations(only_print=False, only_str=False, num=2)


class EquationProcessor:
    """CLASS FOR ANALYZING EQUATIONS"""

    def __init__(self):
        self.regex = re.compile(r', freq:\s\d\S\d+')

    @staticmethod
    def dict_update(d_main: Dict, term: str, coeff: float, k: int) -> Dict:
        """UPDATING DICTIONARIES FOR EQUATIONS"""
        str_t = '_r' if '_r' in term else ''
        arr_term = re.sub('_r', '', term).split(' * ')

        perm_set = list(itertools.permutations(range(len(arr_term))))
        structure_added = False

        for p_i in perm_set:
            temp = " * ".join([arr_term[i] for i in p_i]) + str_t
            if temp in d_main:
                if k - len(d_main[temp]) >= 0:
                    d_main[temp] += [0 for _ in range(k - len(d_main[temp]))] + [coeff]
                else:
                    d_main[temp][-1] += coeff
                structure_added = True

        if not structure_added:
            d_main[term] = [0 for _ in range(k)] + [coeff]

        return d_main

    def equation_table(self, k: int, equation, dict_main: Dict, dict_right: Dict) -> List[Dict]:
        """CREATING EQUATION TABLES"""
        equation_s = equation.structure
        equation_c = equation.weights_final
        text_form_eq = self.regex.sub('', equation.text_form)

        flag = False
        for t_eq in equation_s:
            term = self.regex.sub('', t_eq.name)
            for t in range(len(equation_c)):
                c = equation_c[t]
                if f'{c} * {term} +' in text_form_eq:
                    dict_main = self.dict_update(dict_main, term, c, k)
                    equation_c = np.delete(equation_c, t)
                    break
                elif f'+ {c} =' in text_form_eq:
                    dict_main = self.dict_update(dict_main, "C", c, k)
                    equation_c = np.delete(equation_c, t)
                    break
            if f'= {term}' == text_form_eq[text_form_eq.find('='):] and not flag:
                flag = True
                dict_main = self.dict_update(dict_main, term, -1., k)

        return [dict_main, dict_right]

    def object_table(self, res: List, variable_names: List[str],
                     table_main: List[Dict], k: int, title: str) -> Tuple[List[Dict], int]:
        """CREATING OBJECT FOR TABLES"""

        def filter_func(*args, **kwargs):
            return True

        for list_SoEq in res:
            for SoEq in list_SoEq:
                if filter_func(SoEq, variable_names):
                    for n, value in enumerate(variable_names):
                        gene = SoEq.vals.chromosome.get(value)
                        table_main[n][value] = self.equation_table(
                            k, gene.value, *table_main[n][value]
                        )
                    k += 1
        return table_main, k

    def preprocessing_table(self, variable_name: List[str],
                            table_main: List[Dict], k: int) -> pd.DataFrame:
        """PREPROCESSING FOR CREATING DATAFRAME"""
        data_frame_total = pd.DataFrame()

        for dict_var in table_main:
            for var_name, list_structure in dict_var.items():
                general_dict = {}
                for structure in list_structure:
                    general_dict.update(structure)
                dict_var[var_name] = general_dict

        for dict_var in table_main:
            for var_name, general_dict in dict_var.items():
                for key, value in general_dict.items():
                    if len(value) < k:
                        general_dict[key] = value + [0. for _ in range(k - len(value))]

        data_frame_main = [{i: pd.DataFrame()} for i in variable_name]

        for n, dict_var in enumerate(table_main):
            for var_name, general_dict in dict_var.items():
                data_frame_main[n][var_name] = pd.DataFrame(general_dict)

        for n, var_name in enumerate(variable_name):
            data_frame_temp = data_frame_main[n].get(var_name).copy()
            list_columns = [f'{col}_{var_name}' for col in data_frame_temp.columns]
            data_frame_temp.columns = list_columns
            data_frame_total = pd.concat([data_frame_total, data_frame_temp], axis=1)

        return data_frame_total


def main():
    file_path = Path(r'C:\Users\Ksenia\NSS\ODE_projects\robots\new_data')
    pickle_file = file_path / 'data_00_330_[30_bots_PWM_10_15cw_15ccw_D_41cm].MP4.pickle'

    with open(pickle_file, 'rb') as f:
        raw_data = pickle.load(f)

    # Обработка данных
    processor = DataProcessor(raw_data)
    dict_variable, coord_data = processor.extract_data()
    normalized_coord_data = processor.normalize_all_coordinates()

    # CHANGE OBJECT_ID FOR ANALYZING OTHER ROBOT
    object_id = 79
    if object_id not in normalized_coord_data:
        print(f"Object ID {object_id} not found in data")
        return

    points = np.array(normalized_coord_data[object_id])
    total_points = len(points)

    output_dir = file_path / 'output'
    output_dir.mkdir(exist_ok=True)
    pickle_dir = output_dir / 'pickle_objects'
    pickle_dir.mkdir(exist_ok=True)

    # CUTTING FOR THREE PARTS
    part_size = total_points // 3
    boundaries = [0, part_size, 2 * part_size, total_points]

    systems_equations = []
    systems_tables = []
    equation_processor = EquationProcessor()
    variable_names = ['x', 'y']

    # CONTROLLING EACH PART
    for part_idx in range(3):
        try:
            start_idx = boundaries[part_idx]
            end_idx = boundaries[part_idx + 1]

            part_points = points[start_idx:end_idx]
            part_time = np.arange(len(part_points))

            x_data = part_points[:, 0]
            y_data = part_points[:, 1]
            t_data = part_time

            epde_search_obj_system = EpdeSearch(
                use_solver=False,
                boundary=10,
                coordinate_tensors=[t_data]
            )
            epde_search_obj_system.set_preprocessor(
                default_preprocessor_type='poly',
                preprocessor_kwargs={'use_smoothing': True}
            )

            monte_exp_system = MonteExp([x_data, y_data])

            part_equations = monte_exp_system.discover_system_equation(
                epde_search_obj_system, additional_tokens=[], dimensionality=0
            )

            systems_equations.append(part_equations)

            # PREPROCESSING
            table_main = [{i: [{}, {}]} for i in variable_names]
            k = 0

            vals_flat = []
            for elem in part_equations:
                vals_flat.extend(elem)

            table_main, k = equation_processor.object_table(
                [vals_flat], variable_names, table_main, k, ''
            )
            systems_tables.append(table_main)

            print(f"\nSystem of DE {part_idx + 1} (startpoint {start_idx}-{end_idx - 1}):")
            for i, eq in enumerate(part_equations):
                print(f"EQ {i + 1}:")
                print(eq)

            # SAVE TABLE
            frame_main = equation_processor.preprocessing_table(variable_names, table_main, k)
            output_filename = f'output_{object_id}_part_{part_idx + 1}.csv'
            output_path = output_dir / output_filename
            frame_main.to_csv(output_path, sep=',', encoding='utf-8')
            print(f"Saved: {output_path}")

            loader = Loader.EPDELoader()
            equations_filename = f'equations_{object_id}_part_{part_idx + 1}.pickle'
            equations_path = pickle_dir / equations_filename
            loader.save(obj=part_equations, filename=equations_path)
            print(f"Saved equations: {equations_path}")

            pool_filename = f'pool_{object_id}_part_{part_idx + 1}.pickle'
            pool_path = pickle_dir / pool_filename
            loader.save(obj=epde_search_obj_system.pool, filename=pool_path)
            print(f"Saved pool: {pool_path}")

        except Exception as e:
            print(f"ERROR FOR part {part_idx + 1}: {e}")
            output_filename = f'output_{object_id}_part_{part_idx + 1}.csv'
            output_path = output_dir / output_filename
            pd.DataFrame().to_csv(output_path)
            print(f"Created empty file: {output_path}")


if __name__ == "__main__":
    main()