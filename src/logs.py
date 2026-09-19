from pathlib import Path
from src.Config import PROJECT_ROOT, PROJECT_NAME
from datetime import datetime
import time
import json
import numpy
from numpy.typing import NDArray

from src._T_typing import (_T_root, _T_rel_path, _T_rel_target_path, _T_target_path, _T_path)


class logs:
    """
    Utility class for storing application data and informational logs.

    The class provides functionality for:
        - writing messages to log files and the console,
        - saving and loading JSON and NumPy data,
        - measuring execution time,
        - generating formatted reports,
        - handling project-relative paths and file names.
    """

    __logs_source_path: Path = PROJECT_ROOT / Path("docs/logs")
    assert __logs_source_path.exists(), FileExistsError("Logs path >>{__logs_source_path} not exists")
    __logs_name: str = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

    @staticmethod
    def create_name_by_datetime() -> str:
        """
        Creates a file name based on the current date and time.

        :return: Current date and time formatted as a string.
        """
        return datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

    @staticmethod
    def path(spec_path: str) -> Path:
        """
        Converts a string path into a `Path` object.

        :param spec_path: Path represented as a string.
        :return: Path object created from the provided path.
        """
        return Path(spec_path)

    __tags: dict[str, str] = {
        'i': "[INFO]",
        's': "... STEP: ",
        'd': "@ PROGRAM:",
        'e': "<ERROR>",
        'c': "[CHECK]",

        'p': "(PROCESS)",
        'p-s': "(PROCESS-START)",
        'p-e': "(PROCESS-END)",

        't': '[TIME]',
        'f-s': "(FILE-SAVE)",
        'f-l': "(FILE-LOAD)",

        'r': "=== REPORT ===",
    }
    __timer_count: float = 0

    class storage:
        """
        Provides methods for storing and loading application data.

        The storage interface is divided into `save` and `load` operations
        and uses project-relative paths.
        """

        @staticmethod
        def nametag(filename: str, tag: str) -> str:
            """
            Adds an optional tag to a file name.

            :param filename: Base file name.
            :param tag: Optional tag appended to the file name.
            :return: File name with the optional tag.
            """
            if tag:
                return f"{filename}_{tag}"
            else:
                return filename

        @staticmethod
        def create_name_by_datetime() -> str:
            """
            Creates a file name based on the current date and time.

            :return: Current date and time formatted as a string.
            """
            return datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

        class save:

            @classmethod
            def json(
                    cls, filename: str, project_relative_path: _T_rel_path, data: dict,
            ) -> None:
                """
                Saves a dictionary as a JSON file.

                :param filename: File name without the extension.
                :param project_relative_path: Path relative to PROJECT_ROOT.
                :param data: Dictionary to save.
                """
                try:
                    folder: _T_path = PROJECT_ROOT / project_relative_path
                    folder.mkdir(parents=True, exist_ok=True)

                    if not filename.endswith(".json"): filename += ".json"
                    file_path: _T_target_path = (folder / filename)

                    with open(file_path, mode="w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False, )

                    logs.print(
                        "f-s",
                        f"Data block in "
                        f"{PROJECT_NAME}>>{project_relative_path}/{filename}"
                    )

                except Exception as e:
                    logs.print("e", f"Can't save JSON file: {e}")
                    raise

            @classmethod
            def numpy(
                    cls,
                    filename: str,
                    project_relative_path: _T_rel_path,
                    data: NDArray,
            ) -> None:
                """
                Saves a NumPy array as a `.npy` file.

                :param filename: File name without the extension.
                :param project_relative_path: Path relative to PROJECT_ROOT.
                :param data: NumPy array to save.
                """
                try:
                    if not isinstance(data, numpy.ndarray):
                        raise TypeError("data must be numpy.ndarray")

                    folder: _T_path = PROJECT_ROOT / project_relative_path
                    folder.mkdir(parents=True, exist_ok=True)

                    if not filename.endswith(".npy"):  filename += ".npy"
                    file_path: _T_target_path = (folder / filename)

                    numpy.save(file_path, data)

                    logs.print(
                        "f-s",
                        f"NumPy array in "
                        f"{PROJECT_NAME}>>{project_relative_path}/{filename}"
                    )

                except Exception as e:
                    logs.print("e", f"Can't save NumPy array: {e}")
                    raise

            @classmethod
            def plot(
                    cls,
                    filename: str,
                    project_relative_path: _T_rel_path,
                    plt,
                    dpi: int = 300,
                    bbox_inches: str = "tight",
            ) -> None:
                """
                Saves a Matplotlib plot as a JPG image.

                :param filename: File name without the extension.
                :param project_relative_path: Path relative to PROJECT_ROOT.
                :param plt: Matplotlib pyplot module.
                :param dpi: Image resolution in dots per inch.
                :param bbox_inches: Bounding box option passed to Matplotlib.
                """
                try:
                    folder: _T_path = PROJECT_ROOT / project_relative_path
                    folder.mkdir(parents=True, exist_ok=True)

                    if not filename.endswith(".jpg"):
                        filename += ".jpg"

                    file_path: _T_target_path = folder / filename

                    plt.savefig(
                        file_path,
                        bbox_inches=bbox_inches,
                        dpi=dpi,
                    )

                    logs.print(
                        "f-s",
                        f"Plot in "
                        f"{PROJECT_NAME}>>{project_relative_path}/{filename}"
                    )

                except Exception as e:
                    logs.print("e", f"Can't save plot: {e}")
                    raise

        class load:

            @classmethod
            def json(
                    cls,
                    filename: str,
                    project_relative_path: _T_rel_path,
            ) -> dict:
                """
                Loads a dictionary from a JSON file.

                :param filename: File name with or without the extension.
                :param project_relative_path: Path relative to PROJECT_ROOT.
                :return: Dictionary loaded from the JSON file.
                """
                try:
                    if not filename.endswith(".json"): filename += ".json"

                    file_path: _T_target_path = (PROJECT_ROOT / project_relative_path / filename)

                    with open(file_path, mode="r", encoding="utf-8") as f:
                        data = json.load(f)

                    logs.print(
                        "f-l",
                        f"Data block from "
                        f"{PROJECT_NAME}>>{project_relative_path}/{filename}"
                    )

                    return data

                except Exception as e:
                    logs.print("e", f"Can't load JSON file: {e}")
                    raise

            @classmethod
            def numpy(
                    cls,
                    filename: str,
                    project_relative_path: _T_rel_path,
            ) -> NDArray:
                """
                Loads a NumPy array from a `.npy` file.

                :param filename: File name with or without the extension.
                :param project_relative_path: Path relative to PROJECT_ROOT.
                :return: NumPy array loaded from the file.
                """
                try:
                    if not filename.endswith(".npy"):  filename += ".npy"

                    file_path: _T_target_path = (PROJECT_ROOT / project_relative_path / filename)

                    data = numpy.load(file_path)

                    logs.print(
                        "f-l",
                        f"NumPy array from "
                        f"{PROJECT_NAME}>>{project_relative_path}/{filename}"
                    )
                    return data

                except Exception as e:
                    logs.print("e", f"Can't load NumPy array: {e}")
                    raise

    class stopwatch:
        """
        Utility class for measuring elapsed execution time.

        The stopwatch can be started, stopped, and restarted. The elapsed time
        is accumulated between consecutive `run()` and `stop()` calls.
        """

        def __init__(self, name: str):
            self.name: str = name
            self.elapsed: float = 0.0
            self._start_time: float | None = None

        def run(self) -> None:
            """
            Starts the stopwatch.

            If the stopwatch is already running, an error message is logged
            and the current measurement is preserved.
            """
            if self._start_time is not None:
                logs.print('e', f"Stopwatch: {self.name}; TIMER ALREADY RUNNING")
                return
            self._start_time = time.perf_counter()
            logs.print('t', f"Stopwatch: {self.name}; TIMER RUN")

        def stop(self) -> float:
            """
            Stops the stopwatch and updates the accumulated elapsed time.

            If the stopwatch is not running, the current elapsed time is
            returned without modification.

            :return: Total elapsed time in seconds.
            """
            if self._start_time is None:
                logs.print('e', f"Stopwatch: {self.name}; TIMER IS NOT RUNNING")
                return self.elapsed
            self.elapsed += time.perf_counter() - self._start_time
            self._start_time = None
            logs.print('t', f"Stopwatch: {self.name}; TIME COUNT: {self.elapsed:.6f} s")
            return self.elapsed

        def restart(self) -> None:
            """
            Resets the elapsed time and starts the stopwatch again.
            """
            self.elapsed = 0.0
            self._start_time = time.perf_counter()
            logs.print('t', f"Stopwatch: {self.name}; TIMER RESTART")

    # class timer:  ## <<== useless
    #     def __init__(self, name: str):
    #         self.name: str = name
    #         self.timer_count: float = 0.0
    #
    #     def timer_on(self):
    #         logs.print('t', f"Special Timer: {self.name}; TIMER ON")
    #         self.timer_count = time.perf_counter()
    #
    #     def timer_off(self):
    #         self.timer_count = time.perf_counter() - self.timer_count
    #         logs.print('t', f"Special Timer: {self.name}; TIME COUNT: {self.timer_count} s")
    # @classmethod
    # def timer_on(cls):
    #     cls.print('t', "TIMER ON")
    #     cls.__timer_count = time.perf_counter()
    #
    # @classmethod
    # def timer_off(cls):
    #     cls.__timer_count = time.perf_counter() - cls.__timer_count
    #     cls.print('t', f"TIME COUNT: {cls.__timer_count} s")

    @staticmethod
    def list(object: list) -> str:
        """
        Formats a list as a multi-line string.

        Each list element is placed on a separate line.

        :param object: List of values to format.
        :return: Formatted multi-line string.
        """
        ret = "\n"
        for o in object:
            ret += f"...\t{o}\n"
        return ret

    @staticmethod
    def dict(object: dict) -> str:
        """
        Formats a dictionary as a multi-line string.

        Dictionary keys and values are aligned into separate columns.

        :param object: Dictionary to format.
        :return: Formatted multi-line string.
        """
        ret = "\n"
        for k, v in object.items():
            ret += f"...\t{k:<20}\t{v}\n"
        return ret

    @classmethod
    def endline(cls, console_display=True):
        """
        Adds an empty line to the current log file and optionally to the console.

        :param console_display: If `True`, also prints an empty line to the console.
        """
        with open(cls.__logs_source_path / f"{cls.__logs_name}.log", "+a") as f:
            f.write("\n")
            if console_display: print()

    @classmethod
    def highlines(cls, console_display=True):
        """
        Adds a separator line to the current log file and optionally to the console.

        :param console_display: If `True`, also prints the separator to the console.
        """
        with open(cls.__logs_source_path / f"{cls.__logs_name}.log", "+a") as f:
            f.write("\n")
            if console_display: print("=====" * 21)

    @classmethod
    def print(cls, tag: str, message: str, console_display=True):
        """
        Writes a tagged message to the log file and optionally to the console.

        Available tags:

            'i': "[INFO]",
            's': "... STEP: ",
            'd': "@ PROGRAM:",
            'e': "<ERROR>",
            'c': "[CHECK]",

            'p': "(PROCESS)",
            'p-s': "(PROCESS-START)",
            'p-e': "(PROCESS-END)",

            't': '[TIME]',
            'f-s': "(FILE-SAVE)",
            'f-l': "(FILE-LOAD)",

            'r': "=== REPORT ==="

        :param tag: Tag identifying the type of log message.
        :param message: Message to write to the log.
        :param console_display: If `True`, also prints the message to the console.
        """

        # Get the corresponding tag label from the tag dictionary.
        t = cls.__tags.get(tag, "[...]")

        # Create the formatted log message.
        comu = f"{t}\t{message}"

        with open(cls.__logs_source_path / f"{cls.__logs_name}.log", "+a") as f:
            f.write("\n" + comu)
            if console_display: print(comu)

    class report:
        """
        Utility class for creating and displaying tabular reports.
        """

        # Default display precision for supported data types.
        _accuracy = {
            int: 3,
            float: 3,
            str: -1,
            bool: -1
        }

        def __init__(self, columns: dict[str, type]):

            # Store column names while preserving their dictionary order.
            self.col_names: list[str] = list(columns.keys())

            # Store the type and display precision for each column.
            self.types: dict[str, dict] = {
                name: {'type': type_, 'accuracy': self._accuracy.get(type_, -1)}
                for name, type_ in columns.items()
            }

            # Store data separately for each column.
            self.data: dict[str, list] = {name: [] for name in columns}

            self.generated: str = "Empty"

            logs.print('r', "creat new raport!")
            logs.print('i', logs.dict(self.types))

        def set_type(self, key: str, type_: type, accuracy: int = None):
            """
            Updates the type and display precision of a report column.

            :param key: Name of the column to update.
            :param type_: Data type assigned to the column.
            :param accuracy: Optional number of decimal places used when
                displaying numeric values.
            """

            # Check whether the specified column exists.
            if key not in self.data: raise KeyError(f"Unknown column: {key}")

            # Set the column type and display precision.
            self.types[key] = {
                'type': type_,
                'accuracy': self._accuracy.get(type_, -1) if accuracy is None else accuracy
            }

        def put(self, key: str, value):
            """
            Adds a value to the specified report column.

            :param key: Name of the column to which the value is added.
            :param value: Value to store.
            """

            # Check whether the specified column exists.
            if key not in self.data: raise KeyError(f"Unknown column: {key}")

            # Add the value to the corresponding column.
            self.data[key].append(value)

        def get(self, key: str) -> list:
            """
            Returns all values stored in a report column.

            :param key: Name of the column to retrieve.
            :return: List containing all values stored in the column.
            """

            # Check whether the specified column exists.
            if key not in self.data: raise KeyError(f"Unknown column: {key}")
            return self.data[key]

        def show(self) -> str:
            """
            Formats and displays the report as a text-based table.

            Column widths are calculated dynamically based on both the column
            names and their stored values. Numeric values are rounded according
            to the configured column precision.
            """

            # Determine the initial column width from the column headers.
            widths = {key: len(key) for key in self.col_names}

            # Adjust column widths to fit the stored values.
            for key in self.col_names:
                for value in self.data[key]:
                    accuracy = self.types[key]['accuracy']
                    type_ = self.types[key]['type']

                    # Round numeric values according to the configured precision.
                    if accuracy >= 0 and type_ in (int, float):
                        value = round(value, accuracy)

                    widths[key] = max(widths[key], len(str(value)))

            # Create the horizontal table separator.
            line = "+" + "+".join("-" * (widths[key] + 2) for key in self.col_names) + "+"

            ret = "\n" + line + "\n"

            # Create the table header.
            ret += "|" + "|".join(
                f" {key:<{widths[key]}} "
                for key in self.col_names
            ) + "|\n"

            # Create the separator between the header and the data rows.
            ret += line + "\n"

            # The number of rows is determined by the longest column.
            rows = max((len(values) for values in self.data.values()), default=0)

            # Create each data row.
            for i in range(rows):
                row = []

                for key in self.col_names:
                    value = self.data[key][i]
                    accuracy = self.types[key]['accuracy']
                    type_ = self.types[key]['type']

                    # Round numeric values before displaying them.
                    if accuracy >= 0 and type_ in (int, float):
                        value = round(value, accuracy)

                    row.append(f" {str(value):<{widths[key]}} ")

                ret += "|" + "|".join(row) + "|\n"

            # Add the bottom table separator.
            ret += line

            self.generated = ret
            logs.print('r', ret)

        def saveSelf(
                    self, filename: str, project_relative_path: _T_rel_path,
            ) -> None:
                """
                Saves a dictionary as a JSON file.

                :param filename: File name without the extension.
                :param project_relative_path: Path relative to PROJECT_ROOT.
                :param data: Dictionary to save.
                """
                try:
                    folder: _T_path = PROJECT_ROOT / project_relative_path
                    folder.mkdir(parents=True, exist_ok=True)

                    if not filename.endswith(".raport"): filename += ".raport"
                    file_path: _T_target_path = (folder / filename)

                    with open(file_path, mode="w", encoding="utf-8") as f:
                        f.write(self.generated)

                    logs.print(
                        "f-s",
                        f"Report in "
                        f"{PROJECT_NAME}>>{project_relative_path}/{filename}"
                    )

                except Exception as e:
                    logs.print("e", f"Can't save JSON file: {e}")
                    raise


if __name__ == "__main__":
    logs.print("I", "Hello World")

# class save:
#
#     console_display = True
#
#     @classmethod
#     def json(cls, filename: str, project_path: Path, data: dict, ):
#         """
#         Saves a dictionary as a UTF-8 encoded JSON file.
#
#         :param filename: File name without the extension.
#         :param project_path: Path relative to PROJECT_ROOT.
#         :param data: Dictionary to save.
#         :return: None.
#         """
#         try:
#
#             file = PROJECT_ROOT / project_path / Path(f"{filename}.json")
#             (PROJECT_ROOT / project_path).mkdir(parents=True, exist_ok=True)
#
#             with open(file, mode="w", encoding="utf-8") as f:
#                 json.dump(data, f, indent=4, ensure_ascii=False)
#
#             logs.print('f-s', f"Data block in {PROJECT_NAME}>>{project_path}/{filename}.json")
#         except Exception as e:
#             logs.print('e', f"Cant asve file{e}")
#
#     @classmethod
#     def numpy(cls, filename: str, project_path: Path, data):
#         """
#         Saves a NumPy array to a `.npy` file.
#
#         :param filename: File name without the extension.
#         :param project_path: Path relative to PROJECT_ROOT.
#         :param data: NumPy array to save.
#         """
#         try:
#             if not isinstance(data, numpy.ndarray):
#                 raise TypeError("data must numpy.ndarray")
#
#             file = PROJECT_ROOT / project_path / Path(f"{filename}.npy")
#             (PROJECT_ROOT / project_path).mkdir(parents=True, exist_ok=True)
#
#             numpy.save(file, data)
#
#             logs.print('f-s', f"Saved NumPy array in {PROJECT_NAME}>>{project_path}/{filename}.npy")
#
#         except Exception as e:
#             logs.print('e', f"Can't save NumPy array: {e}")
#
# class load:
#
#     console_display = True
#
#     @classmethod
#     def json(cls, filename: str, project_path: Path) -> dict:
#         """
#         Loads a JSON file from the specified project path using UTF-8 encoding.
#
#         :param filename: File name with or without the `.json` extension.
#         :param project_path: Path relative to PROJECT_ROOT.
#         :return: Dictionary loaded from the JSON file.
#         """
#         try:
#             if not filename.endswith(".json"): filename += ".json"
#
#             file = PROJECT_ROOT / project_path / filename
#
#             with open(file, mode="r", encoding="utf-8") as f:
#                 data = json.load(f)
#
#             logs.print('f-l', f"Data block from {PROJECT_NAME}>>{project_path}/{filename}")
#
#             return data
#
#         except Exception as e:
#             logs.print('e', f"Can't load file: {e}")
#             raise
#
#     @classmethod
#     def numpy(cls, filename: str, project_path: Path) -> NDArray:
#         """
#         Loads a NumPy array from a `.npy` file.
#
#         :param filename: File name with or without the extension.
#         :param project_path: Path relative to PROJECT_ROOT.
#         :return: NumPy array loaded from the file.
#         """
#         try:
#             if not filename.endswith(".npy"): filename += ".npy"
#
#             file = PROJECT_ROOT / project_path / filename
#
#             data = numpy.load(file)
#
#             logs.print('f-l', f"Loaded NumPy array from {PROJECT_NAME}>>{project_path}/{filename}")
#
#             return data
#
#         except Exception as e:
#             logs.print('e', f"Can't load NumPy array: {e}")
#             raise
