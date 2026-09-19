"""Run the breathing-sphere moving-surface ADR manufactured problem."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).parents[1] / "scripts" / "moving_surface_adr_breathing_sphere.py"
    runpy.run_path(str(script), run_name="__main__")
