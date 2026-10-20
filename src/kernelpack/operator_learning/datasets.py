"""Read-only adapters for Ram's MATLAB/Fekete benchmark files.

Arrays use examples x points x [time] x components. Fekete indices in the
original files are one-based; all public Python indices are zero-based.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

# name: folder, data file, Fekete file, input field, spatial count, nugget
_FILES = {
    "taylor_green": (
        "taylor_green",
        "data_exact_matt.mat",
        "taylor_green_fekete.mat",
        "init_velocity",
        500,
        1e-8,
    ),
    "taylor_green_time": (
        "taylor_green",
        "data_time.mat",
        "taylor_green_fekete.mat",
        "init_velocity",
        500,
        1e-8,
    ),
    "lid_cavity_flow": (
        "lid_cavity_flow",
        "data.mat",
        "lid_cavity_flow_fekete.mat",
        "init_velocity",
        1000,
        1e-6,
    ),
    "flow_cylinder_laminar": (
        "flow_cylinder",
        "data_laminar.mat",
        "flow_cylinder_laminar_fekete.mat",
        "init_velocity",
        1000,
        1e-6,
    ),
    "flow_cylinder_shedding": (
        "flow_cylinder",
        "data_shedding.mat",
        "flow_cylinder_shedding_fekete.mat",
        "init_velocity",
        1000,
        1e-6,
    ),
    "backward_facing_step": (
        "backward_facing_step",
        "data.mat",
        "backward_facing_step_fekete.mat",
        "init_velocity",
        1000,
        1e-6,
    ),
    "species_transport": (
        "species_transport",
        "data.mat",
        "species_transport_fekete.mat",
        "init_velocity",
        7000,
        1e-4,
    ),
    "forced_turb": (
        "forced_turb",
        "data.mat",
        "forced_turb_fekete.mat",
        "forcing",
        7000,
        1e-4,
    ),
    "merge_vortices_easier": (
        "merge_vortices_easier",
        "data.mat",
        "merge_vortices_easier_fekete.mat",
        "init_vorticity",
        500,
        1e-4,
    ),
    "buoyancy_cavity_flow": (
        "buoyancy_cavity_flow",
        "data.mat",
        "buoyancy_cavity_flow_fekete.mat",
        "init_temperature",
        5000,
        1e-4,
    ),
    "airfoil": ("airfoil", "data.mat", "airfoil_fekete.mat", "geom", 500, 1e-4),
}


def read_mat_fields(path, names, *, example_indices=None):
    """MAT v5/v7.3 numeric fields, preserving MATLAB axis order and empties."""
    import h5py

    if not h5py.is_hdf5(path):
        from scipy.io import loadmat

        result = loadmat(path, variable_names=list(names))
        result = {k: result[k] for k in names}
        if example_indices is not None:
            result = {
                k: (v if k == "points" else v[example_indices])
                for k, v in result.items()
            }
        return result
    result = {}
    with h5py.File(path, "r") as f:
        for name in names:
            value = f[name]
            if not isinstance(value, h5py.Dataset):
                raise ValueError(f"{name} must be a numeric array")
            if value.attrs.get("MATLAB_empty", 0):
                result[name] = np.empty((0, 0))
                continue
            if example_indices is None or name == "points":
                array = np.asarray(value)
            else:
                ids = np.asarray(example_indices, dtype=int)
                unique, restore = np.unique(ids, return_inverse=True)
                array = value[(slice(None),) * (value.ndim - 1) + (unique,)]
                array = array[..., restore]
            result[name] = array.T.copy()
    return result


@dataclass
class FastOLDataset:
    inputs: object  # CUDA JAX arrays; NumPy is used only by file decoders below.
    points: object
    velocity: object
    model_options: dict
    source_files: tuple[Path, Path]


def load_frozen_split(path):
    """Read Ram's saved train/validation/test IDs, converting once to zero base."""
    import h5py

    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            group = f["split"]
            split = {}
            for key in ("TrainIndices", "ValidationIndices", "TestIndices"):
                value = group[key]
                if h5py.check_dtype(ref=value.dtype):
                    value = f[value[0, 0]]
                split[key] = np.asarray(value).ravel()
    else:
        from scipy.io import loadmat

        split = loadmat(path, simplify_cells=True)["split"]
    result = {
        name: np.asarray(split[key], dtype=np.int64).ravel() - 1
        for name, key in [
            ("train", "TrainIndices"),
            ("validation", "ValidationIndices"),
            ("test", "TestIndices"),
        ]
    }
    if any(np.any(v < 0) or len(np.unique(v)) != len(v) for v in result.values()):
        raise ValueError("invalid split IDs")
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if np.intersect1d(result[a], result[b]).size:
            raise ValueError("saved split is not disjoint")
    return result


def load_fastol_dataset(
    root,
    name,
    *,
    output_points=None,
    input_points=None,
    representation="vector",
    example_indices=None,
    device=None,
):
    """Load the benchmark directly; no MATLAB executable or exported coefficients.

    ``example_indices`` address the loader's logical order. Forced turbulence
    first applies NumPy PCG64(seed=0), matching Ram's dataset.py. No periodic
    endpoint deletion is applied to the local spatial decoder.
    """
    import jax
    import jax.numpy as jnp
    if not jax.config.x64_enabled:
        raise ValueError("GPU dataset preprocessing requires jax_enable_x64=True")
    device=jax.devices("gpu")[0] if device is None else device
    if device.platform!="gpu":
        raise ValueError("FastOL dataset preprocessing requires a GPU")
    if name not in _FILES:
        raise ValueError(f"Unknown FastOL dataset {name}")
    if representation not in ("vector", "magnitude"):
        raise ValueError("representation must be 'vector' or 'magnitude'")
    folder, data_name, fekete_name, field, default_count, nugget = _FILES[name]
    root = Path(root)
    data_path = root / folder / data_name
    fekete_path = root / folder / fekete_name
    if not fekete_path.exists():
        fekete_path = root / name / fekete_name
    output_points = default_count if output_points is None else int(output_points)
    input_points = output_points if input_points is None else int(input_points)
    e = jnp.asarray(read_mat_fields(fekete_path, ["E"])["E"],device=device).ravel()
    if not bool(jnp.all(jnp.isfinite(e)) & jnp.all(e==jnp.floor(e))):
        raise ValueError("Fekete indices must be finite integers")
    e = e.astype(jnp.int32) - 1
    if min(output_points, input_points) < 1 or max(output_points, input_points) > len(
        e
    ):
        raise ValueError("requested spatial count exceeds Fekete ordering")
    names = [field, "points"] + (
        ["vel_7", "vel_8", "vel_9", "vel_10"]
        if name == "taylor_green_time"
        else ["velocity"]
    )
    physical_ids = example_indices
    if name == "forced_turb":
        import h5py

        if h5py.is_hdf5(data_path):
            with h5py.File(data_path, "r") as f:
                count = f[field].shape[-1]
        else:
            from scipy.io import whosmat

            count = next(
                shape[0] for key, shape, _ in whosmat(data_path) if key == field
            )
        permutation = np.random.default_rng(0).permutation(count)
        physical_ids = (
            permutation
            if example_indices is None
            else permutation[np.asarray(example_indices, dtype=int)]
        )
    raw = read_mat_fields(data_path, names, example_indices=physical_ids)
    raw={key:jnp.asarray(value,dtype=jnp.float64,device=device) for key,value in raw.items()}
    points = raw["points"]
    picked = e[:output_points]
    input_ids = e[:input_points]
    used=e[:max(output_points,input_points)]
    if not bool(jnp.all((used>=0)&(used<len(points)))):
        raise ValueError(
            "Fekete ordering is incompatible with the original point array"
        )
    if name == "taylor_green_time":
        velocity = jnp.stack([raw[k] for k in names[2:]], axis=2)
    else:
        velocity = raw["velocity"]
    velocity = velocity[:, picked]
    if velocity.ndim == 2:
        velocity = velocity[:, :, None]
    points = points[picked, : velocity.shape[-1]]
    input_data = raw[field]
    if input_data.ndim in (2, 3) and input_data.shape[1] > int(jnp.max(input_ids)):
        input_data = input_data[:, input_ids]
    if representation == "magnitude":
        if input_data.ndim != 3:
            raise ValueError("magnitude representation requires a vector field")
        inputs = jnp.linalg.norm(input_data, axis=2)
    elif input_data.ndim == 3:
        inputs = input_data.transpose(0, 2, 1).reshape(len(input_data), -1)
    elif input_data.ndim == 2:
        inputs = input_data
    else:
        raise ValueError("Input fields must be sampled scalar/vector numeric functions")
    options = {
        "nugget": nugget,
        "use_periodic": name
        in (
            "taylor_green",
            "taylor_green_time",
            "forced_turb",
            "merge_vortices_easier",
            "airfoil",
        ),
    }
    if name == "forced_turb":
        options.update(
            periodic_dims=[True, True, True], domain_bounds=[[0, 0, 0], [2 * np.pi] * 3]
        )
    if name == "airfoil":
        options["periodic_dims"] = [False, False, True]
    if name == "taylor_green_time":
        options["time_points"] = [0.7, 0.8, 0.9, 1.0]
        options["temporal_interpolation"] = "cubic_polynomial"
    if name in ("forced_turb","species_transport"):
        options.update(spatial_kernel="matern_multiscale",matern_base_length_factor=16.,
                       matern_num_modes=5,matern_gamma=8/3)
    return FastOLDataset(
        inputs,
        points,
        velocity,
        options,
        (data_path, fekete_path),
    )


def read_fastol_field_gpu(path, name, example_ids=None, *, device=None):
    """Decode only requested logical rows, transfer before field arithmetic.

    MAT v5 cannot decode partial named fields; its transient decoded host
    array is restricted to the requested rows before numerical delivery.
    """
    import h5py
    import jax
    import jax.numpy as jnp
    if not jax.config.x64_enabled:
        raise ValueError('GPU field decoding requires jax_enable_x64=True')
    device=jax.devices('gpu')[0] if device is None else device
    if device.platform!='gpu':
        raise ValueError('FastOL field preprocessing requires GPU arrays')
    if h5py.is_hdf5(path):
        with h5py.File(path,'r') as stream:
            dataset=stream[name]
            if example_ids is None:
                return jnp.asarray(dataset[()],device=device).T
            unique=sorted(set(example_ids))
            lookup={value:index for index,value in enumerate(unique)}
            restore=[lookup[value] for value in example_ids]
            raw=dataset[(slice(None),)*(dataset.ndim-1)+(unique,)]
            return jnp.asarray(raw,device=device).T[jnp.asarray(restore,dtype=jnp.int32,device=device)]
    from scipy.io import loadmat
    raw=loadmat(path,variable_names=[name])[name]
    return jnp.asarray(raw if example_ids is None else raw[example_ids],device=device)


def fastol_physical_indices(path, field, name, logical_ids):
    """Return physical I/O row IDs and Ram's optional archived permutation.

    PCG64 is identity metadata: changing its generator changes the benchmark
    examples. No field or covariance arithmetic is done on the host.
    """
    if name!='forced_turb':
        return list(logical_ids),None
    import h5py
    if h5py.is_hdf5(path):
        with h5py.File(path,'r') as stream:
            count=stream[field].shape[-1]
    else:
        from scipy.io import whosmat
        count=next(shape[0] for key,shape,_ in whosmat(path) if key==field)
    permutation=np.random.default_rng(0).permutation(count).tolist()
    return [permutation[index] for index in logical_ids],permutation


def load_fastol_paper_dataset(root,name,logical_ids,*,output_points=None,input_points=None,
                             representation='vector',device=None):
    """GPU preprocessing plus exact physical identities for a frozen split.

    Returns ``(FastOLDataset, physical_ids, loader_permutation)``. The ordinary
    public loader remains available; this adapter also exposes provenance IDs
    without storing or transferring unused target examples.
    """
    import jax
    import jax.numpy as jnp
    if not jax.config.x64_enabled:
        raise ValueError('GPU paper preprocessing requires jax_enable_x64=True')
    device=jax.devices('gpu')[0] if device is None else device
    if device.platform!='gpu':
        raise ValueError('GPU paper preprocessing requires a CUDA device')
    if name not in _FILES or representation not in ('vector','magnitude'):
        raise ValueError('Unknown FastOL dataset or input representation')
    folder,data_name,fekete_name,field,default_count,nugget=_FILES[name]
    data_path=Path(root)/folder/data_name
    fekete_path=Path(root)/folder/fekete_name
    if not fekete_path.exists():
        fekete_path=Path(root)/name/fekete_name
    output_points=default_count if output_points is None else int(output_points)
    input_points=output_points if input_points is None else int(input_points)
    if min(output_points,input_points)<1:
        raise ValueError('Spatial counts must be positive')
    actual,permutation=fastol_physical_indices(data_path,field,name,list(logical_ids))
    ordering=read_fastol_field_gpu(fekete_path,'E',device=device).reshape(-1)
    if max(output_points,input_points)>len(ordering):
        raise ValueError('Requested counts exceed the Fekete ordering')
    ordering=ordering[:max(output_points,input_points)]
    if not bool(jnp.all(jnp.isfinite(ordering))&jnp.all(ordering==jnp.floor(ordering))):
        raise ValueError('Invalid Fekete ordering')
    ordering=ordering.astype(jnp.int32)-1
    points=read_fastol_field_gpu(data_path,'points',device=device).astype(jnp.float64)
    if not bool(jnp.all((ordering>=0)&(ordering<len(points)))):
        raise ValueError('Fekete ordering does not address original points')
    picked,input_picked=ordering[:output_points],ordering[:input_points]
    names=('vel_7','vel_8','vel_9','vel_10') if name=='taylor_green_time' else ('velocity',)
    outputs=[read_fastol_field_gpu(data_path,key,actual,device=device).astype(jnp.float64)[:,picked] for key in names]
    velocity=jnp.stack(outputs,axis=2) if len(outputs)>1 else outputs[0]
    if velocity.ndim==2:
        velocity=velocity[:,:,None]
    raw=read_fastol_field_gpu(data_path,field,actual,device=device).astype(jnp.float64)
    if raw.ndim not in (2,3):
        raise ValueError('Paper inputs must be scalar/vector sampled functions')
    if raw.shape[1]>int(jnp.max(input_picked)):
        raw=raw[:,input_picked]
    if representation=='magnitude':
        if raw.ndim!=3:
            raise ValueError('Magnitude representation requires vector inputs')
        inputs=jnp.linalg.norm(raw,axis=2)
    else:
        inputs=jnp.transpose(raw,(0,2,1)).reshape(len(raw),-1) if raw.ndim==3 else raw
    options=dict(nugget=nugget,use_periodic=name in ('taylor_green','taylor_green_time','forced_turb','merge_vortices_easier','airfoil'))
    if name=='forced_turb':
        import math
        options.update(periodic_dims=[True]*3,domain_bounds=[[0.]*3,[2*math.pi]*3])
    if name=='airfoil':
        options['periodic_dims']=[False,False,True]
    if name=='taylor_green_time':
        options.update(time_points=[.7,.8,.9,1.],temporal_interpolation='cubic_polynomial')
    if name in ('forced_turb','species_transport'):
        options.update(spatial_kernel='matern_multiscale',matern_base_length_factor=16.,matern_num_modes=5,matern_gamma=8/3)
    return FastOLDataset(inputs,points[picked,:velocity.shape[-1]],velocity,options,(data_path,fekete_path)),actual,permutation
