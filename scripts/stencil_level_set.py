"""Attach the implicit layer of a paper cloud to its domain for boundary-aware stencil selection.

The layer is rebuilt exactly as scripts/generate_paper_clouds.py built it for the cloud
(base fit at h = 0.16, fit_resolution 41, CFPU max_sites 512, feature_grade_factor 2, band far field).
"""
from kernelpack.geometry import CFPUOptions
from cfpu_refinement_study import initial_geometry, resample

_BASE = {}


def implicit_layer(geometry, h, seed=11, fit_resolution=41, max_sites=512):
    if (geometry, fit_resolution) not in _BASE:
        _BASE[(geometry, fit_resolution)] = initial_geometry(geometry, .16, fit_resolution=fit_resolution)[0]
    surface = resample(_BASE[(geometry, fit_resolution)], geometry, h, seed=seed, fit_resolution=fit_resolution)
    surface.build_level_set(options=CFPUOptions(feature_grade_factor=2, max_sites=max_sites, farfield="band"))
    return surface.level_set.cfpu_model


def attach(domain, geometry, h, seed=11):
    domain.set_stencil_level_set(implicit_layer(geometry, h, seed), h)
    return domain
